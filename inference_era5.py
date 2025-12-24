# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ERA5降尺度推理脚本

支持自动patch切分：当输入尺寸与模型patch尺寸不匹配时，
自动将输入切分成多个patch进行推理，然后拼接还原。

输入: 文件夹，包含多个单时间戳的ERA5低分辨率nc文件
输出: 对应的高分辨率nc文件

用法:
    python inference_era5.py \
        --input_dir <输入文件夹> \
        --output_dir <输出文件夹> \
        --reg_ckpt <回归模型检查点> \
        --res_ckpt <扩散模型检查点> \
        --stats_path <统计信息json文件> \
        --patch_shape 448 448 \
        --overlap_pix 32
"""

import argparse
import datetime
import json
import math
import os
from functools import partial
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import xarray as xr
import netCDF4 as nc
import cftime
from tqdm import tqdm

from physicsnemo import Module
from physicsnemo.utils.logging import PythonLogger
from physicsnemo.models.diffusion.sampling import deterministic_sampler
from physicsnemo.models.diffusion.corrdiff_utils import regression_step, diffusion_step
from physicsnemo.models.diffusion.patching import GridPatching2D


class ERA5SingleTimeReader:
    """读取单时间戳ERA5 nc文件"""

    def __init__(
        self,
        nc_path: str,
        input_variables: Optional[List[str]] = None,
        stats: Optional[Dict] = None,
    ):
        """
        Parameters
        ----------
        nc_path : str
            单时间戳nc文件路径
        input_variables : List[str], optional
            输入变量列表，None则自动检测
        stats : Dict, optional
            归一化统计信息
        """
        self.nc_path = nc_path

        with xr.open_dataset(nc_path) as ds:
            # 获取变量名（排除坐标变量）
            coord_names = {"lat", "latitude", "lon", "longitude", "time", "level"}
            if input_variables is None:
                input_variables = [
                    v for v in ds.data_vars if v.lower() not in coord_names
                ]
            self.input_variables = input_variables

            # 读取数据 - 处理可能存在的time维度
            data_list = []
            for v in input_variables:
                arr = ds[v].values
                # 如果有time维度且为1，squeeze掉
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr[0]
                data_list.append(arr)
            self.data = np.stack(data_list, axis=0).astype(np.float32)  # (C, H, W)

            # 读取坐标
            self.lat = self._get_coord(ds, ["lat", "latitude"])
            self.lon = self._get_coord(ds, ["lon", "longitude"])

            # 读取时间
            self.time = self._get_time(ds)

        self.img_shape = self.data.shape[-2:]

        # 加载归一化统计
        if stats is not None:
            self._setup_normalization(stats)
        else:
            n_channels = len(self.input_variables)
            self.input_mean = np.zeros((n_channels, 1, 1), dtype=np.float32)
            self.input_std = np.ones((n_channels, 1, 1), dtype=np.float32)
            self.output_mean = np.zeros((n_channels, 1, 1), dtype=np.float32)
            self.output_std = np.ones((n_channels, 1, 1), dtype=np.float32)

    def _get_coord(self, ds, names: List[str]) -> Optional[np.ndarray]:
        """获取坐标数据"""
        for name in names:
            if name in ds.coords:
                return ds[name].values
            if name in ds.data_vars:
                return ds[name].values
        return None

    def _get_time(self, ds) -> Optional[datetime.datetime]:
        """获取时间信息"""
        if "time" in ds.coords:
            time_val = ds["time"].values
            if hasattr(time_val, "__len__") and len(time_val) > 0:
                time_val = time_val[0]
            # 转换为datetime
            if isinstance(time_val, np.datetime64):
                return time_val.astype("datetime64[s]").astype(datetime.datetime)
            return time_val
        return None

    def _setup_normalization(self, stats: Dict):
        """设置归一化参数"""
        input_mean, input_std = [], []
        for v in self.input_variables:
            if "input" in stats and v in stats["input"]:
                input_mean.append(stats["input"][v]["mean"])
                input_std.append(stats["input"][v]["std"])
            else:
                input_mean.append(0.0)
                input_std.append(1.0)

        self.input_mean = np.array(input_mean)[:, None, None].astype(np.float32)
        self.input_std = np.array(input_std)[:, None, None].astype(np.float32)

        # 输出统计
        output_mean, output_std = [], []
        for v in self.input_variables:
            if "output" in stats and v in stats["output"]:
                output_mean.append(stats["output"][v]["mean"])
                output_std.append(stats["output"][v]["std"])
            else:
                output_mean.append(0.0)
                output_std.append(1.0)

        self.output_mean = np.array(output_mean)[:, None, None].astype(np.float32)
        self.output_std = np.array(output_std)[:, None, None].astype(np.float32)

    def get_normalized_input(self) -> np.ndarray:
        """获取归一化后的输入数据"""
        return (self.data - self.input_mean) / self.input_std

    def get_raw_input(self) -> np.ndarray:
        """获取原始输入数据"""
        return self.data.copy()

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        """反归一化输出数据"""
        return x * self.output_std + self.output_mean

    def get_lat_lon_2d(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取2D经纬度网格"""
        if self.lat is None or self.lon is None:
            return (
                np.full(self.img_shape, np.nan),
                np.full(self.img_shape, np.nan),
            )

        if self.lat.ndim == 1 and self.lon.ndim == 1:
            lon_2d, lat_2d = np.meshgrid(self.lon, self.lat)
            return lat_2d, lon_2d

        return self.lat, self.lon


class PatchInferenceManager:
    """
    Patch推理管理器
    
    自动处理输入图像的patch切分、推理和拼接还原
    """

    def __init__(
        self,
        model_patch_shape: Tuple[int, int],
        overlap_pix: int = 32,
        boundary_pix: int = 0,
    ):
        """
        Parameters
        ----------
        model_patch_shape : Tuple[int, int]
            模型期望的patch尺寸 (H, W)
        overlap_pix : int
            相邻patch之间的重叠像素数（用于平滑拼接）
        boundary_pix : int
            边界像素数
        """
        self.model_patch_shape = model_patch_shape
        self.overlap_pix = overlap_pix
        self.boundary_pix = boundary_pix

    def needs_patching(self, img_shape: Tuple[int, int]) -> bool:
        """检查是否需要patch切分"""
        return (
            img_shape[0] > self.model_patch_shape[0]
            or img_shape[1] > self.model_patch_shape[1]
        )

    def get_patching(self, img_shape: Tuple[int, int]) -> Optional[GridPatching2D]:
        """获取patch切分器"""
        if not self.needs_patching(img_shape):
            return None

        return GridPatching2D(
            img_shape=img_shape,
            patch_shape=self.model_patch_shape,
            overlap_pix=self.overlap_pix,
            boundary_pix=self.boundary_pix,
        )

    def infer_with_patching(
        self,
        img_lr: torch.Tensor,
        inference_fn,
        n_output_channels: int,
        num_ensembles: int = 1,
    ) -> torch.Tensor:
        """
        对输入进行patch切分推理并拼接还原

        Parameters
        ----------
        img_lr : torch.Tensor
            输入低分辨率图像 (1, C, H, W)
        inference_fn : callable
            推理函数，接受 (img_lr_patch, patch_shape) 返回 (ensemble, C, H_p, W_p)
        n_output_channels : int
            输出通道数
        num_ensembles : int
            集合成员数

        Returns
        -------
        torch.Tensor
            拼接还原后的输出 (ensemble, C, H, W)
        """
        img_shape = img_lr.shape[-2:]
        patching = self.get_patching(img_shape)

        if patching is None:
            # 不需要patch，直接推理
            return inference_fn(img_lr, img_shape)

        device = img_lr.device
        batch_size = 1  # 单张图片

        # 切分成patches
        # img_lr: (1, C, H, W) -> patches: (P, C, H_p, W_p)
        patches_lr = patching.apply(img_lr)
        n_patches = patching.patch_num

        print(f"    图像尺寸 {img_shape} -> 切分为 {n_patches} 个patches")

        # 对每个patch进行推理
        all_patch_results = []
        for p_idx in range(n_patches):
            patch_lr = patches_lr[p_idx : p_idx + 1]  # (1, C, H_p, W_p)

            # 推理
            patch_out = inference_fn(patch_lr, self.model_patch_shape)  # (ensemble, C, H_p, W_p)

            all_patch_results.append(patch_out)

        # 对每个ensemble成员分别拼接
        ensemble_results = []
        for e_idx in range(num_ensembles):
            # 收集当前ensemble的所有patch
            patches_for_ensemble = torch.stack(
                [r[e_idx] for r in all_patch_results], dim=0
            )  # (P, C, H_p, W_p)

            # 拼接还原
            # fuse期望输入: (P * B, C, H_p, W_p)，这里B=1
            fused = patching.fuse(patches_for_ensemble, batch_size=1)  # (1, C, H, W)
            ensemble_results.append(fused[0])  # (C, H, W)

        return torch.stack(ensemble_results, dim=0)  # (ensemble, C, H, W)


def write_output_nc(
    output_path: str,
    prediction: np.ndarray,
    time_val: Optional[datetime.datetime],
    lat: np.ndarray,
    lon: np.ndarray,
    variables: List[str],
    input_data: Optional[np.ndarray] = None,
    num_ensembles: int = 1,
):
    """
    保存预测结果为ERA5格式nc文件
    """
    # 处理维度
    if prediction.ndim == 3:
        prediction = prediction[np.newaxis, ...]  # (1, C, H, W)

    n_ensemble, n_channels, n_lat, n_lon = prediction.shape

    # 处理坐标
    lat_1d = lat[:, 0] if lat.ndim == 2 else lat
    lon_1d = lon[0, :] if lon.ndim == 2 else lon

    with nc.Dataset(output_path, "w", format="NETCDF4") as f:
        # 创建维度
        f.createDimension("time", 1)
        f.createDimension("lat", n_lat)
        f.createDimension("lon", n_lon)
        if n_ensemble > 1:
            f.createDimension("ensemble", n_ensemble)

        # 时间变量
        time_var = f.createVariable("time", "f8", ("time",))
        time_var.units = "hours since 1900-01-01 00:00:00"
        time_var.calendar = "standard"
        time_var.standard_name = "time"
        if time_val is not None:
            if isinstance(time_val, np.datetime64):
                time_val = time_val.astype("datetime64[s]").astype(datetime.datetime)
            time_var[0] = cftime.date2num(
                time_val, units=time_var.units, calendar=time_var.calendar
            )
        else:
            time_var[0] = 0

        # 纬度
        lat_var = f.createVariable("lat", "f4", ("lat",))
        lat_var.units = "degrees_north"
        lat_var.standard_name = "latitude"
        lat_var[:] = lat_1d

        # 经度
        lon_var = f.createVariable("lon", "f4", ("lon",))
        lon_var.units = "degrees_east"
        lon_var.standard_name = "longitude"
        lon_var[:] = lon_1d

        # 集合成员
        if n_ensemble > 1:
            ens_var = f.createVariable("ensemble", "i4", ("ensemble",))
            ens_var.long_name = "ensemble member"
            ens_var[:] = np.arange(n_ensemble)

        # 预测变量
        for i, var_name in enumerate(variables):
            if n_ensemble > 1:
                dims = ("time", "ensemble", "lat", "lon")
                var = f.createVariable(var_name, "f4", dims, zlib=True, complevel=4)
                var[0, :, :, :] = prediction[:, i, :, :]
            else:
                dims = ("time", "lat", "lon")
                var = f.createVariable(var_name, "f4", dims, zlib=True, complevel=4)
                var[0, :, :] = prediction[0, i, :, :]
            var.long_name = var_name

        # 输入数据（可选）
        if input_data is not None:
            input_grp = f.createGroup("input")
            for i, var_name in enumerate(variables[:input_data.shape[0]]):
                var = input_grp.createVariable(
                    var_name, "f4", ("time", "lat", "lon"), zlib=True, complevel=4
                )
                var[0, :, :] = input_data[i, :, :]

        # 全局属性
        f.Conventions = "CF-1.6"
        f.history = f"Created by inference_era5.py on {datetime.datetime.now()}"
        f.source = "CorrDiff downscaling model"


def get_model_patch_shape(model: torch.nn.Module) -> Optional[Tuple[int, int]]:
    """从模型中获取期望的patch尺寸"""
    # 尝试从模型属性获取
    if hasattr(model, "img_resolution"):
        res = model.img_resolution
        if isinstance(res, (list, tuple)) and len(res) == 2:
            return tuple(res)
    
    # 尝试从模型配置获取
    if hasattr(model, "_meta"):
        meta = model._meta
        if hasattr(meta, "img_resolution"):
            return tuple(meta.img_resolution)
    
    return None


def run_inference(
    input_dir: str,
    output_dir: str,
    reg_ckpt: Optional[str] = None,
    res_ckpt: Optional[str] = None,
    stats_path: Optional[str] = None,
    num_ensembles: int = 1,
    seed_batch_size: int = 1,
    num_steps: int = 18,
    device: str = "cuda",
    use_fp16: bool = False,
    hr_mean_conditioning: bool = True,
    save_input: bool = False,
    file_pattern: str = "*.nc",
    patch_shape: Optional[Tuple[int, int]] = None,
    overlap_pix: int = 32,
    boundary_pix: int = 0,
):
    """
    对文件夹中的ERA5数据执行降尺度推理

    Parameters
    ----------
    input_dir : str
        输入文件夹路径
    output_dir : str
        输出文件夹路径
    reg_ckpt : str, optional
        回归模型检查点路径
    res_ckpt : str, optional
        扩散模型检查点路径
    stats_path : str, optional
        统计信息JSON文件路径
    num_ensembles : int
        集合成员数量
    seed_batch_size : int
        批次大小
    num_steps : int
        扩散采样步数
    device : str
        计算设备
    use_fp16 : bool
        是否使用半精度
    hr_mean_conditioning : bool
        是否使用高分辨率均值条件
    save_input : bool
        是否在输出中保存输入数据
    file_pattern : str
        输入文件匹配模式
    patch_shape : Tuple[int, int], optional
        模型patch尺寸 (H, W)，None则从模型自动获取
    overlap_pix : int
        patch重叠像素数
    boundary_pix : int
        边界像素数
    """
    logger = PythonLogger("inference")
    logger.info("=" * 60)
    logger.info("ERA5降尺度推理（支持自动patch切分）")
    logger.info("=" * 60)
    logger.info(f"输入文件夹: {input_dir}")
    logger.info(f"输出文件夹: {output_dir}")

    # 检查设备
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA不可用，使用CPU")
        device = "cpu"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取输入文件列表
    input_files = sorted(glob(os.path.join(input_dir, file_pattern)))
    if not input_files:
        raise FileNotFoundError(f"在 {input_dir} 中未找到匹配 {file_pattern} 的文件")

    logger.info(f"找到 {len(input_files)} 个输入文件")

    # 加载统计信息
    stats = None
    if stats_path and os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
        logger.info(f"已加载统计信息: {stats_path}")

    # 加载模型
    net_reg = None
    net_res = None
    model_patch_shape = patch_shape

    if reg_ckpt:
        logger.info(f"加载回归模型: {reg_ckpt}")
        net_reg = Module.from_checkpoint(reg_ckpt)
        net_reg.use_fp16 = use_fp16
        net_reg = net_reg.eval().to(device).to(memory_format=torch.channels_last)
        if hasattr(net_reg, "amp_mode"):
            net_reg.amp_mode = False
        # 尝试获取模型patch尺寸
        if model_patch_shape is None:
            model_patch_shape = get_model_patch_shape(net_reg)

    if res_ckpt:
        logger.info(f"加载扩散模型: {res_ckpt}")
        net_res = Module.from_checkpoint(res_ckpt)
        net_res.use_fp16 = use_fp16
        net_res = net_res.eval().to(device).to(memory_format=torch.channels_last)
        if hasattr(net_res, "amp_mode"):
            net_res.amp_mode = False
        # 尝试获取模型patch尺寸
        if model_patch_shape is None:
            model_patch_shape = get_model_patch_shape(net_res)

    if net_reg is None and net_res is None:
        raise ValueError("至少需要提供回归模型或扩散模型检查点")

    # 确保有patch尺寸
    if model_patch_shape is None:
        raise ValueError(
            "无法从模型获取patch尺寸，请通过 --patch_shape 参数手动指定"
        )

    logger.info(f"模型patch尺寸: {model_patch_shape}")
    logger.info(f"patch重叠像素: {overlap_pix}")

    # 创建patch推理管理器
    patch_manager = PatchInferenceManager(
        model_patch_shape=model_patch_shape,
        overlap_pix=overlap_pix,
        boundary_pix=boundary_pix,
    )

    # 设置采样器
    sampler_fn = partial(
        deterministic_sampler,
        num_steps=num_steps,
        solver="euler",
        patching=None,
    )

    # 设置种子批次
    seeds = list(np.arange(num_ensembles))
    num_batches = max(1, (len(seeds) - 1) // seed_batch_size + 1)
    rank_batches = np.array_split(seeds, num_batches)

    logger.info(f"集合成员数: {num_ensembles}")
    logger.info(f"扩散采样步数: {num_steps}")
    logger.info("开始推理...")

    # 定义单个patch的推理函数
    def inference_fn(img_lr_patch: torch.Tensor, patch_shape: Tuple[int, int]) -> torch.Tensor:
        """对单个patch进行推理"""
        n_channels = img_lr_patch.shape[1]

        # 回归步骤
        if net_reg:
            latents_shape = (num_ensembles, n_channels, patch_shape[0], patch_shape[1])
            with torch.no_grad():
                image_reg = regression_step(
                    net=net_reg,
                    img_lr=img_lr_patch,
                    latents_shape=latents_shape,
                )
        else:
            image_reg = None

        # 扩散步骤
        if net_res:
            if hr_mean_conditioning and image_reg is not None:
                mean_hr = image_reg[0:1]
            else:
                mean_hr = None

            with torch.no_grad():
                image_res = diffusion_step(
                    net=net_res,
                    sampler_fn=sampler_fn,
                    img_shape=patch_shape,
                    img_out_channels=n_channels,
                    rank_batches=rank_batches,
                    img_lr=img_lr_patch.expand(seed_batch_size, -1, -1, -1),
                    rank=0,
                    device=device,
                    mean_hr=mean_hr,
                )
        else:
            image_res = None

        # 合并结果
        if image_reg is not None and image_res is not None:
            return image_reg + image_res
        elif image_reg is not None:
            return image_reg
        else:
            return image_res

    # 遍历处理每个文件
    for input_file in tqdm(input_files, desc="处理文件"):
        try:
            # 读取输入数据
            reader = ERA5SingleTimeReader(
                nc_path=input_file,
                stats=stats,
            )

            img_shape = reader.img_shape
            n_channels = len(reader.input_variables)

            # 获取归一化输入
            img_lr = reader.get_normalized_input()  # (C, H, W)
            img_lr = torch.from_numpy(img_lr).float().unsqueeze(0)  # (1, C, H, W)
            img_lr = img_lr.to(device).to(memory_format=torch.channels_last)

            # 使用patch管理器进行推理
            if patch_manager.needs_patching(img_shape):
                logger.info(f"  {Path(input_file).name}: 需要patch切分")
                image_out = patch_manager.infer_with_patching(
                    img_lr=img_lr,
                    inference_fn=inference_fn,
                    n_output_channels=n_channels,
                    num_ensembles=num_ensembles,
                )
            else:
                # 直接推理
                image_out = inference_fn(img_lr, img_shape)

            # 反归一化
            image_out_np = image_out.cpu().numpy()  # (ensemble, C, H, W)
            image_out_np = reader.denormalize_output(image_out_np)

            # 获取坐标
            lat_2d, lon_2d = reader.get_lat_lon_2d()

            # 构建输出文件名
            input_name = Path(input_file).stem
            output_file = os.path.join(output_dir, f"{input_name}_downscaled.nc")

            # 保存结果
            write_output_nc(
                output_path=output_file,
                prediction=image_out_np,
                time_val=reader.time,
                lat=lat_2d,
                lon=lon_2d,
                variables=reader.input_variables,
                input_data=reader.get_raw_input() if save_input else None,
                num_ensembles=num_ensembles,
            )

        except Exception as e:
            logger.error(f"处理文件 {input_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info("=" * 60)
    logger.info("推理完成!")
    logger.info(f"输出保存至: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="ERA5降尺度推理 - 支持自动patch切分",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法（自动从模型获取patch尺寸）
  python inference_era5.py \\
      --input_dir ./data/era5_input/ \\
      --output_dir ./outputs/ \\
      --reg_ckpt ./checkpoints/regression.mdlus \\
      --stats_path ./data/stats.json

  # 手动指定patch尺寸
  python inference_era5.py \\
      --input_dir ./data/era5_input/ \\
      --output_dir ./outputs/ \\
      --reg_ckpt ./checkpoints/regression.mdlus \\
      --stats_path ./data/stats.json \\
      --patch_shape 448 448 \\
      --overlap_pix 32

  # 使用扩散模型生成多集合成员
  python inference_era5.py \\
      --input_dir ./data/era5_input/ \\
      --output_dir ./outputs/ \\
      --reg_ckpt ./checkpoints/regression.mdlus \\
      --res_ckpt ./checkpoints/diffusion.mdlus \\
      --stats_path ./data/stats.json \\
      --num_ensembles 10 \\
      --patch_shape 448 448
        """,
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="输入文件夹路径（包含单时间戳ERA5 nc文件）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出文件夹路径",
    )
    parser.add_argument(
        "--reg_ckpt",
        type=str,
        default=None,
        help="回归模型检查点路径",
    )
    parser.add_argument(
        "--res_ckpt",
        type=str,
        default=None,
        help="扩散模型检查点路径",
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default=None,
        help="统计信息JSON文件路径",
    )
    parser.add_argument(
        "--num_ensembles",
        type=int,
        default=1,
        help="集合成员数量（默认: 1）",
    )
    parser.add_argument(
        "--seed_batch_size",
        type=int,
        default=1,
        help="批次大小（默认: 1）",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=18,
        help="扩散采样步数（默认: 18）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="计算设备（默认: cuda）",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="使用半精度",
    )
    parser.add_argument(
        "--no_hr_mean_conditioning",
        action="store_true",
        help="禁用高分辨率均值条件",
    )
    parser.add_argument(
        "--save_input",
        action="store_true",
        help="在输出nc中保存输入数据",
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*.nc",
        help="输入文件匹配模式（默认: *.nc）",
    )
    parser.add_argument(
        "--patch_shape",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="模型patch尺寸 (高度 宽度)，不指定则从模型自动获取",
    )
    parser.add_argument(
        "--overlap_pix",
        type=int,
        default=32,
        help="patch重叠像素数（默认: 32）",
    )
    parser.add_argument(
        "--boundary_pix",
        type=int,
        default=0,
        help="边界像素数（默认: 0）",
    )

    args = parser.parse_args()

    run_inference(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        reg_ckpt=args.reg_ckpt,
        res_ckpt=args.res_ckpt,
        stats_path=args.stats_path,
        num_ensembles=args.num_ensembles,
        seed_batch_size=args.seed_batch_size,
        num_steps=args.num_steps,
        device=args.device,
        use_fp16=args.fp16,
        hr_mean_conditioning=not args.no_hr_mean_conditioning,
        save_input=args.save_input,
        file_pattern=args.file_pattern,
        patch_shape=tuple(args.patch_shape) if args.patch_shape else None,
        overlap_pix=args.overlap_pix,
        boundary_pix=args.boundary_pix,
    )


if __name__ == "__main__":
    main()
