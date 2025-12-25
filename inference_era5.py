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

输入: 低分辨率ERA5数据（如25km），指定经纬度范围
输出: 高分辨率ERA5数据（如1km），相同经纬度范围

处理流程:
1. 读取低分辨率输入nc文件
2. 将低分辨率数据切分成多个patch
3. 每个patch通过双线性插值放大到模型输入尺寸（如256x256）
4. 对插值后的数据进行模型推理
5. 将推理结果拼接还原为完整的高分辨率输出
6. 保存为ERA5格式nc文件

用法:
    python inference_era5.py \
        --input_dir <输入文件夹> \
        --output_dir <输出文件夹> \
        --reg_ckpt <回归模型检查点> \
        --stats_path <统计信息json文件> \
        --lr_patch_size 10 10 \
        --model_input_size 256 256 \
        --scale_factor 25
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
import torch.nn.functional as F
import xarray as xr
import netCDF4 as nc
import cftime
from tqdm import tqdm

from physicsnemo import Module
from physicsnemo.utils.logging import PythonLogger
from physicsnemo.models.diffusion.sampling import deterministic_sampler
from physicsnemo.models.diffusion.corrdiff_utils import regression_step, diffusion_step


class LowResolutionPatchManager:
    """
    低分辨率数据Patch管理器
    
    处理低分辨率输入的切分、插值、推理和拼接还原
    """

    def __init__(
        self,
        lr_patch_size: Tuple[int, int],
        model_input_size: Tuple[int, int],
        scale_factor: int,
        overlap_lr_pixels: int = 1,
    ):
        """
        Parameters
        ----------
        lr_patch_size : Tuple[int, int]
            低分辨率patch尺寸 (H, W)，如 (10, 10) 表示10x10个低分辨率像素
        model_input_size : Tuple[int, int]
            模型输入尺寸 (H, W)，如 (256, 256)
        scale_factor : int
            分辨率放大倍数，如从25km到1km则为25
        overlap_lr_pixels : int
            低分辨率patch之间的重叠像素数（用于平滑拼接）
        """
        self.lr_patch_size = lr_patch_size
        self.model_input_size = model_input_size
        self.scale_factor = scale_factor
        self.overlap_lr_pixels = overlap_lr_pixels
        
        # 计算高分辨率patch尺寸
        self.hr_patch_size = (
            lr_patch_size[0] * scale_factor,
            lr_patch_size[1] * scale_factor,
        )
        
        # 高分辨率重叠像素数
        self.overlap_hr_pixels = overlap_lr_pixels * scale_factor

    def compute_patches(
        self, lr_shape: Tuple[int, int]
    ) -> List[Dict]:
        """
        计算低分辨率图像的patch划分
        
        Parameters
        ----------
        lr_shape : Tuple[int, int]
            低分辨率图像尺寸 (H, W)
            
        Returns
        -------
        List[Dict]
            每个patch的信息：
            - lr_slice: 低分辨率切片 (y_start, y_end, x_start, x_end)
            - hr_slice: 高分辨率切片
            - lr_size: 实际低分辨率尺寸（边界可能小于标准尺寸）
            - hr_size: 实际高分辨率尺寸
            - needs_padding: 是否需要padding
        """
        lr_h, lr_w = lr_shape
        patches = []
        
        # 计算步长（考虑重叠）
        step_h = self.lr_patch_size[0] - self.overlap_lr_pixels
        step_w = self.lr_patch_size[1] - self.overlap_lr_pixels
        
        # 确保步长至少为1
        step_h = max(1, step_h)
        step_w = max(1, step_w)
        
        # 计算patch数量
        n_patches_h = max(1, math.ceil((lr_h - self.overlap_lr_pixels) / step_h))
        n_patches_w = max(1, math.ceil((lr_w - self.overlap_lr_pixels) / step_w))
        
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                # 低分辨率切片范围
                lr_y_start = i * step_h
                lr_x_start = j * step_w
                lr_y_end = min(lr_y_start + self.lr_patch_size[0], lr_h)
                lr_x_end = min(lr_x_start + self.lr_patch_size[1], lr_w)
                
                # 实际patch尺寸
                actual_lr_h = lr_y_end - lr_y_start
                actual_lr_w = lr_x_end - lr_x_start
                
                # 高分辨率对应范围
                hr_y_start = lr_y_start * self.scale_factor
                hr_x_start = lr_x_start * self.scale_factor
                hr_y_end = lr_y_end * self.scale_factor
                hr_x_end = lr_x_end * self.scale_factor
                
                actual_hr_h = hr_y_end - hr_y_start
                actual_hr_w = hr_x_end - hr_x_start
                
                # 检查是否需要padding（边界patch可能小于标准尺寸）
                needs_padding = (
                    actual_lr_h < self.lr_patch_size[0] or
                    actual_lr_w < self.lr_patch_size[1]
                )
                
                patches.append({
                    "lr_slice": (lr_y_start, lr_y_end, lr_x_start, lr_x_end),
                    "hr_slice": (hr_y_start, hr_y_end, hr_x_start, hr_x_end),
                    "lr_size": (actual_lr_h, actual_lr_w),
                    "hr_size": (actual_hr_h, actual_hr_w),
                    "needs_padding": needs_padding,
                    "patch_idx": (i, j),
                })
        
        return patches

    def extract_and_interpolate_patch(
        self,
        lr_data: torch.Tensor,
        patch_info: Dict,
    ) -> torch.Tensor:
        """
        提取低分辨率patch并插值到模型输入尺寸
        
        Parameters
        ----------
        lr_data : torch.Tensor
            低分辨率数据 (1, C, H, W)
        patch_info : Dict
            patch信息
            
        Returns
        -------
        torch.Tensor
            插值后的数据 (1, C, model_H, model_W)
        """
        lr_y_start, lr_y_end, lr_x_start, lr_x_end = patch_info["lr_slice"]
        
        # 提取patch
        patch = lr_data[:, :, lr_y_start:lr_y_end, lr_x_start:lr_x_end]
        
        # 如果边界patch较小，先padding到标准低分辨率尺寸
        if patch_info["needs_padding"]:
            actual_h, actual_w = patch_info["lr_size"]
            target_h, target_w = self.lr_patch_size
            
            # 使用反射padding
            pad_h = target_h - actual_h
            pad_w = target_w - actual_w
            
            # 先尝试反射padding，如果太小则用复制padding
            if actual_h > 1 and actual_w > 1:
                patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='reflect')
            else:
                patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='replicate')
        
        # 双线性插值到模型输入尺寸
        patch_interpolated = F.interpolate(
            patch,
            size=self.model_input_size,
            mode='bilinear',
            align_corners=False,
        )
        
        return patch_interpolated

    def assemble_output(
        self,
        patches_output: List[torch.Tensor],
        patches_info: List[Dict],
        hr_shape: Tuple[int, int],
        n_channels: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        将推理结果拼接还原为完整的高分辨率输出
        
        Parameters
        ----------
        patches_output : List[torch.Tensor]
            每个patch的推理结果列表，每个 (ensemble, C, model_H, model_W)
        patches_info : List[Dict]
            patch信息列表
        hr_shape : Tuple[int, int]
            目标高分辨率尺寸 (H, W)
        n_channels : int
            通道数
        device : torch.device
            设备
            
        Returns
        -------
        torch.Tensor
            拼接后的完整输出 (ensemble, C, H, W)
        """
        n_ensembles = patches_output[0].shape[0]
        hr_h, hr_w = hr_shape
        
        # 初始化输出和权重
        output = torch.zeros(
            (n_ensembles, n_channels, hr_h, hr_w),
            dtype=torch.float32,
            device=device,
        )
        weight = torch.zeros(
            (1, 1, hr_h, hr_w),
            dtype=torch.float32,
            device=device,
        )
        
        for patch_out, patch_info in zip(patches_output, patches_info):
            hr_y_start, hr_y_end, hr_x_start, hr_x_end = patch_info["hr_slice"]
            actual_hr_h, actual_hr_w = patch_info["hr_size"]
            
            # 如果是边界patch，需要从插值结果中裁剪出实际区域
            if patch_info["needs_padding"]:
                # 计算需要保留的区域
                # 模型输出尺寸 -> 实际高分辨率尺寸
                # 首先将模型输出resize到标准高分辨率patch尺寸
                patch_resized = F.interpolate(
                    patch_out,
                    size=self.hr_patch_size,
                    mode='bilinear',
                    align_corners=False,
                )
                # 然后裁剪出实际区域
                patch_cropped = patch_resized[:, :, :actual_hr_h, :actual_hr_w]
            else:
                # 非边界patch，直接resize到标准高分辨率尺寸
                patch_cropped = F.interpolate(
                    patch_out,
                    size=self.hr_patch_size,
                    mode='bilinear',
                    align_corners=False,
                )
            
            # 创建融合权重（中心权重高，边缘权重低）
            w = self._create_blend_weight(
                (actual_hr_h, actual_hr_w),
                self.overlap_hr_pixels,
                device,
            )
            
            # 加权累加
            output[:, :, hr_y_start:hr_y_end, hr_x_start:hr_x_end] += patch_cropped * w
            weight[:, :, hr_y_start:hr_y_end, hr_x_start:hr_x_end] += w
        
        # 归一化
        output = output / weight.clamp(min=1e-8)
        
        return output

    def _create_blend_weight(
        self,
        size: Tuple[int, int],
        overlap: int,
        device: torch.device,
    ) -> torch.Tensor:
        """创建融合权重（用于平滑拼接）"""
        h, w = size
        
        if overlap <= 0:
            return torch.ones((1, 1, h, w), device=device)
        
        # 创建1D权重
        def create_1d_weight(length: int, fade: int) -> torch.Tensor:
            weight = torch.ones(length, device=device)
            if fade > 0 and length > 2 * fade:
                # 边缘渐变
                fade_in = torch.linspace(0, 1, fade, device=device)
                fade_out = torch.linspace(1, 0, fade, device=device)
                weight[:fade] = fade_in
                weight[-fade:] = fade_out
            return weight
        
        weight_h = create_1d_weight(h, min(overlap, h // 2))
        weight_w = create_1d_weight(w, min(overlap, w // 2))
        
        # 外积创建2D权重
        weight_2d = weight_h.unsqueeze(1) * weight_w.unsqueeze(0)
        
        return weight_2d.unsqueeze(0).unsqueeze(0)


class ERA5SingleTimeReader:
    """读取单时间戳ERA5 nc文件"""

    def __init__(
        self,
        nc_path: str,
        input_variables: Optional[List[str]] = None,
        stats: Optional[Dict] = None,
    ):
        self.nc_path = nc_path

        with xr.open_dataset(nc_path) as ds:
            # 获取变量名（排除坐标变量）
            coord_names = {"lat", "latitude", "lon", "longitude", "time", "level"}
            if input_variables is None:
                input_variables = [
                    v for v in ds.data_vars if v.lower() not in coord_names
                ]
            self.input_variables = input_variables

            # 读取数据
            data_list = []
            for v in input_variables:
                arr = ds[v].values
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr[0]
                data_list.append(arr)
            self.data = np.stack(data_list, axis=0).astype(np.float32)  # (C, H, W)

            # 读取坐标
            self.lat = self._get_coord(ds, ["lat", "latitude"])
            self.lon = self._get_coord(ds, ["lon", "longitude"])
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
        for name in names:
            if name in ds.coords:
                return ds[name].values
            if name in ds.data_vars:
                return ds[name].values
        return None

    def _get_time(self, ds) -> Optional[datetime.datetime]:
        if "time" in ds.coords:
            time_val = ds["time"].values
            if hasattr(time_val, "__len__") and len(time_val) > 0:
                time_val = time_val[0]
            if isinstance(time_val, np.datetime64):
                return time_val.astype("datetime64[s]").astype(datetime.datetime)
            return time_val
        return None

    def _setup_normalization(self, stats: Dict):
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
        return (self.data - self.input_mean) / self.input_std

    def get_raw_input(self) -> np.ndarray:
        return self.data.copy()

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        return x * self.output_std + self.output_mean

    def get_hr_coordinates(self, scale_factor: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成高分辨率坐标网格
        
        Parameters
        ----------
        scale_factor : int
            放大倍数
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            高分辨率 (lat, lon) 数组
        """
        if self.lat is None or self.lon is None:
            hr_h = self.img_shape[0] * scale_factor
            hr_w = self.img_shape[1] * scale_factor
            return np.full((hr_h,), np.nan), np.full((hr_w,), np.nan)
        
        # 对1D坐标进行插值
        if self.lat.ndim == 1:
            lr_lat = self.lat
            lr_lon = self.lon
            
            # 创建高分辨率坐标
            hr_lat = np.linspace(lr_lat[0], lr_lat[-1], len(lr_lat) * scale_factor)
            hr_lon = np.linspace(lr_lon[0], lr_lon[-1], len(lr_lon) * scale_factor)
            
            return hr_lat, hr_lon
        else:
            # 2D坐标使用插值
            from scipy import ndimage
            hr_lat = ndimage.zoom(self.lat, scale_factor, order=1)
            hr_lon = ndimage.zoom(self.lon, scale_factor, order=1)
            return hr_lat, hr_lon


def write_output_nc(
    output_path: str,
    prediction: np.ndarray,
    time_val: Optional[datetime.datetime],
    lat: np.ndarray,
    lon: np.ndarray,
    variables: List[str],
    input_data: Optional[np.ndarray] = None,
    input_lat: Optional[np.ndarray] = None,
    input_lon: Optional[np.ndarray] = None,
):
    """
    保存预测结果为ERA5格式nc文件
    """
    if prediction.ndim == 3:
        prediction = prediction[np.newaxis, ...]

    n_ensemble, n_channels, n_lat, n_lon = prediction.shape

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
        lat_var[:] = lat

        # 经度
        lon_var = f.createVariable("lon", "f4", ("lon",))
        lon_var.units = "degrees_east"
        lon_var.standard_name = "longitude"
        lon_var[:] = lon

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
            
            # 输入维度
            lr_h, lr_w = input_data.shape[-2:]
            input_grp.createDimension("lat_lr", lr_h)
            input_grp.createDimension("lon_lr", lr_w)
            
            if input_lat is not None:
                lat_lr = input_grp.createVariable("lat", "f4", ("lat_lr",))
                lat_lr[:] = input_lat
            if input_lon is not None:
                lon_lr = input_grp.createVariable("lon", "f4", ("lon_lr",))
                lon_lr[:] = input_lon
            
            for i, var_name in enumerate(variables[:input_data.shape[0]]):
                var = input_grp.createVariable(
                    var_name, "f4", ("lat_lr", "lon_lr"), zlib=True, complevel=4
                )
                var[:] = input_data[i, :, :]

        # 全局属性
        f.Conventions = "CF-1.6"
        f.history = f"Created by inference_era5.py on {datetime.datetime.now()}"
        f.source = "CorrDiff downscaling model"
        f.resolution_scale_factor = f"Input upscaled to high resolution"


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
    lr_patch_size: Tuple[int, int] = (10, 10),
    model_input_size: Tuple[int, int] = (256, 256),
    scale_factor: int = 25,
    overlap_lr_pixels: int = 1,
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
    lr_patch_size : Tuple[int, int]
        低分辨率patch尺寸 (H, W)
    model_input_size : Tuple[int, int]
        模型输入尺寸 (H, W)
    scale_factor : int
        分辨率放大倍数（如25km->1km则为25）
    overlap_lr_pixels : int
        低分辨率patch重叠像素数
    """
    logger = PythonLogger("inference")
    logger.info("=" * 60)
    logger.info("ERA5降尺度推理")
    logger.info("=" * 60)
    logger.info(f"输入文件夹: {input_dir}")
    logger.info(f"输出文件夹: {output_dir}")
    logger.info(f"低分辨率patch尺寸: {lr_patch_size}")
    logger.info(f"模型输入尺寸: {model_input_size}")
    logger.info(f"分辨率放大倍数: {scale_factor}x")

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

    if reg_ckpt:
        logger.info(f"加载回归模型: {reg_ckpt}")
        net_reg = Module.from_checkpoint(reg_ckpt)
        net_reg.use_fp16 = use_fp16
        net_reg = net_reg.eval().to(device).to(memory_format=torch.channels_last)
        if hasattr(net_reg, "amp_mode"):
            net_reg.amp_mode = False

    if res_ckpt:
        logger.info(f"加载扩散模型: {res_ckpt}")
        net_res = Module.from_checkpoint(res_ckpt)
        net_res.use_fp16 = use_fp16
        net_res = net_res.eval().to(device).to(memory_format=torch.channels_last)
        if hasattr(net_res, "amp_mode"):
            net_res.amp_mode = False

    if net_reg is None and net_res is None:
        raise ValueError("至少需要提供回归模型或扩散模型检查点")

    # 创建patch管理器
    patch_manager = LowResolutionPatchManager(
        lr_patch_size=lr_patch_size,
        model_input_size=model_input_size,
        scale_factor=scale_factor,
        overlap_lr_pixels=overlap_lr_pixels,
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
    def inference_fn(patch_input: torch.Tensor) -> torch.Tensor:
        """对单个插值后的patch进行推理"""
        n_channels = patch_input.shape[1]
        patch_shape = patch_input.shape[-2:]

        # 回归步骤
        if net_reg:
            latents_shape = (num_ensembles, n_channels, patch_shape[0], patch_shape[1])
            with torch.no_grad():
                image_reg = regression_step(
                    net=net_reg,
                    img_lr=patch_input,
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
                    img_lr=patch_input.expand(seed_batch_size, -1, -1, -1),
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

            lr_shape = reader.img_shape
            n_channels = len(reader.input_variables)

            # 计算高分辨率输出尺寸
            hr_shape = (lr_shape[0] * scale_factor, lr_shape[1] * scale_factor)

            # 获取归一化输入
            img_lr = reader.get_normalized_input()  # (C, H, W)
            img_lr = torch.from_numpy(img_lr).float().unsqueeze(0)  # (1, C, H, W)
            img_lr = img_lr.to(device)

            # 计算patch划分
            patches_info = patch_manager.compute_patches(lr_shape)
            n_patches = len(patches_info)

            logger.info(
                f"  {Path(input_file).name}: "
                f"LR {lr_shape} -> HR {hr_shape}, "
                f"切分为 {n_patches} 个patches"
            )

            # 对每个patch进行推理
            patches_output = []
            for patch_info in patches_info:
                # 提取并插值patch
                patch_input = patch_manager.extract_and_interpolate_patch(
                    img_lr, patch_info
                )
                patch_input = patch_input.to(memory_format=torch.channels_last)

                # 推理
                patch_out = inference_fn(patch_input)  # (ensemble, C, H, W)
                patches_output.append(patch_out)

            # 拼接还原
            output = patch_manager.assemble_output(
                patches_output=patches_output,
                patches_info=patches_info,
                hr_shape=hr_shape,
                n_channels=n_channels,
                device=device,
            )

            # 反归一化
            output_np = output.cpu().numpy()  # (ensemble, C, H, W)
            output_np = reader.denormalize_output(output_np)

            # 生成高分辨率坐标
            hr_lat, hr_lon = reader.get_hr_coordinates(scale_factor)

            # 构建输出文件名
            input_name = Path(input_file).stem
            output_file = os.path.join(output_dir, f"{input_name}_downscaled.nc")

            # 保存结果
            write_output_nc(
                output_path=output_file,
                prediction=output_np,
                time_val=reader.time,
                lat=hr_lat,
                lon=hr_lon,
                variables=reader.input_variables,
                input_data=reader.get_raw_input() if save_input else None,
                input_lat=reader.lat if save_input else None,
                input_lon=reader.lon if save_input else None,
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
        description="ERA5降尺度推理 - 低分辨率切分插值推理拼接",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 福建区域25km ERA5降尺度到1km
  # 输入: 低分辨率ERA5 (如 20x25 像素)
  # 输出: 高分辨率 (500x625 像素)
  
  python inference_era5.py \\
      --input_dir ./data/era5_fujian_25km/ \\
      --output_dir ./outputs/era5_fujian_1km/ \\
      --reg_ckpt ./checkpoints/regression.mdlus \\
      --stats_path ./data/stats.json \\
      --lr_patch_size 10 10 \\
      --model_input_size 256 256 \\
      --scale_factor 25

  # 使用扩散模型生成多集合成员
  python inference_era5.py \\
      --input_dir ./data/era5_fujian_25km/ \\
      --output_dir ./outputs/era5_fujian_1km/ \\
      --reg_ckpt ./checkpoints/regression.mdlus \\
      --res_ckpt ./checkpoints/diffusion.mdlus \\
      --stats_path ./data/stats.json \\
      --lr_patch_size 10 10 \\
      --model_input_size 256 256 \\
      --scale_factor 25 \\
      --num_ensembles 10

处理流程:
  1. 读取低分辨率输入 (如25km, 20x25像素)
  2. 切分为多个低分辨率patch (如10x10像素)
  3. 每个patch双线性插值到模型输入尺寸 (256x256)
  4. 模型推理得到高分辨率输出
  5. 输出缩放到对应高分辨率尺寸 (10*25=250像素)
  6. 所有patch加权融合拼接
  7. 输出完整高分辨率结果 (500x625像素)
        """,
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="输入文件夹路径",
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
        "--lr_patch_size",
        type=int,
        nargs=2,
        default=[10, 10],
        metavar=("H", "W"),
        help="低分辨率patch尺寸（默认: 10 10）",
    )
    parser.add_argument(
        "--model_input_size",
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=("H", "W"),
        help="模型输入尺寸（默认: 256 256）",
    )
    parser.add_argument(
        "--scale_factor",
        type=int,
        default=25,
        help="分辨率放大倍数，如25km->1km则为25（默认: 25）",
    )
    parser.add_argument(
        "--overlap_lr_pixels",
        type=int,
        default=1,
        help="低分辨率patch重叠像素数（默认: 1）",
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
        lr_patch_size=tuple(args.lr_patch_size),
        model_input_size=tuple(args.model_input_size),
        scale_factor=args.scale_factor,
        overlap_lr_pixels=args.overlap_lr_pixels,
    )


if __name__ == "__main__":
    main()
