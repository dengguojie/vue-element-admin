/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021.All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef AICPU_KERNELS_NORMALIZED_GRIDSAMPLER2D_H_
#define AICPU_KERNELS_NORMALIZED_GRIDSAMPLER2D_H_

#include "Eigen/Core"
#include "cpu_kernel.h"

namespace aicpu {
class GridSampler2DCpuKernel : public CpuKernel {
 public:
  GridSampler2DCpuKernel() = default;
  ~GridSampler2DCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  static uint32_t GridSampler2DCompute(CpuKernelContext &ctx);

  static uint32_t GridSampler2DCompute_half(CpuKernelContext &ctx);

  template <typename T>
  static T grid_sampler_compute_source_index(T coord, int64_t size,
                                             std::string padding_mode,
                                             bool align_corners);

  template <typename T>
  static T reflect_coordinates(T coord, int64_t twice_low, int64_t twice_high);

  static bool within_bounds_2d(int64_t h, int64_t w, int64_t H, int64_t W);

  template <typename T>
  static void bilinear(T x, T y, T *x_data_addr, T *y_data_addr, int64_t y_c,
                       std::vector<int64_t> x_dims, int64_t *y_stride,
                       int64_t *x_stride, int64_t x_ptr_NC, int64_t y_ptr_NCHW);

  template <typename T>
  static void nearest(T x, T y, T *x_data_addr, T *y_data_addr, int64_t y_c,
                      std::vector<int64_t> x_dims, int64_t *y_stride,
                      int64_t *x_stride, int64_t x_ptr_NC, int64_t y_ptr_NCHW);

  static void bilinear_half(float x, float y, Eigen::half *x_data_addr,
                            Eigen::half *y_data_addr, int64_t y_c,
                            std::vector<int64_t> x_dims, int64_t *y_stride,
                            int64_t *x_stride, int64_t x_ptr_NC,
                            int64_t y_ptr_NCHW);
  static void nearest_half(float x, float y, Eigen::half *x_data_addr,
                           Eigen::half *y_data_addr, int64_t y_c,
                           std::vector<int64_t> x_dims, int64_t *y_stride,
                           int64_t *x_stride, int64_t x_ptr_NC,
                           int64_t y_ptr_NCHW);

  template <typename T>
  static void Call1(T *x_data_addr, T *y_data_addr, T *grid_data_addr,
                    std::vector<int64_t> x_dims, std::vector<int64_t> y_dims,
                    int64_t *y_stride, int64_t *x_stride, int64_t *grid_stride,
                    std::string interpolation_mode, std::string padding_mode,
                    bool align_corners);
  template <typename T>
  static uint32_t Call2(CpuKernelContext &ctx, T *x_data_addr, T *y_data_addr,
                        T *grid_data_addr, std::vector<int64_t> x_dims,
                        std::vector<int64_t> y_dims, int64_t *y_stride,
                        int64_t *x_stride, int64_t *grid_stride,
                        std::string interpolation_mode,
                        std::string padding_mode, bool align_corners);
  static uint32_t Call1Half(Eigen::half *x_data_addr, Eigen::half *y_data_addr,
                            Eigen::half *grid_data_addr,
                            std::vector<int64_t> x_dims,
                            std::vector<int64_t> y_dims, int64_t *y_stride,
                            int64_t *x_stride, int64_t *grid_stride,
                            std::string interpolation_mode,
                            std::string padding_mode, bool align_corners);
  static uint32_t Call2Half(CpuKernelContext &ctx, Eigen::half *x_data_addr,
                            Eigen::half *y_data_addr,
                            Eigen::half *grid_data_addr,
                            std::vector<int64_t> x_dims,
                            std::vector<int64_t> y_dims, int64_t *y_stride,
                            int64_t *x_stride, int64_t *grid_stride,
                            std::string interpolation_mode,
                            std::string padding_mode, bool align_corners);
};
}  // namespace aicpu
#endif
