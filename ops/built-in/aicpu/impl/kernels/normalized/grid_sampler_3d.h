/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef AICPU_KERNELS_NORMALIZED_GRIDSAMPLER3D_H_
#define AICPU_KERNELS_NORMALIZED_GRIDSAMPLER3D_H_

#include "cpu_kernel.h"
#include "Eigen/Core"

namespace aicpu{
  class GridSampler3DCpuKernel:public CpuKernel{
    public:
      GridSampler3DCpuKernel() = default;
      ~GridSampler3DCpuKernel() override = default;

    protected:
      uint32_t Compute(CpuKernelContext &ctx) override;

    private:
      template <typename T>
      static uint32_t GridSampler3DCompute(CpuKernelContext &ctx);

      static uint32_t GridSampler3DComputeHalf(CpuKernelContext &ctx);

      static bool check_attr(std::string &interpolation_mode, std::string &padding_mode, bool &align_corners,
                             CpuKernelContext &ctx);
      
      static std::vector<int64_t> stride_comput(const std::vector<int64_t> &shape);

      static bool NextIndex(const std::vector<int64_t> &shape, std::vector<int64_t> &iter);

      template <typename T>
      static void bilinear_compute(std::vector<T *> addr, std::vector<T> location, const int64_t y_c,
                                   std::vector<int64_t> x_dims, int64_t x_ptr_NC, std::vector<int64_t> x_stride,
                                   int64_t y_ptr_NCDHW, std::vector<int64_t> y_stride);
      
      static void bilinear_compute_half(std::vector<Eigen::half *> addr, std::vector<float> location, const int64_t y_c,
                                        std::vector<int64_t> x_dims, int64_t x_ptr_NC, std::vector<int64_t> x_stride,
                                        int64_t y_ptr_NCDHW, std::vector<int64_t> y_stride);
      
      template <typename T>
      static void nearest_compute(std::vector<T *> addr, std::vector<T> location, const int64_t y_c,
                                  std::vector<int64_t> x_dims, int64_t x_ptr_NC, std::vector<int64_t> x_stride,
                                  int64_t y_ptr_NCDHW, std::vector<int64_t> y_stride);
      
      static void nearest_compute_half(std::vector<Eigen::half *> addr, std::vector<float> location, const int64_t y_c, 
                                       std::vector<int64_t> x_dims, int64_t x_ptr_NC, std::vector<int64_t> x_stride,
                                       int64_t y_ptr_NCDHW, std::vector<int64_t> y_stride);

      template <typename T>
      static T grid_sampler_compute_source_index(T coord, int64_t size, std::string padding_mode, bool align_corners);
      
      template <typename T>
      static T reflect_coordinates(T coord, int64_t twice_low, int64_t twice_high);

      static bool within_bounds_3d(int64_t d, int64_t h, int64_t w, const std::vector<int64_t> &shape);
  };
}
#endif
