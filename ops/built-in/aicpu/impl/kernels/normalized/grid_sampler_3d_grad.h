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
#ifndef AICPU_KERNELS_NORMALIZED_GRIDSAMPLER3DGRAD_H_
#define AICPU_KERNELS_NORMALIZED_GRIDSAMPLER3DGRAD_H_

#include "cpu_kernel.h"
#include "Eigen/Core"

namespace aicpu{
  class GridSampler3DGradCpuKernel:public CpuKernel{
    public:
      GridSampler3DGradCpuKernel() = default;
      ~GridSampler3DGradCpuKernel() override = default;

    protected:
      uint32_t Compute(CpuKernelContext &ctx) override;

    private:
      template <typename T>
      static uint32_t GridSampler3DGradCompute(CpuKernelContext &ctx);

      static uint32_t GridSampler3DGradComputeHalf(CpuKernelContext &ctx);

      static bool check_attr(std::string &interpolation_mode, std::string &padding_mode, bool &align_corners,
                             CpuKernelContext &ctx);

      static std::vector<int64_t> stride_comput(const std::vector<int64_t> &shape);

      template <typename T>
      static void bilinear_compute(const std::vector<T *> &addr, const std::vector<T> &location,
          const std::vector<int64_t> &NDHW, const int64_t &dgrid_ptr_NDHW, const int64_t &x_ptr_N,
          const std::vector<int64_t> &x_dims, const std::vector<std::vector<int64_t>> &strides, T gx_mult, T gy_mult,
          T gz_mult);

      static void bilinear_compute_half(const std::vector<Eigen::half *> &addr, const std::vector<float> &location,
          const std::vector<int64_t> &NDHW, const int64_t &dgrid_ptr_NDHW, const int64_t &x_ptr_N,
          const std::vector<int64_t> &x_dims, const std::vector<std::vector<int64_t>> &strides, float gx_mult,
          float gy_mult, float gz_mult);

      template <typename T>
      static void nearest_compute(const std::vector<T *> &addr, const std::vector<T> &location,
          const std::vector<int64_t> &NDHW, const int64_t &dgrid_ptr_NDHW,
          const std::vector<std::vector<int64_t>> &vecs);
      
      static void nearest_compute_half(const std::vector<Eigen::half *> &addr, const std::vector<float> &location,
          const std::vector<int64_t> &NDHW, const int64_t &dgrid_ptr_NDHW,
          const std::vector<std::vector<int64_t>> &vecs);

      template <typename T>
      static T grid_sampler_compute_source_index_set_grad(
          T coord, int64_t size, std::string padding_mode, bool align_corners, T *grad_x);
      
      template <typename T>
      static T clip_coordinates_set_grad(T x, int64_t clip_limit, T *grad_x);

      template <typename T>
      static T reflect_coordinates_set_grad(T x, int64_t twice_low, int64_t twice_high, T *grad_x);

      template <typename T>
      static void safe_add_3d(T *data, const std::vector<int64_t> &loc, const std::vector<int64_t> &stride,
                              const std::vector<int64_t> &shape, T delta);

      static bool within_bounds_3d(const std::vector<int64_t> &loc, const std::vector<int64_t> &shape);
  };
}
#endif
