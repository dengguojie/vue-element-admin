/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef AICPU_KERNELS_NORMALIZED_STRIDED_SLICE_H_
#define AICPU_KERNELS_NORMALIZED_STRIDED_SLICE_H_

#include <vector>

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace aicpu {
class StridedSliceCpuKernel : public CpuKernel {
 public:
  ~StridedSliceCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

  /**
   * @brief init strided slice params with masks
   * @param x_shape StridedSlice input [x]'s shape
   * @param begin_mask begin mask
   * @param end_mask end mask
   * @param ellipsis_mask ellipsis mask
   * @param new_axis_mask new axis mask
   * @param shrink_axis_mask shrink axis mask
   * @param begin StridedSlice param begin
   * @param end StridedSlice param end
   * @param strides StridedSlice param strides
   * @return status code
   */
  static uint32_t InitParamsWithMasks(const std::vector<int64_t> &x_shape,
                                      int64_t begin_mask, int64_t end_mask,
                                      int64_t ellipsis_mask,
                                      int64_t new_axis_mask,
                                      int64_t shrink_axis_mask,
                                      std::vector<int64_t> &begin,
                                      std::vector<int64_t> &end,
                                      std::vector<int64_t> &strides);

  /**
   * @brief calculate strided slice
   * @param ctx op context
   * @param begin StridedSlice param begin
   * @param end StridedSlice param end
   * @param strides StridedSlice param strides
   * @param x_tensor StridedSlice input [x]
   * @param y_tensor StridedSlice output [y]
   * @return status code
   */
  template <typename T>
  static uint32_t CalStridedSlice(const CpuKernelContext &ctx,
                                  const std::vector<int64_t> &begin,
                                  const std::vector<int64_t> &end,
                                  const std::vector<int64_t> &strides,
                                  const Tensor *x_tensor, Tensor *y_tensor) {
    KERNEL_CHECK_NULLPTR(x_tensor, KERNEL_STATUS_INNER_ERROR,
                        "[CalStridedSlice] check x_tensor is [nullptr].");
    KERNEL_CHECK_NULLPTR(y_tensor, KERNEL_STATUS_INNER_ERROR,
                        "[CalStridedSlice] check y_tensor is [nullptr].");
    T* x_data = static_cast<T *>(x_tensor->GetData());
    KERNEL_CHECK_NULLPTR(x_data, KERNEL_STATUS_INNER_ERROR,
                        "[CalStridedSlice] check x_data is [nullptr].");
    T* y_data = static_cast<T *>(y_tensor->GetData());
    KERNEL_CHECK_NULLPTR(y_data, KERNEL_STATUS_INNER_ERROR,
                        "[CalStridedSlice] check y_data is [nullptr].");
    int64_t x_size = x_tensor->NumElements();
    int64_t y_size = y_tensor->NumElements();

    auto x_tensor_shape = x_tensor->GetTensorShape();
    KERNEL_CHECK_NULLPTR(x_tensor_shape, KERNEL_STATUS_INNER_ERROR,
                        "[CalStridedSlice] check x_tensor_shape is [nullptr].");
    std::vector<int64_t> x_shape = x_tensor_shape->GetDimSizes();
    KERNEL_LOG_INFO("[CalStridedSlice] x_shape: [%s]", VectorToString(x_shape).c_str());

    // convert negative idx to positive
    // calculate y_shape temp with [begin_tmp, end_tmp, strides]
    std::vector<int64_t> begin_tmp = begin;
    std::vector<int64_t> end_tmp = end;
    std::vector<int64_t> y_shape_tmp(x_shape.size());
    for (size_t i = 0; i < begin_tmp.size(); ++i) {
      if (begin_tmp[i] < 0) {
        begin_tmp[i] += x_shape[i];
      }
      begin_tmp[i] = std::max(begin_tmp[i], int64_t(0));
      begin_tmp[i] = std::min(begin_tmp[i], x_shape[i] - 1);
      if (end_tmp[i] < 0) {
        end_tmp[i] += x_shape[i];
      }
      end_tmp[i] = std::max(end_tmp[i], int64_t(-1));
      end_tmp[i] = std::min(end_tmp[i], x_shape[i]);
      int64_t y_range = end_tmp[i] - begin_tmp[i];
      y_shape_tmp[i] = y_range / strides[i];
      if ((y_range % strides[i]) != 0) {
        y_shape_tmp[i] += 1;
      }
    }

    auto shardCal = [&](int64_t start, int64_t end)->void {
      for (int64_t y_idx = start; y_idx < end; ++y_idx) {
        int64_t x_idx = 0;
        int64_t block = 1;
        int64_t y_idx_tmp = y_idx;
        for (int64_t i = x_shape.size() - 1; i > 0; --i) {
          int64_t idx_in_dim = y_idx_tmp % y_shape_tmp[i];
          x_idx += (begin_tmp[i] + idx_in_dim * strides[i]) * block;
          y_idx_tmp = y_idx_tmp / y_shape_tmp[i];
          block *= x_shape[i];
        }
        x_idx += (begin_tmp[0] + y_idx_tmp * strides[0]) * block;
        KERNEL_CHECK_FALSE_VOID((x_idx < x_size),
            "[CalStridedSlice] x_idx [%lld] overflow x_size [%lld].",
            x_idx, x_size);
        y_data[y_idx] = x_data[x_idx];
      }
    };
    return CpuKernelUtils::ParallelFor(ctx, y_size, 1, shardCal);
  }

 private:
  /**
   * @brief parse kernel parms
   * @param ctx op context
   * @return status code
   */
  uint32_t ParseKernelParams(CpuKernelContext &ctx);

  uint32_t ParseIndexInput(CpuKernelContext &ctx, uint32_t index,
                           std::vector<int64_t> &vec);
  uint32_t GetMaskAttr(CpuKernelContext &ctx, std::string attr, int64_t &mask);

 private:
  std::vector<int64_t> begin_;
  std::vector<int64_t> end_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> x_shape_;
  int64_t begin_mask_ = 0;
  int64_t end_mask_ = 0;
  int64_t ellipsis_mask_ = 0;
  int64_t new_axis_mask_ = 0;
  int64_t shrink_axis_mask_ = 0;
};
}  // namespace aicpu
#endif
