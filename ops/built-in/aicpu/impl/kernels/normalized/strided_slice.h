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
    if (x_data == nullptr){
        KERNEL_LOG_WARN("where x_data is a nullptr");
        y_tensor->SetData(nullptr);
        y_tensor->SetTensorShape(nullptr);
        return KERNEL_STATUS_OK;
    }
    T* y_data = static_cast<T *>(y_tensor->GetData());
    KERNEL_CHECK_NULLPTR(y_data, KERNEL_STATUS_INNER_ERROR,
                        "[CalStridedSlice] check y_data is [nullptr].");
    int64_t x_size = x_tensor->NumElements();
    int64_t y_size = y_tensor->NumElements();

    auto x_tensor_shape = x_tensor->GetTensorShape();
    KERNEL_CHECK_NULLPTR(x_tensor_shape, KERNEL_STATUS_INNER_ERROR,
                        "[CalStridedSlice] check x_tensor_shape is [nullptr].");
    std::vector<int64_t> x_shape = x_tensor_shape->GetDimSizes();

    for (size_t i = 0; i < strides.size(); ++i) {
      KERNEL_CHECK_FALSE((strides[i] != 0), KERNEL_STATUS_PARAM_INVALID,
          "[CalStridedSlice] strides[%zu] must be non-zero.", i);
    }

    // convert negative idx to positive
    // calculate y_shape temp with [begin_tmp, end_tmp, strides]
    std::vector<int64_t> begin_tmp = begin;
    std::vector<int64_t> end_tmp = end;
    std::vector<int64_t> y_shape_tmp = CalYShapeTmp(x_shape,
                                                    strides,
                                                    begin_tmp,
                                                    end_tmp);
    for (size_t i = 0; i < y_shape_tmp.size(); ++i) {
      KERNEL_CHECK_FALSE((y_shape_tmp[i] != 0), KERNEL_STATUS_INNER_ERROR,
          "[CalStridedSlice] y_shape_tmp[%zu] must be non-zero.", i);
    }

    auto turboShardCal = [&](int64_t start, int64_t end)->void {
      int64_t factor = x_shape.back() / y_shape_tmp.back();
      int64_t offest = begin_tmp.back();
      for (int64_t y_idx = start; y_idx < end; ++y_idx) {
        int64_t x_idx = y_idx * factor + offest;
        KERNEL_CHECK_FALSE_VOID((x_idx < x_size),
            "[CalStridedSlice] x_idx [%lld] overflow x_size [%lld].",
            x_idx, x_size);
        y_data[y_idx] = x_data[x_idx];
      }
    };

    std::vector<int64_t> block = CalBlocks(x_shape);
    auto shardCal = [&](int64_t start, int64_t end)->void {
      for (int64_t y_idx = start; y_idx < end; ++y_idx) {
        int64_t x_idx = 0;
        int64_t y_idx_tmp = y_idx;
        for (size_t i = x_shape.size() - 1; i > 0; --i) {
          int64_t idx_in_dim = y_idx_tmp % y_shape_tmp[i];
          x_idx += (begin_tmp[i] + idx_in_dim * strides[i]) * block[i];
          y_idx_tmp = y_idx_tmp / y_shape_tmp[i];
        }
        x_idx += (begin_tmp[0] + y_idx_tmp * strides[0]) * block[0];
        KERNEL_CHECK_FALSE_VOID((x_idx < x_size),
            "[CalStridedSlice] x_idx [%lld] overflow x_size [%lld].",
            x_idx, x_size);
        y_data[y_idx] = x_data[x_idx];
      }
    };

    if (IsTurbo(x_shape, y_shape_tmp)) {
      return CpuKernelUtils::ParallelFor(ctx, y_size, 1, turboShardCal);
    } else {
      return CpuKernelUtils::ParallelFor(ctx, y_size, 1, shardCal);
    }
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

  /**
   * @brief convert negative idx to positive
   *        calculate y_shape temp with [begin_tmp, end_tmp, strides]
   * @param x_shape StridedSlice input [x]'s shape
   * @param strides StridedSlice param strides
   * @param begin_tmp StridedSlice begin temp
   * @param end_tmp StridedSlice end temp
   * @return y_shape temp
   */
  static std::vector<int64_t> CalYShapeTmp(
      const std::vector<int64_t> &x_shape,
      const std::vector<int64_t> &strides,
      std::vector<int64_t> &begin_tmp,
      std::vector<int64_t> &end_tmp) {
    std::vector<int64_t> y_shape_tmp(x_shape.size());
    for (size_t i = 0; i < begin_tmp.size(); ++i) {
      if (begin_tmp[i] < 0) {
        begin_tmp[i] += x_shape[i];
      }
      begin_tmp[i] = std::max(begin_tmp[i], int64_t(0));
      begin_tmp[i] = std::min(begin_tmp[i], x_shape[i] - 1);
      if (end_tmp[i] <= 0) {
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
    return std::move(y_shape_tmp);
  }

  /**
   * @brief calculate blocks for x
   * @param x_shape StridedSlice input [x]'s shape
   * @return blocks for x
   */
  static std::vector<int64_t> CalBlocks(const std::vector<int64_t> &x_shape) {
    std::vector<int64_t> block(x_shape.size());
    int64_t block_tmp = 1;
    for (size_t i = x_shape.size() - 1; i > 0; --i) {
      block[i] = block_tmp;
      block_tmp *= x_shape[i];
    }
    block[0] = block_tmp;
    return std::move(block);
  }

  /**
   * @brief check if StridedSlice can be accelerated
   * @param x_shape StridedSlice input [x]'s shape
   * @param y_shape_tmp StridedSlice input [x]'s shape
   * @return it can be accelerated or not
   */
  static bool IsTurbo(const std::vector<int64_t> &x_shape,
                      const std::vector<int64_t> &y_shape_tmp) {
    for (size_t i = 0; i < y_shape_tmp.size(); ++i) {
      if ((i == (y_shape_tmp.size() - 1)) && (y_shape_tmp[i] == 1)) {
        return true;
      }
      if (y_shape_tmp[i] != x_shape[i]) {
        break;
      }
    }
    return false;
  }

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
