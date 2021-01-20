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
   * @param begin StridedSlice param begin
   * @param end StridedSlice param end
   * @param strides StridedSlice param strides
   * @param x_tensor StridedSlice input [x]
   * @param y_tensor StridedSlice output [y]
   * @return status code
   */
  template <typename T>
  static uint32_t CalStridedSlice(const std::vector<int64_t> &begin,
                                  const std::vector<int64_t> &end,
                                  const std::vector<int64_t> &strides,
                                  const Tensor *x_tensor, Tensor *y_tensor) {
    KERNEL_CHECK_NULLPTR(x_tensor, KERNEL_STATUS_INNER_ERROR,
                        "[CalStridedSlice] check x_tensor is [nullptr].");
    KERNEL_CHECK_NULLPTR(x_tensor, KERNEL_STATUS_INNER_ERROR,
                        "[CalStridedSlice] check y_tensor is [nullptr].");
    T* x_data = static_cast<T *>(x_tensor->GetData());
    KERNEL_CHECK_NULLPTR(x_data, KERNEL_STATUS_INNER_ERROR,
                        "[CalStridedSlice] check x_data is [nullptr].");
    T* y_data = static_cast<T *>(y_tensor->GetData());
    KERNEL_CHECK_NULLPTR(y_data, KERNEL_STATUS_INNER_ERROR,
                        "[CalStridedSlice] check y_data is [nullptr].");
    int64_t x_size = x_tensor->NumElements();
    int64_t y_size = y_tensor->NumElements();

    std::vector<int64_t> begin_tmp = begin;
    auto tensor_shape = x_tensor->GetTensorShape();
    KERNEL_CHECK_NULLPTR(tensor_shape, KERNEL_STATUS_INNER_ERROR,
                        "[CalStridedSlice] check tensor_shape is [nullptr].");
    std::vector<int64_t> x_shape = tensor_shape->GetDimSizes();
    int64_t y_pos = 0;
    while (y_pos < y_size) {
      int64_t x_pos = 0;
      int64_t block = 1;
      bool carry_flag = true;
      for (int64_t i = x_shape.size() - 1; i >= 0; --i) {
        int64_t factor =
            (strides[i] > 0) ? begin_tmp[i] : (2 * begin[i] - begin_tmp[i]);
        x_pos += factor * block;
        block *= x_shape[i];
      }
      for (int64_t i = x_shape.size() - 1; i >= 0; --i) {
        if (carry_flag) {
          begin_tmp[i] += std::abs(strides[i]);
        }
        int64_t size_i = (end[i] > begin[i]) ? end[i] : (2 * begin[i] - end[i]);
        if (begin_tmp[i] >= size_i) {
          begin_tmp[i] = begin[i];
          carry_flag = true;
        } else {
          carry_flag = false;
        }
      }
      KERNEL_CHECK_FALSE((x_pos < x_size), KERNEL_STATUS_INNER_ERROR,
          "[CalStridedSlice] x_pos [%lld] overflow x_size [%lld].",
          x_pos, x_size);
      y_data[y_pos++] = x_data[x_pos];
    }

    return KERNEL_STATUS_OK;
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
