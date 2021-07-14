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
#ifndef AICPU_KERNELS_NORMALIZED_DYNAMIC_STITCH_H_
#define AICPU_KERNELS_NORMALIZED_DYNAMIC_STITCH_H_

#include <type_traits>
#include "cpu_kernel.h"

namespace aicpu {
class DynamicStitchKernel : public CpuKernel {
 public:
  ~DynamicStitchKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  static bool SameExtraShape(const Tensor *data0, const Tensor *indices0,
                             const Tensor *data1, const Tensor *indices1) {
    int64_t indices0_dim = indices0->GetTensorShape()->GetDims();
    int64_t indices1_dim = indices1->GetTensorShape()->GetDims();

    const int extra0 = data0->GetTensorShape()->GetDims() - indices0_dim;
    const int extra1 = data1->GetTensorShape()->GetDims() - indices1_dim;
    if (extra0 != extra1) {
      return false;
    }
    for (int i = 0; i < extra0; i++) {
      if (data0->GetTensorShape()->GetDimSize(indices0_dim + i) !=
          data1->GetTensorShape()->GetDimSize(indices1_dim + i)) {
        return false;
      }
    }
    return true;
  }

  static bool StartsWith(const std::shared_ptr<TensorShape> &shape,
                         const std::shared_ptr<TensorShape> &prefix) {
    if (shape->GetDims() < prefix->GetDims()) {
      return false;
    }
    for (int i = 0; i < prefix->GetDims(); ++i) {
      if (shape->GetDimSize(i) != prefix->GetDimSize(i)) {
        return false;
      }
    }
    return true;
  }

  template <typename Ta, typename Tb>
  static bool FastBoundsCheck(const Ta index, const Tb limit) {
    static_assert(std::is_integral<Ta>::value && std::is_integral<Tb>::value,
                  "FastBoundsCheck can only be used on integer types.");
    typedef typename std::make_unsigned<decltype(index + limit)>::type UIndex;
    return static_cast<UIndex>(index) < static_cast<UIndex>(limit);
  }

  uint32_t GetInputAndCheck(CpuKernelContext &ctx, int *first_dim_size,
                            int *data_elements_size);
  int n_ = 1;
  DataType input_dtype_ = DT_INT32;
};
}  // namespace aicpu
#endif
