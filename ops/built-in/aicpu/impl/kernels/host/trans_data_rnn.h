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
#ifndef AICPU_KERNELS_HOST_TRANS_DATA_RNN_H_
#define AICPU_KERNELS_HOST_TRANS_DATA_RNN_H_

#include "cpu_kernel.h"

namespace aicpu{

class TransDataRNNCpuKernel : public CpuKernel {
 public:
  TransDataRNNCpuKernel();
  ~TransDataRNNCpuKernel() override = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t GenDataNdRnnBias(std::vector<int64_t> &dims, int32_t hiddenSize,
                            const Tensor *srcTensor, Tensor *dstTensor);
  uint32_t GenDataFractalZnCase1(std::vector<int64_t> &dims, int32_t hiddenSize, int32_t inputSize,
                                 const Tensor *srcTensor, Tensor *dstTensor);
  uint32_t GenDataFractalZnCase2(std::vector<int64_t> &dims, int32_t hiddenSize,
                                 const Tensor *srcTensor, Tensor *dstTensor);
  uint32_t GenDataFractalZn(std::vector<int64_t> &dims, int32_t hiddenSize, int32_t inputSize,
                            const Tensor *srcTensor, Tensor *dstTensor);
  uint32_t GetInputAttrs(CpuKernelContext &ctx, int32_t &inputSize, int32_t &hiddenSize,
                         std::string &srcFormat, std::string &dstFormat);
};

}  // namespace aicpu

#endif //AICPU_KERNELS_HOST_TRANS_DATA_RNN_H_
