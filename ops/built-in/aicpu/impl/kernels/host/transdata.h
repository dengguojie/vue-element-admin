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
#ifndef AICPU_KERNELS_HOST_TRANSDATA_H_
#define AICPU_KERNELS_HOST_TRANSDATA_H_

#include "cpu_kernel.h"

namespace aicpu {
struct TransArgs {
  const uint8_t *data;
  std::vector<int64_t> src_shape;
  std::vector<int64_t> dst_shape;
  DataType src_data_type;
};
class TransDataCpuKernel : public CpuKernel {
 public:
  ~TransDataCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  uint32_t DealData(T *input_Data, T *output_data, Tensor *input_tensor,
                    Tensor *out_put_tensor, int64_t group);
  /**
   * Get format from primary and sub-format,
   * in bits field:
   * ----------------------------------
   * |  1 byte  |   2 bytes  | 1 byte |
   * |----------|------------|--------|
   * | reserved | sub-format | format |
   * ----------------------------------
   * @param primary_format
   * @param sub_format
   * @return
   */
  int32_t GetPrimaryFormat(int32_t format);
  uint32_t FormatTransferHwcnToFZC04(TransArgs &args, uint8_t *output_addr,
                                     uint64_t length);
  uint32_t PaddingOne(TransArgs &args, std::shared_ptr<uint8_t> &dst);
  uint32_t PaddingTwo(TransArgs &args, std::shared_ptr<uint8_t> &dst);
  uint32_t GetPaddingOneShape(const TransArgs &args,
                              std::vector<int64_t> &dst_shape);
  uint32_t GetPaddingTwoShape(const TransArgs &args,
                              std::vector<int64_t> &dst_shape);
  uint32_t Transpose(TransArgs &args, const std::vector<int64_t> &perm_arg,
                     std::shared_ptr<uint8_t> &dst);
  int64_t GetCubeSizeByDataType(DataType data_type);

  bool IsOriginSupportFormatTransfer(Format src_fromat, Format dst_format);

  uint32_t NewCompute(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif