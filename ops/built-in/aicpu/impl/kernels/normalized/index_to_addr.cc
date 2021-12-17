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

#include "index_to_addr.h"

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const char *kIndexToAddr = "IndexToAddr";
const char *kMatrix = "Matrix";
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 1;

template <typename T>
uint32_t IndexToAddr(aicpu::CpuKernelContext &ctx) {
  T *base_addr = static_cast<T *>(ctx.Input(0)->GetData());
  T row = static_cast<T *>(ctx.Input(1)->GetData())[0];
  T col = static_cast<T *>(ctx.Input(1)->GetData())[1];
  T *addr_table = static_cast<T *>(ctx.Output(0)->GetData());

  std::vector<int64_t> ori_shape = ctx.GetAttr("ori_shape")->GetListInt();
  std::string ori_storage_mode = ctx.GetAttr("ori_storage_mode")->GetString();
  std::vector<int64_t> block_size = ctx.GetAttr("block_size")->GetListInt();
  std::string block_mode = ctx.GetAttr("block_storage_mode")->GetString();
  int64_t rank_id = ctx.GetAttr("rank_id")->GetInt();
  aicpu::DataType dtype = ctx.GetAttr("dtype")->GetDataType();

  int64_t i = row * block_size[0];
  int64_t j = col * block_size[1];
  KERNEL_CHECK_FALSE((i < ori_shape[0]), aicpu::KERNEL_STATUS_PARAM_INVALID,
                     "Ori shape row index[%ld] "
                     "must be < Attr[ori_shape] shape[0]:%d",
                     i, ori_shape[0])
  KERNEL_CHECK_FALSE((j < ori_shape[1]), aicpu::KERNEL_STATUS_PARAM_INVALID,
                     "Ori shape col index[%ld] "
                     "must be < Attr[ori_shape] shape[0]:%d",
                     j, ori_shape[1])
  int64_t row_size = block_size[1];
  KERNEL_LOG_INFO(
      "Input row[%lld], col[%lld], ori row[%lld], ori col[%lld], "
      "ori mode[%s], block row[%lld], block col[%lld], block mode[%s], "
      "rank id[%lld], dtype[%s].",
      static_cast<int64_t>(row), static_cast<int64_t>(col), ori_shape[0],
      ori_shape[1], ori_storage_mode.c_str(), block_size[0], block_size[1],
      block_mode.c_str(), rank_id, DTypeStr(dtype).c_str());

  for (int64_t r = 0; r < block_size[0]; ++r, ++i) {
    int64_t bias = ori_shape[1] * i + j;
    int64_t bias_addr = bias * aicpu::GetSizeByDataType(dtype);
    addr_table[r * 4] = rank_id;
    addr_table[r * 4 + 1] = base_addr[0] + bias_addr;
    addr_table[r * 4 + 2] = base_addr[1] + bias_addr;
    addr_table[r * 4 + 3] = row_size * aicpu::GetSizeByDataType(dtype);
  }
  return aicpu::KERNEL_STATUS_OK;
}
}  // namespace

namespace aicpu {
uint32_t IndexToAddrCpuKernel::Check(CpuKernelContext &ctx) const {
  Tensor *base_addr = ctx.Input(0);
  Tensor *x = ctx.Input(1);
  Tensor *output = ctx.Output(0);
  DataType base_addr_dt = base_addr->GetDataType();
  DataType x_dt = x->GetDataType();
  KERNEL_CHECK_FALSE((base_addr_dt == x_dt), KERNEL_STATUS_PARAM_INVALID,
                     "Input[base_addr] data type[%s] and input[x] data type[%s] "
                     "must be same.",
                     DTypeStr(base_addr_dt).c_str(), DTypeStr(x_dt).c_str());

  KERNEL_CHECK_FALSE((IsVector(base_addr->GetTensorShape()->GetDimSizes()) &&
                      base_addr->NumElements() == 2),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Input[base_addr] must be a 1D with two elements")
  KERNEL_CHECK_FALSE(
      (IsVector(x->GetTensorShape()->GetDimSizes()) && x->NumElements() == 2),
      KERNEL_STATUS_PARAM_INVALID, "Input[x] must be a vector with shape=[2]")

  std::vector<int64_t> ori_shape = ctx.GetAttr("ori_shape")->GetListInt();
  KERNEL_CHECK_FALSE((IsMatrix(ori_shape)), KERNEL_STATUS_PARAM_INVALID,
                     "Attr[ori_shape] must be a matrix")
  std::string ori_storage_mode = ctx.GetAttr("ori_storage_mode")->GetString();
  KERNEL_CHECK_FALSE((ori_storage_mode == kMatrix), KERNEL_STATUS_PARAM_INVALID,
                     "Attr[ori_storage_mode] value[%s] must be a UT or Matrix",
                     ori_storage_mode.c_str())
  std::vector<int64_t> block_size = ctx.GetAttr("block_size")->GetListInt();
  KERNEL_CHECK_FALSE((IsMatrix(block_size)), KERNEL_STATUS_PARAM_INVALID,
                     "Attr[block_size] must be a matrix")
  KERNEL_CHECK_FALSE((4 * block_size[0] <= output->NumElements()),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Attr[block_size] row[%lld] must be <= "
                     "[%lld]",
                     block_size[0], output->NumElements() / 4)
  std::string block_mode = ctx.GetAttr("block_storage_mode")->GetString();
  KERNEL_CHECK_FALSE(
      (block_mode == kMatrix), KERNEL_STATUS_PARAM_INVALID,
      "Attr[block_storage_mode] value[%s] must be a UT or Matrix",
      block_mode.c_str())

  return KERNEL_STATUS_OK;
}

uint32_t IndexToAddrCpuKernel::Compute(CpuKernelContext &ctx) {
  std::vector<std::string> attr_names = {"ori_shape",  "ori_storage_mode",
                                         "block_size", "block_storage_mode",
                                         "rank_id",    "dtype"};
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum, attr_names),
                      "Check IndexToAddr params failed.");

  KERNEL_HANDLE_ERROR(Check(ctx), "Check IndexToAddr params failed.");

  DataType base_addr_dt = ctx.Input(0)->GetDataType();
  uint32_t ret = KERNEL_STATUS_OK;
  switch (base_addr_dt) {
    case DT_INT64:
      ret = IndexToAddr<int64_t>(ctx);
      break;
    case DT_UINT64:
      ret = IndexToAddr<uint64_t>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("Unsupported input[base_addr] data type[%s]",
                       DTypeStr(base_addr_dt).c_str());
      ret = KERNEL_STATUS_PARAM_INVALID;
  }
  return ret;
}

REGISTER_CPU_KERNEL(kIndexToAddr, IndexToAddrCpuKernel);
}  // namespace aicpu
