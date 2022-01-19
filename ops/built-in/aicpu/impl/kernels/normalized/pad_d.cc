/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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

#include "pad_d.h"

#include "cpu_tensor.h"
#include <vector>
#include <algorithm>
#include "cpu_tensor_shape.h"
#include "cpu_types.h"
#include "utils/kernel_util.h"
#include "cpu_kernel_utils.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "status.h"
#include "log.h"
#include "securec.h"

namespace {
const char *kPadD = "PadD";
const uint32_t kInput = 1;
const uint32_t kOutput = 1;

#define RETURN_PAD_COMPUTE_CASE(DTYPE, TYPE, CTX)        \
  case (DTYPE): {                                        \
    uint32_t result = DoCompute<TYPE>(CTX);              \
    if (result != KERNEL_STATUS_OK) {                    \
      KERNEL_LOG_ERROR("PadD kernel compute failed.");   \
      return result;                                     \
    }                                                    \
    break;                                               \
  }
}

namespace aicpu {
uint32_t PadDCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("PadD starts.");
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInput, kOutput),
                      "[%s] checks input and output number failed.", kPadD);
  Tensor *x = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(x, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get input_data failed.", kPadD);
  KERNEL_CHECK_NULLPTR(ctx.GetAttr("paddings"), KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get attribute failed.", kPadD);
  // check type
  auto x_type = x->GetDataType();
  switch (x_type) {
    RETURN_PAD_COMPUTE_CASE(DT_FLOAT, float, ctx)
    RETURN_PAD_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    RETURN_PAD_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    default:
      KERNEL_LOG_ERROR("PadD kernel data type [%s] not support.",
                       DTypeStr(x_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

int64_t PadDCpuKernel::multi(int64_t x, int64_t rank, std::vector<int64_t> &dims_y) {
  if (x == rank) {
    return 1;
  } else {
    return dims_y[x] * multi(x + 1, rank, dims_y);
  }
}

int64_t PadDCpuKernel::sumLR(int64_t x, std::vector<int64_t> &vec) {
  if (x == -1) {
    return 0;
  } else {
    return vec[x] + sumLR(x - 1, vec);
  }
}

template <typename T>
uint32_t PadDCpuKernel::DoCompute(CpuKernelContext &ctx) {
  Tensor *x = ctx.Input(0);
  // input
  auto xdata = reinterpret_cast<T *>(x->GetData());
  int64_t x_num = x->NumElements();
  std::vector<int64_t> dims_x = x->GetTensorShape()->GetDimSizes();
  std::vector<T> x_data;
  for (auto i = 0; i < x_num; i++) {
    x_data.emplace_back(xdata[i]);
  }
  // output
  Tensor *y = ctx.Output(0);
  auto y_data = reinterpret_cast<T *>(y->GetData());
  auto y_shape = y->GetTensorShape();
  std::vector<int64_t> dims_y = y_shape->GetDimSizes();
  std::vector<T> y_ori;
  // paddings
  std::vector<std::vector<int64_t>> pad;
  std::vector<int64_t> pad1, pad2;
  pad = ctx.GetAttr("paddings")->GetListListInt();
  if (pad.empty()) {
    auto ret = memcpy_s(y->GetData(), y->GetDataSize(), x->GetData(), x->GetDataSize());
    KERNEL_CHECK_FALSE((ret == EOK), KERNEL_STATUS_PARAM_INVALID,
                       "PadD Memcpy failed, result = [%d].", ret);
    return KERNEL_STATUS_OK;
  }
  int64_t rank = pad.size();
  for (int64_t t = 0; t < rank; t++) {
    pad1.emplace_back(pad[t][0]);
    pad2.emplace_back(pad[t][1]);
  }
    // offset
  std::vector<int64_t> offsetL;
  std::vector<int64_t> offsetR;
  for (int64_t k = 0; k < rank; k ++) {  // rank从大到小
    offsetL.emplace_back(pad1[k] * multi(k + 1, rank, dims_y));
    offsetR.emplace_back(pad2[k] * multi(k + 1, rank, dims_y));
  }
    // push data into y_ori
  int64_t step = dims_x[rank - 1];
  y_ori.insert(y_ori.end(), sumLR(rank - 1, offsetL), (T)0);
  for (int64_t i = step; i < x_num - 1; i = i + step) {
    y_ori.insert(y_ori.end(), x_data.begin() + i - step, x_data.begin() + i);
    for (int64_t j = 0; j < rank; j++) {
      if (i % multi(j, rank, dims_x) == 0) {
        y_ori.insert(y_ori.end(), offsetR[j], (T)0);
        y_ori.insert(y_ori.end(), offsetL[j], (T)0);
      }
    }
  }
  y_ori.insert(y_ori.end(), x_data.begin() + x_num - step, x_data.begin() + x_num);
  y_ori.insert(y_ori.end(), sumLR(rank - 1, offsetR), (T)0);
  for (size_t i = 0; i < y_ori.size(); i ++) {
    y_data[i] = y_ori[i];
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kPadD, PadDCpuKernel);
}