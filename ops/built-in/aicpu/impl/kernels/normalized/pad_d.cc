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
#include <algorithm>
#include "cpu_tensor.h"
#include "cpu_tensor_shape.h"
#include "cpu_types.h"
#include "utils/kernel_util.h"
#include "cpu_kernel_utils.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "status.h"
#include "log.h"
#include "securec.h"
namespace {
// Add pad op
const char *kPadD = "PadD";
const char *kPad = "Pad";
const uint32_t kInput_padD = 1;
const uint32_t kInput_pad = 2;
const uint32_t kOutput = 1;
}

namespace aicpu {
int64_t multi(int64_t x, int64_t rank, std::vector<int64_t> &dims_y) {
  if (x == rank) {
    return 1;
  } else {
    return dims_y[x] * multi(x + 1, rank, dims_y);
  }
}

int64_t sumLR(int64_t x, std::vector<int64_t> &vec) {
  if (x == -1) {
    return 0;
  } else {
    return vec[x] + sumLR(x - 1, vec);
  }
}

template<typename T>
void GetIntConstData(const Tensor& const_tensor, std::vector<std::vector<int64_t>> & pad_data){
  size_t pading_size = 0;
  std::vector<int64_t> vector_data;
  const T*const_data_ptr = reinterpret_cast<const T*>(const_tensor.GetData());
  if (const_data_ptr == nullptr) {
    KERNEL_LOG_DEBUG("const_data_pttr is null");
    pad_data.clear();
    return;
  }
  pading_size = const_tensor.GetDataSize() / sizeof(T);
  for (size_t i = 0; i < pading_size; ++i){
    vector_data.push_back((T)((*(const_data_ptr + 1))));
    KERNEL_LOG_DEBUG("idex:value = %d", i);
  }
  for (size_t i = 1; i < vector_data.size(); i += 2) {
    std::vector<int64_t> one_value;
    one_value.push_back(vector_data[i - 1]);
    one_value.push_back(vector_data[i]);
    pad_data.push_back(one_value);
  }
}

template <typename T>
uint32_t DoCompute(CpuKernelContext &ctx) {
  Tensor *x = ctx.Input(0);
  // input
  auto xdata = reinterpret_cast<T *>(x->GetData());
  int64_t x_num = x->NumElements();
  std::vector<int64_t> dims_x = x->GetTensorShape()->GetDimSizes();
  const int64_t rank = x->GetTensorShape()->GetDims();
  std::vector<T> x_data;
  for (auto i = 0; i < x_num; i++) {
    x_data.push_back(xdata[i]);
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
  // get pad data for padd
  if (strcmp(ctx.GetOpType().c_str(), kPadD) == 0) {
      pad = ctx.GetAttr("paddings")->GetListListInt();
  } else {
    // get pad data for pad
    Tensor *pading_tensor = ctx.Input(1);
    auto pading_type = pading_tensor->GetDataType();
    std::map<int, std::function<void(const Tensor&, std::vector<std::vector<int64_t>>&)>> calls_GetIntConstData;
    calls_GetIntConstData[DT_INT32] = GetIntConstData<int32_t>;
    calls_GetIntConstData[DT_INT64] = GetIntConstData<int64_t>;
    if (calls_GetIntConstData.find(pading_type) == calls_GetIntConstData.end()) {
      KERNEL_LOG_ERROR("Pad Kernel data type [%s] is not supported", DTypeStr(pading_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
    calls_GetIntConstData[pading_type](*pading_tensor, pad);
  }
  // the constraint of column
  const uint32_t col = 2;
  if (pad.empty()) {
    auto ret = memcpy_s(y->GetData(), y->GetDataSize(), x->GetData(), x->GetDataSize());
    KERNEL_CHECK_FALSE((ret == EOK), KERNEL_STATUS_PARAM_INVALID,
                       "PadD Memcpy failed, result = [%d].", ret);
    return KERNEL_STATUS_OK;
  }
  if ((pad.size() != (size_t)rank) || (pad[0].size() != (size_t)col)) {
    KERNEL_LOG_ERROR("PadD attr data format not support");
      return KERNEL_STATUS_PARAM_INVALID;
  }
  for (int64_t t = 0; t < rank; t++) {
    pad1.push_back(pad[t][0]);
    pad2.push_back(pad[t][1]);
  }
  // offset
  std::vector<int64_t> offsetL;
  std::vector<int64_t> offsetR;
  for (int64_t k = 0; k < rank; k ++) {  // rank从大到小
    offsetL.push_back(pad1[k] * multi(k + 1, rank, dims_y));
    offsetR.push_back(pad2[k] * multi(k + 1, rank, dims_y));
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

uint32_t PadDCpuKernel::Compute(CpuKernelContext &ctx) {
  uint32_t res = KERNEL_STATUS_OK;
  // get the type of op
  std::string op = ctx.GetOpType();
  KERNEL_LOG_INFO("%s starts.", op.c_str());
  if (strcmp(op.c_str(), kPadD) == 0) {
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInput_padD, kOutput),
                        "[%s] checks input and output number failed.", kPadD);
    KERNEL_CHECK_NULLPTR(ctx.GetAttr("paddings"), KERNEL_STATUS_PARAM_INVALID,
                         "[%s] get attribute failed.", kPadD);
  } else if (strcmp(op.c_str(), kPad) == 0) {
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInput_pad, kOutput),
                        "[%s] checks input and output number failed.", kPad);
  } else {
    KERNEL_LOG_ERROR("the [%s] op is not  supported in PadDCpuKernel", op.c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *x = ctx.Input(0);
  // check type
  auto x_type = x->GetDataType();
  // processed sfor padD
  if (strcmp(op.c_str(), kPadD) == 0) {
    SetMap_padd();
    if (calls_padd.find(x_type) == calls_padd.end()) {
      KERNEL_LOG_ERROR("PadD Kernel data type [%s] is not supported", DTypeStr(x_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
    res = calls_padd[x_type](ctx);
  } else {
    SetMap_pad();
    if (calls_pad.find(x_type) == calls_pad.end()) {
      KERNEL_LOG_ERROR("Pad Kernel data type [%s] is not supported", DTypeStr(x_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
    res = calls_pad[x_type](ctx);
  }
  if (res != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

void PadDCpuKernel::SetMap_padd() {
  calls_padd[DT_FLOAT] = DoCompute<float>;
  calls_padd[DT_FLOAT16] = DoCompute<Eigen::half>;
  calls_padd[DT_INT32] = DoCompute<int32_t>;
}

void PadDCpuKernel::SetMap_pad() {
  calls_pad[DT_INT8] = DoCompute<int8_t>;
  calls_pad[DT_UINT8] = DoCompute<uint8_t>;
  calls_pad[DT_INT16] = DoCompute<int16_t>;
  calls_pad[DT_UINT16] = DoCompute<uint16_t>;
  calls_pad[DT_INT32] = DoCompute<int32_t>;
  calls_pad[DT_UINT32] = DoCompute<uint32_t>;
  calls_pad[DT_INT64] = DoCompute<int64_t>;
  calls_pad[DT_UINT64] = DoCompute<uint64_t>;
  calls_pad[DT_BOOL] = DoCompute<bool>;
  calls_pad[DT_FLOAT16] = DoCompute<Eigen::half>;
  calls_pad[DT_FLOAT] = DoCompute<float>;
  calls_pad[DT_DOUBLE] = DoCompute<double>;
}

REGISTER_CPU_KERNEL(kPadD, PadDCpuKernel);
REGISTER_CPU_KERNEL(kPad, PadDCpuKernel);
}