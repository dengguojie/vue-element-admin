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

#include "multinomial_alias_setup.h"

#include <stack>

#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "securec.h"

namespace {
const uint32_t kOutputNum = 2;
const uint32_t kInputNum = 1;
const char *kMultinomialAliasSetup = "MultinomialAliasSetup";
const double ZERO = 0.;

#define MULTINOMIAL_ALIAS_SETUP_COMPUTE_CASE(DTYPE, TYPE, CTX)              \
  case (DTYPE): {                                                           \
    uint32_t result = MultinomialAliasSetupCompute<TYPE>(CTX);              \
    if (result != KERNEL_STATUS_OK) {                                       \
      KERNEL_LOG_ERROR("MultinomialAliasSetup kernel compute failed.");     \
      return result;                                                        \
    }                                                                       \
    break;                                                                  \
  }
}

namespace aicpu {
uint32_t MultinomialAliasSetupCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "MultinomialAliasSetup check input and output number failed.");
  KERNEL_HANDLE_ERROR(MultinomialAliasSetupParamCheck(ctx), "MultinomialAliasSetup check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    MULTINOMIAL_ALIAS_SETUP_COMPUTE_CASE(DT_FLOAT, float, ctx)
    MULTINOMIAL_ALIAS_SETUP_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("MultinomialAliasSetup kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t MultinomialAliasSetupCpuKernel::MultinomialAliasSetupParamCheck(CpuKernelContext &ctx) {
  // the non null of input, output has been verified in NormalCheck
  Tensor *probs = ctx.Input(0);
  auto probs_shape = probs->GetTensorShape();
  KERNEL_CHECK_FALSE(probs_shape->GetDims() == 1, KERNEL_STATUS_PARAM_INVALID,
                     "Input must be 1D.")
  DataType probs_type = probs->GetDataType();
  KERNEL_CHECK_FALSE((probs_type == DT_FLOAT || probs_type == DT_DOUBLE), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of probs need be DT_FLOAT or DT_DOUBLE.")
  int64_t data_num = ctx.Input(0)->NumElements();
  if (probs_type == DT_DOUBLE) {
    auto probs_x = reinterpret_cast<double *>(ctx.Input(0)->GetData());
    for (int64_t i = 0; i < data_num; i++) {
      KERNEL_CHECK_FALSE((*(probs_x + i)) >= (ZERO), KERNEL_STATUS_PARAM_INVALID,
                         "probs[%llu] must be at least more than 0.", i);
    }
  } else {
    auto probs_x = reinterpret_cast<float *>(ctx.Input(0)->GetData());
    for (int64_t i = 0; i < data_num; i++) {
      KERNEL_CHECK_FALSE((*(probs_x + i)) >= (ZERO), KERNEL_STATUS_PARAM_INVALID,
                         "probs[%llu] must be at least more than 0.", i);
    }
  }
  KERNEL_LOG_DEBUG(
      "MultinomialAliasSetupCpuKernel[%s], probs: size[%llu];", ctx.GetOpType().c_str(), data_num);
  return KERNEL_STATUS_OK;
}

template <typename T>
void MultinomialAliasSetupCpuKernel::MultinomialAliasSetupCal(CpuKernelContext &ctx,
                                                              std::vector<T> &accept,
                                                              std::vector<int64_t> &alias,
                                                              double &max) {
  int64_t data_num = ctx.Input(0)->NumElements();
  std::vector<T> area_ratio;
  std::stack<int> large;
  std::stack<int> small;
  size_t large_idx = -1;
  size_t small_idx = -1;

  auto probs_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  for (int64_t i = 0; i < data_num; i++) {
    area_ratio.push_back((*(probs_x + i))*data_num);
  }
  for (int64_t i = 0; i < data_num; i++) {
    if (area_ratio[i] > 1.0) {
      large.push(i);
    } else {
      small.push(i);
    }
  }
  while (!large.empty()&&!small.empty()) {
    small_idx = small.top();
    small.pop();
    large_idx = large.top();
    large.pop();
    accept[small_idx] = area_ratio[small_idx];
    alias[small_idx] = large_idx;
    area_ratio[large_idx] = area_ratio[large_idx] + area_ratio[small_idx] - 1;
    if (area_ratio[large_idx] < 1) {
      small.push(large_idx);
    } else {
      large.push(large_idx);
    }
  }
  while (!large.empty()) {
    large_idx = large.top();
    large.pop();
    if (area_ratio[large_idx] > max) {
      max=area_ratio[large_idx];
    }
    accept[large_idx] = 1;
  }
  while (!small.empty()) {
    small_idx = small.top();
    small.pop();
    accept[small_idx] = 1;
  }
  return;
}

template <typename T>
uint32_t MultinomialAliasSetupCpuKernel::MultinomialAliasSetupCompute(CpuKernelContext &ctx) {
  int64_t data_num = ctx.Input(0)->NumElements();
  auto out_j = reinterpret_cast<int64_t *>(ctx.Output(0)->GetData());
  auto out_q = reinterpret_cast<T *>(ctx.Output(1)->GetData());
  std::vector<T> accept(data_num, -1);
  std::vector<int64_t> alias(data_num, -1);
  double max = -1;
  MultinomialAliasSetupCal(ctx, accept, alias, max);
  if (max>1) {
    for (int64_t i = 0; i < data_num; i++) {
      if (accept[i] < 1) {
        accept[i] /= max;
      }
      *(out_j + i) = alias[i];
      *(out_q + i) = accept[i];
    }
  } else {
    for (int64_t i = 0; i < data_num; i++) {
      *(out_j + i) = alias[i];
      *(out_q + i) = accept[i];
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kMultinomialAliasSetup, MultinomialAliasSetupCpuKernel);
}  // namespace aicpu
