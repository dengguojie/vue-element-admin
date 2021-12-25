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

#include "multinomial_alias_draw.h"

#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *kMultinomialAliasDraw = "MultinomialAliasDraw";

#define MULTINOMIAL_ALIAS_DRAW_COMPUTE_CASE(DTYPE, TYPE, CTX)          \
  case (DTYPE): {                                                      \
    uint32_t result = MultinomialAliasDrawCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                                  \
      KERNEL_LOG_ERROR("MultinomialAliasDraw kernel compute failed."); \
      return result;                                                   \
    }                                                                  \
    break;                                                             \
  }
}  // namespace

namespace aicpu {
uint32_t MultinomialAliasDrawCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(
      NormalCheck(ctx, kInputNum, kOutputNum),
      "MultinomialAliasDraw check input and output number failed.");
  KERNEL_HANDLE_ERROR(MultinomialAliasDrawParamCheck(ctx),
                      "MultinomialAliasDraw check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    MULTINOMIAL_ALIAS_DRAW_COMPUTE_CASE(DT_FLOAT, float, ctx)
    MULTINOMIAL_ALIAS_DRAW_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR(
          "MultinomialAliasDraw kernel data type [%s] not support.",
          DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t MultinomialAliasDrawCpuKernel::MultinomialAliasDrawParamCheck(
    CpuKernelContext &ctx) {
  // the non null of input, output has been verified in NormalCheck
  Tensor *q = ctx.Input(0);
  Tensor *j = ctx.Input(1);
  Tensor *y = ctx.Output(0);
  AttrValue *num_samples_ = ctx.GetAttr("num_samples");
  auto num_samples = (num_samples_ == nullptr) ? -1 : (num_samples_->GetInt());
  DataType q_type = q->GetDataType();
  DataType j_type = j->GetDataType();
  KERNEL_CHECK_FALSE((q_type == DT_FLOAT || q_type == DT_DOUBLE),
                     KERNEL_STATUS_PARAM_INVALID,
                     "The data type of q need be DT_FLOAT or DT_DOUBLE.")
  KERNEL_CHECK_FALSE((j_type == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of j need be DT_INT64.")
  KERNEL_CHECK_FALSE((num_samples > 0), KERNEL_STATUS_PARAM_INVALID,
                     "The number of num_samples should be greater than 0.")
  auto input_q_Shape = q->GetTensorShape();
  KERNEL_CHECK_FALSE(input_q_Shape->GetDims() == 1, KERNEL_STATUS_PARAM_INVALID,
                     "Input q must be 1D.")
  auto input_j_Shape = j->GetTensorShape();
  KERNEL_CHECK_FALSE(input_j_Shape->GetDims() == 1, KERNEL_STATUS_PARAM_INVALID,
                     "Input j must be 1D.")
  int64_t data_q_num = ctx.Input(0)->NumElements();
  int64_t data_j_num = ctx.Input(1)->NumElements();
  KERNEL_CHECK_FALSE((data_q_num == data_j_num), KERNEL_STATUS_PARAM_INVALID,
                    "The input data number of j need be equal to data number q.")
  KERNEL_LOG_DEBUG(
      "MultinomialAliasDrawCpuKernel[%s], input0: size[%llu];"
      "input1: size[%llu], output: size[%llu]",
      ctx.GetOpType().c_str(), q->GetDataSize(), j->GetDataSize(),
      y->GetDataSize());
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MultinomialAliasDrawCpuKernel::MultinomialAliasDrawCompute(
    CpuKernelContext &ctx) {
  int data_num = ctx.Input(0)->NumElements();
  auto q_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto j_x = reinterpret_cast<int64_t *>(ctx.Input(1)->GetData());
  int num_samples = ctx.GetAttr("num_samples")->GetInt();
  AttrValue *attr_seed_ = ctx.GetAttr("seed");
  int attr_seed = (attr_seed_ == nullptr) ? -1 : (attr_seed_->GetInt());
  auto out_y = reinterpret_cast<int64_t *>(ctx.Output(0)->GetData());
  srand(attr_seed);
  for (int i = 0; i < num_samples; i++) {
    int t = rand() % data_num;
    double r = (rand() % 10000) / 10000.0;
    if (r < *(q_x + t)) {
      *(out_y + i) = t;
    } else
      *(out_y + i) = *(j_x + t);
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kMultinomialAliasDraw, MultinomialAliasDrawCpuKernel);
}  // namespace aicpu
