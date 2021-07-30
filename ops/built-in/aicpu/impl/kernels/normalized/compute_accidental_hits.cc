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
#include "compute_accidental_hits.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace {
const char *kComputeAccidentalHits = "ComputeAccidentalHits";
}

namespace aicpu {
uint32_t ComputeAccidentalHitsMsCpuKernel::GetInputAndCheck(
    CpuKernelContext &ctx) {
  AttrValue *num_true = ctx.GetAttr("num_true");
  KERNEL_CHECK_NULLPTR(num_true, KERNEL_STATUS_PARAM_INVALID,
                       "Get attr:[num_true] failed.");
  num_true_ = num_true->GetInt();

  // input0: true_classes
  Tensor *x_tensor = ctx.Input(0);
  x_dtype_ = static_cast<DataType>(x_tensor->GetDataType());
  std::shared_ptr<TensorShape> x_shape = x_tensor->GetTensorShape();
  for (auto i = 0; i < x_shape->GetDims(); i++) {
    x_shape_.emplace_back(x_shape->GetDimSize(i));
  }

  // input_1: sampled_candidates
  Tensor *sampled_candidates_tensor = ctx.Input(1);
  std::shared_ptr<TensorShape> paddings_shape =
      sampled_candidates_tensor->GetTensorShape();
  for (auto i = 0; i < paddings_shape->GetDims(); i++) {
    in_sampled_candidates_shape_.emplace_back(paddings_shape->GetDimSize(i));
  }

  if (in_sampled_candidates_shape_.size() != 1) {
    KERNEL_LOG_ERROR(
        "The sampled_candidates(input[1]:[%zu]-D) must be a vector(1-D),"
        "which is typically an output from CandidateSampler",
        in_sampled_candidates_shape_.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (x_shape_.size() != 2 || x_shape_[1] != num_true_) {
    KERNEL_LOG_ERROR(
        "The input[0] must be 2-D (batch_size * num_true matrix),"
        "but got dim of input[0]: [%zu]",
        x_shape_.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // output_2: sampled_candidates
  Tensor *weights_tensor = ctx.Output(2);
  weights_dtype_ = static_cast<DataType>(weights_tensor->GetDataType());

  inputs_.push_back(x_tensor);
  inputs_.push_back(sampled_candidates_tensor);
  // get output Tensors
  const int kNumOutput = 3;
  for (int i = 0; i < kNumOutput; ++i) {
    Tensor *tensor = ctx.Output(i);
    KERNEL_CHECK_NULLPTR(tensor, KERNEL_STATUS_PARAM_INVALID,
                         "Get output tensor[%d] failed", i)

    if (tensor->NumElements() < 1) {
      KERNEL_LOG_ERROR(
          "The number of elements in output[%d]:[%lld]"
          "should be >= [1]",
          i, tensor->NumElements());
      return KERNEL_STATUS_PARAM_INVALID;
    }
    outputs_.push_back(tensor);
  }
  return KERNEL_STATUS_OK;
}

uint32_t ComputeAccidentalHitsMsCpuKernel::Compute(CpuKernelContext &ctx) {
  uint32_t res = GetInputAndCheck(ctx);
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res,
                     "GetInputAndCheck failed.");

  std::map<int,
           std::map<int, std::function<uint32_t(
                             CpuKernelContext &, int, std::vector<Tensor *> &,
                             std::vector<Tensor *> &, std::vector<int64_t> &,
                             std::vector<int64_t> &)>>>
      calls;
  calls[DT_INT32][DT_FLOAT16] = CalComputeAccidentalHits<int32_t, Eigen::half>;
  calls[DT_INT32][DT_FLOAT] = CalComputeAccidentalHits<int32_t, float>;
  calls[DT_INT32][DT_DOUBLE] = CalComputeAccidentalHits<int32_t, double>;
  calls[DT_INT64][DT_FLOAT16] = CalComputeAccidentalHits<int64_t, Eigen::half>;
  calls[DT_INT64][DT_FLOAT] = CalComputeAccidentalHits<int64_t, float>;
  calls[DT_INT64][DT_DOUBLE] = CalComputeAccidentalHits<int64_t, double>;

  return calls[x_dtype_][weights_dtype_](ctx, num_true_, inputs_, outputs_,
                                         x_shape_,
                                         in_sampled_candidates_shape_);
}

REGISTER_CPU_KERNEL(kComputeAccidentalHits, ComputeAccidentalHitsMsCpuKernel);
}  // namespace aicpu
