/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#include "candidate_sampler.h"
#include <securec.h>
#include "utils/range_sampler.h"
#include "utils/kernel_util.h"

namespace {
const char *kLogUniformCandidateSampler = "LogUniformCandidateSampler";
const char *kUniformCandidateSampler = "UniformCandidateSampler";
}  // namespace
namespace aicpu {
uint32_t CandidateSamplerMsCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  AttrValue *num_true = ctx.GetAttr("num_true");
  KERNEL_CHECK_NULLPTR(num_true, KERNEL_STATUS_PARAM_INVALID,
                       "Get attr:[num_true] failed.");
  num_true_ = num_true->GetInt();

  AttrValue *num_sampled = ctx.GetAttr("num_sampled");
  KERNEL_CHECK_NULLPTR(num_sampled, KERNEL_STATUS_PARAM_INVALID,
                       "Get attr:[num_sampled] failed.");
  num_sampled_ = num_sampled->GetInt();

  AttrValue *unique = ctx.GetAttr("unique");
  KERNEL_CHECK_NULLPTR(unique, KERNEL_STATUS_PARAM_INVALID,
                       "Get attr:[unique] failed.");
  unique_ = unique->GetBool();

  AttrValue *range_max = ctx.GetAttr("range_max");
  KERNEL_CHECK_NULLPTR(range_max, KERNEL_STATUS_PARAM_INVALID,
                       "Get attr:[range_max] failed.");
  range_max_ = range_max->GetInt();

  // input0: true_classes
  Tensor *x_tensor = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(x_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "Get tensor[0] failed.")
  x_dtype_ = static_cast<DataType>(x_tensor->GetDataType());
  std::shared_ptr<TensorShape> x_shape = x_tensor->GetTensorShape();
  KERNEL_CHECK_NULLPTR(x_shape, KERNEL_STATUS_PARAM_INVALID,
                       "The value of x_shape is null.")
  x_shape_ = x_shape->GetDimSizes();
  if (x_shape_.size() != 2) {
    KERNEL_LOG_ERROR("The input:[0] must be a 2-D, but got [%zu]-D",
                     x_shape_.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (x_shape_[1] != num_true_) {
    KERNEL_LOG_ERROR(
        "The input[0] must have "
        "num_true columns, expected: [%lld] was: [%d]",
        x_shape_[1], num_true_);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  batch_size_ = x_shape->GetDimSize(0);
  if (x_dtype_ != DT_INT64) {
    KERNEL_LOG_ERROR("Invalid type of input[0]: [%s], should be [%s].",
                     DTypeStr(x_dtype_).c_str(), DTypeStr(DT_INT64).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // output_2: sampled_candidates
  Tensor *true_expected_count_tensor = ctx.Output(1);
  true_expected_count_size_ = true_expected_count_tensor->GetDataSize();
  true_expected_count_dtype_ =
      static_cast<DataType>(true_expected_count_tensor->GetDataType());
  if (true_expected_count_dtype_ != DT_FLOAT) {
    KERNEL_LOG_ERROR("Invalid type of output[1]: [%s], should be [%s].",
                     DTypeStr(true_expected_count_dtype_).c_str(),
                     DTypeStr(DT_FLOAT).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  ioAddrs_.push_back(reinterpret_cast<void *>(x_tensor->GetData()));
  ioAddrs_.push_back(reinterpret_cast<void *>(ctx.Output(0)->GetData()));
  ioAddrs_.push_back(
      reinterpret_cast<void *>(true_expected_count_tensor->GetData()));
  ioAddrs_.push_back(reinterpret_cast<void *>(ctx.Output(2)->GetData()));
  KERNEL_CHECK_FALSE(
      (ioAddrs_.size() == 4), KERNEL_STATUS_PARAM_INVALID,
      "The size of input and output must be [4], but got: [%zu].",
      ioAddrs_.size());
  return KERNEL_STATUS_OK;
}

template <class RangeSamplerType>
uint32_t CandidateSamplerMsCpuKernel::DoComputeForEachType() {
  const int64_t kBatchSize = x_shape_[0];
  // input
  int64_t *true_classes = reinterpret_cast<int64_t *>(ioAddrs_[0]);
  aicpu::cpu::ArraySlice<int64_t> true_candidate(
      true_classes, true_classes + kBatchSize * num_true_);

  aicpu::cpu::MutableArraySlice<int64_t> sampled_candidate(num_sampled_);
  aicpu::cpu::MutableArraySlice<float> true_expected_count(kBatchSize *
                                                           num_true_);
  aicpu::cpu::MutableArraySlice<float> sampled_expected_count(num_sampled_);

  set_sampler(new (std::nothrow) RangeSamplerType(range_max_));
  KERNEL_CHECK_NULLPTR(sampler_, KERNEL_STATUS_PARAM_INVALID,
                       "Get RangeSampler failed.");
  if (unique_ && num_sampled_ > static_cast<int>(sampler_->range())) {
    KERNEL_LOG_ERROR(
        "Sampler's range is too small [%lld], should be >= num_sampled:[%d].",
        sampler_->range(), num_sampled_);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // Pick sampled candidates.
  uint32_t sample_ret = sampler_->SampleBatchGetExpectedCount(
      unique_, sampled_candidate, sampled_expected_count,
      true_candidate, true_expected_count);
  KERNEL_CHECK_FALSE((sample_ret == KERNEL_STATUS_OK), sample_ret,
                     "Sampler failed!");

  int true_count_size = kBatchSize * num_true_ * sizeof(float);
  int ret = memcpy_s(reinterpret_cast<void *>(ioAddrs_[1]),
                     num_sampled_ * sizeof(int64_t),
                     reinterpret_cast<void *>(&sampled_candidate.front()),
                     sampled_candidate.size() * sizeof(int64_t));
  KERNEL_CHECK_FALSE((ret == EOK), KERNEL_STATUS_PARAM_INVALID,
                     "Memcpy failed, result = [%d].", ret);
  ret =
      memcpy_s(reinterpret_cast<void *>(ioAddrs_[2]), true_expected_count_size_,
               reinterpret_cast<void *>(&true_expected_count.front()),
               true_count_size);
  KERNEL_CHECK_FALSE((ret == EOK), KERNEL_STATUS_PARAM_INVALID,
                     "Memcpy failed, result = [%d].", ret);
  ret = memcpy_s(reinterpret_cast<void *>(ioAddrs_[3]),
                 num_sampled_ * sizeof(float),
                 reinterpret_cast<void *>(&sampled_expected_count.front()),
                 sampled_expected_count.size() * sizeof(float));
  KERNEL_CHECK_FALSE((ret == EOK), KERNEL_STATUS_PARAM_INVALID,
                     "Memcpy failed, result = [%d].", ret);
  return KERNEL_STATUS_OK;
}

uint32_t LogUniformCandidateSamplerMsCpuKernel::Compute(CpuKernelContext &ctx) {
  uint32_t res = LogUniformCandidateSamplerMsCpuKernel::GetInputAndCheck(ctx);
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res,
                     "GetInputAndCheck failed.");

  LogUniformCandidateSamplerMsCpuKernel::DoComputeForEachType<
      aicpu::cpu::LogUniformSampler>();
  return KERNEL_STATUS_OK;
}

uint32_t UniformCandidateSamplerMsCpuKernel::Compute(CpuKernelContext &ctx) {
  uint32_t res = UniformCandidateSamplerMsCpuKernel::GetInputAndCheck(ctx);
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res,
                     "GetInputAndCheck failed.");

  UniformCandidateSamplerMsCpuKernel::DoComputeForEachType<
      aicpu::cpu::UniformSampler>();
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kLogUniformCandidateSampler,
                    LogUniformCandidateSamplerMsCpuKernel);
REGISTER_CPU_KERNEL(kUniformCandidateSampler,
                    UniformCandidateSamplerMsCpuKernel);
}  // namespace aicpu
