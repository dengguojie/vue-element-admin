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
#ifndef AICPU_KERNELS_NORMALIZED_CANDIDATE_SAMPLER_H
#define AICPU_KERNELS_NORMALIZED_CANDIDATE_SAMPLER_H

#include <utility>
#include "cpu_kernel.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/range_sampler.h"

namespace aicpu {
class CandidateSamplerMsCpuKernel : public CpuKernel {
 public:
  ~CandidateSamplerMsCpuKernel() = default;

 protected:
  template <class RangeSamplerType>
  uint32_t DoComputeForEachType();
  uint32_t GetInputAndCheck(CpuKernelContext &ctx);
  std::vector<void *> ioAddrs_;

 private:
  int num_true_ = 0;
  int num_sampled_ = 0;
  bool unique_ = true;
  int64_t range_max_ = 0;
  std::unique_ptr<aicpu::cpu::RangeSampler> sampler_;

  uint64_t true_expected_count_size_ = 0;
  int batch_size_ = 0;
  std::vector<int64_t> x_shape_;

  DataType x_dtype_ = DT_INT32;
  DataType true_expected_count_dtype_ = DT_INT32;

  void set_sampler(aicpu::cpu::RangeSampler *sampler) {
    sampler_.reset(sampler);
  }

};  // CandidateSamplerMsCpuKernel

class LogUniformCandidateSamplerMsCpuKernel
    : public CandidateSamplerMsCpuKernel {
 public:
  explicit LogUniformCandidateSamplerMsCpuKernel()
      : CandidateSamplerMsCpuKernel(){};
  ~LogUniformCandidateSamplerMsCpuKernel() = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;
};

class UniformCandidateSamplerMsCpuKernel : public CandidateSamplerMsCpuKernel {
 public:
  explicit UniformCandidateSamplerMsCpuKernel()
      : CandidateSamplerMsCpuKernel(){};
  ~UniformCandidateSamplerMsCpuKernel() = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;
};

}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_CANDIDATE_SAMPLER_H_
