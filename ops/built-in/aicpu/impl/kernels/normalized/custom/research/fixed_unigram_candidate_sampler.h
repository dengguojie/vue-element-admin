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
#ifndef AICPU_KERNELS_NORMALIZED_FIXED_UNIGRAM_CANDIDATE_SAMPLER_H_
#define AICPU_KERNELS_NORMALIZED_FIXED_UNIGRAM_CANDIDATE_SAMPLER_H_

#include <string>
#include <vector>
#include <random>
#include "cpu_kernel.h"
#include "utils/philox_random.h"

namespace aicpu {
class FUCSCpuKernel : public CpuKernel {
 public:
  FUCSCpuKernel() = default;
  ~FUCSCpuKernel() override = default;

  using ResultType = Array<uint32_t, 4>;
  using ResultElementType = uint32_t;
  using Key = Array<uint32_t, 2>;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  int num_true;
  uint num_sampled;
  bool unique;
  uint range_max;
  std::string vocab_file;
  float distortion;
  int num_reserved_ids;
  int num_shards;
  int shard;
  std::vector<float> unigrams;
  int seed;
  int seed2;

  PhiloxRandom generator_;

  float RandFloat();
  uint32_t Uniform(uint32_t n);

  uint64_t New64();
  void InitPhiloxRandom(uint64_t seed, uint64_t seed2);

  ResultType unused_results_;
  int used_result_index_ = PhiloxRandom::kResultElementCount;
  ResultElementType GenerateSingle();

  int num_;
  std::unique_ptr<std::pair<float, int>[]> data_;
  std::vector<float> weights_;
  uint32_t InitDistSampler(std::vector<float> &weights);
  int DistSamplerSample();

  int64_t range_;
  float total_weight_;
  int32_t num_shards_;
  int32_t shard_;
  float ExpectedCountHelper(float p, int batch_size, int num_tries);
  float Probability(int64_t value);
  void FillReservedIds(int32_t num_reserved_ids);
  uint32_t LoadFromFile(std::string vocab_file, float distortion);
  uint32_t LoadFromUnigrams(std::vector<float> &unigrams, float distortion);

  uint32_t FUCSCheck(CpuKernelContext &ctx);
  uint32_t FUCSCompute(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
