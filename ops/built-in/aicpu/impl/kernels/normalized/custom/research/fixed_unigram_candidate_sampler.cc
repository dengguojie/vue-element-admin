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
#include "fixed_unigram_candidate_sampler.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 3;
const uint32_t kInputNum = 1;
const char *kFUCS = "FixedUnigramCandidateSampler";
}  // namespace

namespace aicpu {
uint32_t FUCSCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "[%s] check input and output number failed.", kFUCS);
  KERNEL_HANDLE_ERROR(FUCSCheck(ctx), "[%s] check params failed.", kFUCS);
  auto data_type = ctx.Input(0)->GetDataType();
  if (data_type != DT_INT64) {
    KERNEL_LOG_ERROR(
        "FixedUnigramCandidateSampler kernel data type [%s] not support.",
        DTypeStr(data_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  uint32_t result = FUCSCompute(ctx);
  if (result != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("FixedUnigramCandidateSampler kernel compute failed.");
    return result;
  }
  return KERNEL_STATUS_OK;
}

uint32_t FUCSCpuKernel::FUCSCheck(CpuKernelContext &ctx) {
  auto true_classes = ctx.Input(0);
  auto sampled_candidates = ctx.Output(0);
  auto true_expected_count = ctx.Output(1);
  auto sampled_expected_count = ctx.Output(2);
  KERNEL_CHECK_NULLPTR(true_classes->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input data failed.")
  KERNEL_CHECK_NULLPTR(sampled_candidates->GetData(),
                       KERNEL_STATUS_PARAM_INVALID, "Get output 0 data failed")
  KERNEL_CHECK_NULLPTR(true_expected_count->GetData(),
                       KERNEL_STATUS_PARAM_INVALID, "Get output 1 data failed")
  KERNEL_CHECK_NULLPTR(sampled_expected_count->GetData(),
                       KERNEL_STATUS_PARAM_INVALID, "Get output 2 data failed")

  auto attr_num_true = ctx.GetAttr("num_true");
  KERNEL_CHECK_NULLPTR(attr_num_true, KERNEL_STATUS_PARAM_INVALID,
                       "Get num_true attr failed.")
  num_true = attr_num_true->GetInt();

  auto attr_num_sampled = ctx.GetAttr("num_sampled");
  KERNEL_CHECK_NULLPTR(attr_num_sampled, KERNEL_STATUS_PARAM_INVALID,
                       "Get num_sampled attr failed.")
  num_sampled = attr_num_sampled->GetInt();

  auto attr_unique = ctx.GetAttr("unique");
  KERNEL_CHECK_NULLPTR(attr_unique, KERNEL_STATUS_PARAM_INVALID,
                       "Get unique attr failed.")
  unique = attr_unique->GetBool();

  auto attr_range_max = ctx.GetAttr("range_max");
  KERNEL_CHECK_NULLPTR(attr_range_max, KERNEL_STATUS_PARAM_INVALID,
                       "Get range_max attr failed.")
  range_max = attr_range_max->GetInt();

  auto attr_vocab_file = ctx.GetAttr("vocab_file");
  KERNEL_CHECK_NULLPTR(attr_vocab_file, KERNEL_STATUS_PARAM_INVALID,
                       "Get vocab_file attr failed.")
  vocab_file = attr_vocab_file->GetString();

  auto attr_distortion = ctx.GetAttr("distortion");
  KERNEL_CHECK_NULLPTR(attr_distortion, KERNEL_STATUS_PARAM_INVALID,
                       "Get distortion attr failed.")
  distortion = attr_distortion->GetFloat();

  auto attr_num_reserved_ids = ctx.GetAttr("num_reserved_ids");
  KERNEL_CHECK_NULLPTR(attr_num_reserved_ids, KERNEL_STATUS_PARAM_INVALID,
                       "Get num_reserved_ids attr failed.")
  num_reserved_ids = attr_num_reserved_ids->GetInt();

  auto attr_num_shards = ctx.GetAttr("num_shards");
  KERNEL_CHECK_NULLPTR(attr_num_shards, KERNEL_STATUS_PARAM_INVALID,
                       "Get num_shards attr failed.")
  num_shards = attr_num_shards->GetInt();

  auto attr_shard = ctx.GetAttr("shard");
  KERNEL_CHECK_NULLPTR(attr_shard, KERNEL_STATUS_PARAM_INVALID,
                       "Get shard attr failed.")
  shard = attr_shard->GetInt();

  auto attr_unigrams = ctx.GetAttr("unigrams");
  KERNEL_CHECK_NULLPTR(attr_unigrams, KERNEL_STATUS_PARAM_INVALID,
                       "Get unigrams attr failed.")
  unigrams = attr_unigrams->GetListFloat();

  auto attr_seed = ctx.GetAttr("seed");
  KERNEL_CHECK_NULLPTR(attr_seed, KERNEL_STATUS_PARAM_INVALID,
                       "Get seed attr failed.")
  seed = attr_seed->GetInt();

  auto attr_seed2 = ctx.GetAttr("seed2");
  KERNEL_CHECK_NULLPTR(attr_seed2, KERNEL_STATUS_PARAM_INVALID,
                       "Get seed2 attr failed.")
  seed2 = attr_seed2->GetInt();

  KERNEL_CHECK_NULLPTR(true_classes->GetTensorShape(),
                       KERNEL_STATUS_PARAM_INVALID,
                       "Get input true_classes shape failed.")
  std::vector<int64_t> shape_true_classes =
      true_classes->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((shape_true_classes.size() == 2),
                     KERNEL_STATUS_PARAM_INVALID,
                     "True_classes must be a matrix.")
  KERNEL_CHECK_FALSE(
      (shape_true_classes.at(1) == num_true), KERNEL_STATUS_PARAM_INVALID,
      "True_classes must have num_true columns, expected: [%zu], was: [%zu].",
      shape_true_classes.at(1), num_true)
  if (unique) {
    KERNEL_CHECK_FALSE((num_sampled <= range_max), KERNEL_STATUS_PARAM_INVALID,
                       "Num_sampled cannot be greater than range_max, but got "
                       "[%zu] and [%zu].",
                       num_sampled, range_max)
  }
  KERNEL_CHECK_FALSE((!vocab_file.empty() || !unigrams.empty()),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Must provide either vocab_file or unigrams.")
  KERNEL_CHECK_FALSE((vocab_file.empty() || unigrams.empty()),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Must only provide one of vocab_file and unigrams.")
  return KERNEL_STATUS_OK;
}

float FUCSCpuKernel::RandFloat() {
  uint32_t x = GenerateSingle();
  uint32_t man = x & 0x7fffffu;  // 23 bit mantissa
  uint32_t exp = static_cast<uint32_t>(127);
  uint32_t val = (exp << 23) | man;
  float result;
  memcpy(&result, &val, sizeof(val));
  return result - 1.0f;
}

uint32_t FUCSCpuKernel::Uniform(uint32_t n) {
  if (n == 0) {
    return GenerateSingle() * n;
  } else if (0 == (n & (n - 1))) {
    return GenerateSingle() & (n - 1);
  } else {
    uint32_t range = ~static_cast<uint32_t>(0);
    uint32_t rem = (range % n) + 1;
    uint32_t rnd;
    do {
      rnd = GenerateSingle();
    } while (rnd < rem);
    return rnd % n;
  }
}

uint64_t FUCSCpuKernel::New64() {
  std::random_device device("/dev/urandom");
  std::mt19937_64 rng = std::mt19937_64(device());
  return (rng)();
}

void FUCSCpuKernel::InitPhiloxRandom(uint64_t seed, uint64_t seed2) {
  if (seed == 0 && seed2 == 0) {
    seed = New64();
    seed2 = New64();
  }
  generator_ = PhiloxRandom(seed, seed2);
}

FUCSCpuKernel::ResultElementType FUCSCpuKernel::GenerateSingle() {
  if (used_result_index_ == PhiloxRandom::kResultElementCount) {
    unused_results_ = generator_();
    used_result_index_ = 0;
  }
  return unused_results_[used_result_index_++];
}

uint32_t FUCSCpuKernel::InitDistSampler(std::vector<float> &weights) {
  int n = weights.size();
  num_ = n;
  data_.reset(new std::pair<float, int>[n]);
  std::unique_ptr<double[]> pr(new double[n]);

  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += weights[i];
    data_[i].second = -1;
  }

  if (sum == 0) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  std::vector<int> high;
  high.reserve(n);
  std::vector<int> low;
  low.reserve(n);

  for (int i = 0; i < n; i++) {
    double p = (weights[i] * n) / sum;
    pr[i] = p;
    if (p < 1.0) {
      low.push_back(i);
    } else {
      high.push_back(i);
    }
  }

  while (!high.empty() && !low.empty()) {
    int l = low.back();
    low.pop_back();
    int h = high.back();
    high.pop_back();

    data_[l].second = h;
    double remaining = pr[h] - (1.0 - pr[l]);
    pr[h] = remaining;

    if (remaining < 1.0) {
      low.push_back(h);
    } else {
      high.push_back(h);
    }
  }
  for (int i = 0; i < n; i++) {
    data_[i].first = pr[i];
  }
  for (size_t i = 0; i < high.size(); i++) {
    int idx = high[i];
    data_[idx].first = 1.0;
    data_[idx].second = idx;
  }
  for (size_t i = 0; i < low.size(); i++) {
    int idx = low[i];
    data_[idx].first = 1.0;
    data_[idx].second = idx;
  }
  return KERNEL_STATUS_OK;
}

int FUCSCpuKernel::DistSamplerSample() {
  float r = RandFloat();
  int idx = Uniform(num_);
  if (r < data_[idx].first) return idx;
  return data_[idx].second;
}

float FUCSCpuKernel::ExpectedCountHelper(float p, int batch_size,
                                         int num_tries) {
  if (num_tries == batch_size) {
    return p * batch_size;
  }
  return -std::expm1(num_tries * std::log1p(-p));
}

float FUCSCpuKernel::Probability(int64_t value) {
  if (value < 0 || static_cast<size_t>(value) >= weights_.size()) {
    return 0.0;
  }
  return weights_.at(value) / total_weight_;
}

void FUCSCpuKernel::FillReservedIds(int32_t num_reserved_ids) {
  for (int32_t word_id = 0; word_id < num_reserved_ids; ++word_id) {
    if (word_id % num_shards_ == shard_) weights_.push_back(0.0);
  }
}

uint32_t FUCSCpuKernel::LoadFromFile(std::string vocab_file, float distortion) {
  std::ifstream file(vocab_file, std::ios::in);
  if (!file.good()) {
    KERNEL_LOG_ERROR("Fail to open the file [%s].", vocab_file.c_str());
    return KERNEL_STATUS_INNER_ERROR;
  }

  std::string line;
  int32_t word_id = weights_.size();
  while (std::getline(file, line)) {
    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string str;
    while (std::getline(ss, str, ',')) {
      cols.push_back(str);
    }
    if (cols.empty()) {
      continue;
    }
    if (word_id % num_shards_ == shard_) {
      float w = 0.0;
      std::istringstream iss(cols.at(cols.size() - 1));
      iss >> w;
      w = std::pow(w, distortion);
      total_weight_ += w;
      weights_.push_back(w);
    }
    ++word_id;
  }
  return KERNEL_STATUS_OK;
}

uint32_t FUCSCpuKernel::LoadFromUnigrams(std::vector<float> &unigrams,
                                         float distortion) {
  int32_t word_id = weights_.size();
  for (float w : unigrams) {
    if (word_id % num_shards_ == shard_) {
      w = std::pow(w, distortion);
      total_weight_ += w;
      weights_.push_back(w);
    }
    ++word_id;
  }
  return KERNEL_STATUS_OK;
}

uint32_t FUCSCpuKernel::FUCSCompute(CpuKernelContext &ctx) {
  auto true_classes = reinterpret_cast<int64_t *>(ctx.Input(0)->GetData());
  auto sampled_candidates =
      reinterpret_cast<int64_t *>(ctx.Output(0)->GetData());
  auto true_expected_count =
      reinterpret_cast<float *>(ctx.Output(1)->GetData());
  auto sampled_expected_count =
      reinterpret_cast<float *>(ctx.Output(2)->GetData());
  uint64_t size = ctx.Input(0)->NumElements();

  used_result_index_ = 4;
  total_weight_ = 0.0;
  num_shards_ = num_shards;
  shard_ = shard;
  InitPhiloxRandom(seed, seed2);
  FillReservedIds(num_reserved_ids);
  if (unigrams.empty()) {
    if (LoadFromFile(vocab_file, distortion) != KERNEL_STATUS_OK) {
      return KERNEL_STATUS_INNER_ERROR;
    }
  } else {
    if (LoadFromUnigrams(unigrams, distortion) != KERNEL_STATUS_OK) {
      return KERNEL_STATUS_INNER_ERROR;
    }
  }
  KERNEL_CHECK_FALSE((range_max == weights_.size()),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Check failed: range == weights_.size() ([%d] vs. [%d]).",
                     range_max, weights_.size())

  if (InitDistSampler(weights_) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  int batch_size = num_sampled;
  int num_tries;
  std::vector<int64_t> avoided_values;
  if (unique) {
    std::unordered_set<int64_t> used(batch_size);
    used.insert(avoided_values.begin(), avoided_values.end());
    int num_picked = 0;
    num_tries = 0;
    while (num_picked < batch_size) {
      num_tries++;
      int64_t value = DistSamplerSample();
      if (used.insert(value).second) {
        sampled_candidates[num_picked++] = value;
      }
    }
  } else {
    for (int i = 0; i < batch_size; i++) {
      sampled_candidates[i] = DistSamplerSample();
    }
    num_tries = batch_size;
  }

  if (num_sampled > 0) {
    for (int i = 0; i < batch_size; i++) {
      sampled_expected_count[i] = ExpectedCountHelper(
          Probability(sampled_candidates[i]), batch_size, num_tries);
    }
  }
  for (size_t i = 0; i < size; i++) {
    true_expected_count[i] = ExpectedCountHelper(Probability(true_classes[i]),
                                                 batch_size, num_tries);
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kFUCS, FUCSCpuKernel);
}  // namespace aicpu