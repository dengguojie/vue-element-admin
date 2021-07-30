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
#ifndef AICPU_KERNELS_NORMALIZED_EDIT_DISTANCE_H_
#define AICPU_KERNELS_NORMALIZED_EDIT_DISTANCE_H_

#include <vector>
#include "cpu_kernel.h"

namespace aicpu {
class EditDistanceMsCpuKernel : public CpuKernel {
 public:
  ~EditDistanceMsCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t DoCompute();

  uint32_t GetInputAndCheck(CpuKernelContext &ctx);

 private:
  bool normalize_ = true;
  std::vector<Tensor *> inputs_;
  std::vector<Tensor *> outputs_;
  DataType param_type_ = DT_INT32;
};

// Calculate the Levenshtein Edit Distance between two contiguous
// sequences, s and t, of type T.
//
// The Levenshtein distance is a symmetric distance defined as the
// smallest number of insertions, deletions, and substitutions
// required to convert sequence s to t (and vice versa).
// Note, this distance does not consider transpositions.
//
// For more details and a reference implementation, see:
//   https://en.wikipedia.org/wiki/Levenshtein_distance
template <typename T, typename Cmp>
int64_t LevenshteinDistance(const std::vector<T> &s,
                                   const std::vector<T> &t, const Cmp &cmp) {
  const int64_t kSSize = s.size();
  const int64_t kTSize = t.size();

  if (kTSize > kSSize) {
    return LevenshteinDistance(t, s, cmp);
  }
  const T *s_data = s.data();
  const T *t_data = t.data();
  if (kTSize == 0) {
    return kSSize;
  }
  if (s == t) {
    return 0;
  }

  // Create work vector
  std::vector<int64_t> scratch_holder(kTSize);
  int64_t *scratch = scratch_holder.data();

  // Special case for i = 0: Distance between empty string and string
  // of length j is just j.
  for (int64_t j = 1; j < kTSize; ++j)
    scratch[j - 1] = j;

  for (int64_t i = 1; i <= kSSize; ++i) {
    // Invariant: scratch[j - 1] equals cost(i - 1, j).
    int substitution_base_cost = i - 1;
    int insertion_cost = i + 1;
    for (int64_t j = 1; j <= kTSize; ++j) {
      // Invariants:
      //  scratch[k - 1] = cost(i, k)  for 0 < k < j.
      //  scratch[k - 1] = cost(i - 1, k)  for j <= k <= kTSize.
      //  substitution_base_cost = cost(i - 1, j - 1)
      //  insertion_cost = cost(i, j - 1)
      const int kReplacementCost = cmp(s_data[i - 1], t_data[j - 1]) ? 0 : 1;
      const int kSubstitutionCost = substitution_base_cost + kReplacementCost;
      const int kDeletionCost = scratch[j - 1] + 1;

      // Select the cheapest edit.
      const int kCheapest =  // = cost(i, j)
          std::min(kDeletionCost, std::min(insertion_cost, kSubstitutionCost));

      // Restore invariant for the next iteration of the loop.
      substitution_base_cost = scratch[j - 1];  // = cost(i - 1, j)
      scratch[j - 1] = kCheapest;                // = cost(i, j)
      insertion_cost = kCheapest + 1;            // = cost(i, j) + 1
    }
  }
  return scratch[kTSize - 1];
}

template <typename Container1, typename Container2, typename Cmp>
int64_t LevenshteinDistance(const Container1 &s, const Container2 &t,
                                   const Cmp &cmp) {
  return LevenshteinDistance(std::vector<typename Container1::value_type>(
                                 s.data(), s.data() + s.size()),
                             std::vector<typename Container1::value_type>(
                                 t.data(), t.data() + t.size()),
                             cmp);
}
}  // namespace aicpu
#endif
