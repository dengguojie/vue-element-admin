/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of reshape
 */

#ifndef _AICPU_EDIT_DISTANCE_KERNELS_H_
#define _AICPU_EDIT_DISTANCE_KERNELS_H_

#include <vector>
#include "cpu_kernel.h"

namespace aicpu {
class EditDistanceKernel : public CpuKernel {
public:
    ~EditDistanceKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

protected:
    uint32_t DoCompute();

    uint32_t GetInputAndCheck(CpuKernelContext &ctx);

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
inline int64_t LevenshteinDistance(const std::vector<T> &s, const std::vector<T> &t, const Cmp &cmp)
{
    const int64_t s_size = s.size();
    const int64_t t_size = t.size();

    if (t_size > s_size)
        return LevenshteinDistance(t, s, cmp);

    const T *s_data = s.data();
    const T *t_data = t.data();

    if (t_size == 0)
        return s_size;
    if (s == t)
        return 0;

    // Create work vector
    std::vector<int64_t> scratch_holder(t_size);

    int64_t *scratch = scratch_holder.data();

    // Special case for i = 0: Distance between empty string and string
    // of length j is just j.
    for (int64_t j = 1; j < t_size; ++j)
        scratch[j - 1] = j;

    for (int64_t i = 1; i <= s_size; ++i) {
        // Invariant: scratch[j - 1] equals cost(i - 1, j).
        int substitution_base_cost = i - 1;
        int insertion_cost = i + 1;
        for (int64_t j = 1; j <= t_size; ++j) {
            // Invariants:
            //  scratch[k - 1] = cost(i, k)  for 0 < k < j.
            //  scratch[k - 1] = cost(i - 1, k)  for j <= k <= t_size.
            //  substitution_base_cost = cost(i - 1, j - 1)
            //  insertion_cost = cost(i, j - 1)
            const int replacement_cost = cmp(s_data[i - 1], t_data[j - 1]) ? 0 : 1;
            const int substitution_cost = substitution_base_cost + replacement_cost;
            const int deletion_cost = scratch[j - 1] + 1;

            // Select the cheapest edit.
            const int cheapest = // = cost(i, j)
                std::min(deletion_cost, std::min(insertion_cost, substitution_cost));

            // Restore invariant for the next iteration of the loop.
            substitution_base_cost = scratch[j - 1]; // = cost(i - 1, j)
            scratch[j - 1] = cheapest;               // = cost(i, j)
            insertion_cost = cheapest + 1;           // = cost(i, j) + 1
        }
    }
    return scratch[t_size - 1];
}

template <typename Container1, typename Container2, typename Cmp>
inline int64_t LevenshteinDistance(const Container1 &s, const Container2 &t, const Cmp &cmp)
{
    return LevenshteinDistance(std::vector<typename Container1::value_type>(s.data(), s.data() + s.size()),
        std::vector<typename Container1::value_type>(t.data(), t.data() + t.size()), cmp);
}
} // namespace aicpu
#endif
