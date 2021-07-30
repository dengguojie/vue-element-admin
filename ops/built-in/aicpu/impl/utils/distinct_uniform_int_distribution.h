#ifndef AICPU_UTILS_DISTINCT_UNIFORM_INT_DISTRIBUTION_H_
#define AICPU_UTILS_DISTINCT_UNIFORM_INT_DISTRIBUTION_H_

#include <random>
#include <unordered_set>
#include "log.h"
#include "status.h"

namespace aicpu {
template <typename IntType = int>
class DistinctUniformIntDistribution {
 public:
  using ResultType = IntType;

 private:
  using SetType = std::unordered_set<ResultType>;
  using DistrType = std::uniform_int_distribution<ResultType>;

 public:
  DistinctUniformIntDistribution(ResultType inf, ResultType sup)
      : inf_(inf), sup_(sup), range_(sup_ - inf_ + 1), distr_(inf_, sup_) {
  }
  ~DistinctUniformIntDistribution() = default;
  void Reset() {
    uset_.clear();
    distr_.reset();
  }

  template <typename Generator>
  ResultType exec(Generator &engine) {
    if (not(uset_.size() < range_)) {
      std::terminate();
    }
    ResultType res;
    do {
      res = distr_(engine);
    } while (uset_.count(res) > 0);
    uset_.insert(res);
    return res;
  }

 private:
  const ResultType inf_;
  const ResultType sup_;
  const size_t range_ = 0;
  DistrType distr_;
  SetType uset_;
};
}  // namespace aicpu

#endif  // AICPU_UTILS_DISTINCT_UNIFORM_INT_DISTRIBUTION_H_
