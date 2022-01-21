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
#ifndef AICPU_KERNELS_NORMALIZED_RANDOM_UNIFORM_H_
#define AICPU_KERNELS_NORMALIZED_RANDOM_UNIFORM_H_
#define EIGEN_USE_THREADS
#define EIGEN_USE_SIMPLE_THREAD_POOL

#include "cpu_kernel.h"

#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace aicpu
{
  class RandomGammaCpuKernel : public CpuKernel
  {
  public:
    RandomGammaCpuKernel() = default;
    ~RandomGammaCpuKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

  private:
    /**
   * @brief generate data
   * @param ctx cpu kernel context
   * @param output using to output data
   * @return status if success
   */
    template <typename T>
    void Generate(CpuKernelContext &ctx, Tensor *output);
  };

  uint64_t get_random_seed() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t rnd = ::random() ^ ts.tv_nsec;
    return rnd;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint64_t
  PCG_XSH_RS_state(uint64_t seed) {
    seed = seed ? seed : get_random_seed();
    return seed * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
  }

  template <typename T>
  T RandomToTypeGamma(uint64_t *state,
                      double alpha)
  {
    using Eigen::numext::exp;
    using Eigen::numext::log;
    using Eigen::numext::pow;

    T result;

    // 若alpha==1,则变为指数分布
    if (alpha == 1.0)
    {
      T u;
      u = static_cast<T>(Eigen::internal::RandomToTypeUniform<double>(state));
      result = -T(1.0) * log(T(1.0) - u);
    }
    else
    { // if alpha != 1.0
      // Transformation-rejection from pairs of uniform and normal random
      // variables. http://dl.acm.org/citation.cfm?id=358414
      //
      // The algorithm has an acceptance rate of ~95% for small alpha (~1),
      // and higher accept rates for higher alpha, so runtime is
      // O(NumAlphas * NumSamples * k) with k ~ 1 / 0.95.
      //
      // For alpha<1, we add one to d=alpha-1/3, and multiply the final
      // result by uniform()^(1/alpha)
      const bool alpha_less_than_one = alpha < 1.0;
      const double d = alpha + (alpha_less_than_one ? 2.0 / 3 : -1.0 / 3);
      const double c = 1.0 / 3 / sqrt(d);

      // Keep trying until we don't reject a sample. In practice, we will
      // only reject ~5% at worst, for low alpha near 1.
      while (true)
      {
        double x =
            Eigen::internal::RandomToTypeNormal<double>(state);

        double v = 1 + c * x;
        if (v <= 0)
        {
          continue;
        }
        v = v * v * v;

        double u =
            Eigen::internal::RandomToTypeUniform<double>(state);
        // The first option in the if is a "squeeze" short-circuit to
        // dodge the two logs. Magic constant sourced from the paper
        // linked above. Upward of .91 of the area covered by the log
        // inequality is covered by the squeeze as well (larger coverage
        // for smaller values of alpha).
        if ((u < 1 - 0.0331 * (x * x) * (x * x)) ||
            (log(u) < 0.5 * x * x + d * (1 - v + log(v))))
        {
          double res = d * v;
          if (alpha_less_than_one)
          {
            double b =
                Eigen::internal::RandomToTypeUniform<double>(state);
            res *= pow(b, 1 / static_cast<double>(alpha));
          }
          result = static_cast<T>(res);
          break;
        }
      } // while: true
    }
    return result;
  }

  template <typename T>
  class GammaRandomGenerator
  {
  public:
    // Uses the given "seed" if non-zero, otherwise uses a random seed.
    explicit GammaRandomGenerator(double alpha, uint64_t seed = 0)
    {
      m_state = PCG_XSH_RS_state(seed);
      m_alpha = alpha;
    }

    void setAlpha(double alpha) { m_alpha = alpha; }

    T gen() const
    {
      T result = RandomToTypeGamma<T>(&m_state, m_alpha);
      return result;
    }

  private:
    double m_alpha;
    mutable uint64_t m_state;
  };

} // namespace aicpu
#endif // AICPU_KERNELS_NORMALIZED_RANDOM_GAMMA_H_
