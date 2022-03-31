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
#ifndef AICPU_KERNELS_NORMALIZED_PRODENVMATA_CALCRIJ_H
#define AICPU_KERNELS_NORMALIZED_PRODENVMATA_CALCRIJ_H

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace aicpu {
  class ProdEnvMatACalcRijCpuKernel : public CpuKernel {
    public:
      ProdEnvMatACalcRijCpuKernel() = default;
      ~ProdEnvMatACalcRijCpuKernel() = default;
      uint32_t Compute(CpuKernelContext &ctx) override;

    private:
      struct InputNlist {
        // Number of core region atoms
        int32_t nlocnum;

        // Array stores the core region atom's index
        int32_t *ilist;

        // Array stores the core region atom's neighbor atom number
        int32_t *numneigh;

        // Array stores the core region atom's neighbor index
        int32_t (*firstneigh)[1024];

        int32_t *nallmaptable;

        InputNlist() : nlocnum(0), ilist(NULL), numneigh(NULL), firstneigh(NULL), nallmaptable(NULL){};
        InputNlist(int32_t nlocnum_, int32_t *ilist_, int32_t *numneigh_,
                   int32_t (*firstneigh_)[1024], int32_t *nallmaptable_)
                   : nlocnum(nlocnum_),
                     ilist(ilist_),
                     numneigh(numneigh_),
                     firstneigh(firstneigh_),
                     nallmaptable(nallmaptable_){};
        ~InputNlist(){};
      };

      struct NeighborInfo {
        int32_t type;
        double dist;
        int32_t index;
        NeighborInfo() : type(0), dist(0), index(0) {}
        NeighborInfo(int32_t tt, double dd, int32_t ii)
                     : type(tt), dist(dd), index(ii) {}
        bool operator<(const NeighborInfo &b) const {
          return (type < b.type ||
                  (type == b.type &&
                  (dist < b.dist || (dist == b.dist && index < b.index))));
        }
      };

      template <typename FPTYPE>
      struct SingleNatomInfo {
        std::vector<FPTYPE> d_distance_a;
        std::vector<int32_t> fmt_nlist_a;
      };

      uint32_t DoCompute(CpuKernelContext &ctx);
      template <typename FPTYPE>
      uint32_t DoProdEnvMatACalcRijCompute(CpuKernelContext &ctx);
      uint32_t GetInputAndCheck(CpuKernelContext &ctx);
      int32_t max_numneigh(const InputNlist &nlist);
      void cum_sum(std::vector<int64_t> &sec, const std::vector<int64_t> &n_sel);
      template <typename FPTYPE>
      int32_t format_nlist_i_cpu(SingleNatomInfo<FPTYPE> &singleNatomInfo,
                                 const int32_t &i_idx,
                                 const std::vector<int32_t> &nei_idx_a,
                                 std::vector<int64_t> &sec_a,
                                 const FPTYPE *coord, const int32_t *type,
                                 float rcutsquared);
      template <typename FPTYPE>
      void env_mat_a_cpu(FPTYPE *rij_a, const int32_t &i_idx,
                         const std::vector<int32_t> &fmt_nlist_a,
                         std::vector<int64_t> &sec_a, const FPTYPE *coord);
      template <typename FPTYPE>
      void prod_env_mat_a_rij_cal(FPTYPE *rij, CpuKernelContext &ctx,
                                  int32_t batchIndex, int32_t nnei,
                                  std::vector<int64_t> &sec_a);
      template <typename FPTYPE>
      inline FPTYPE dot3(const FPTYPE *r0,
                        const FPTYPE *r1);
  };
}  // namespace aicpu
#endif