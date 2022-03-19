/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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

#include "prodenvmata_calcrij.h"

#include <iostream>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "securec.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/kernel_util.h"

namespace
{
  const char *kProdEnvMatACalcRij = "ProdEnvMatACalcRij";
  const uint32_t kInputNum = 5;
  const uint32_t kOutputNum = 6;
  const uint32_t kIndexZero = 0;
  const uint32_t kIndexOne = 1;
  const uint32_t kIndexTwo = 2;
  const uint32_t kIndexThree = 3;
  const uint32_t kIndexFour = 4;
  const uint32_t kIndexFive = 5;
  const uint32_t coordinateXyzNum = 3;
  const uint32_t neighborMaxNum = 1024;
} // namespace

namespace aicpu
{
  template <typename FPTYPE>
  uint32_t ProdEnvMatACalcRijCpuKernel::DoProdEnvMatACalcRijCompute(CpuKernelContext &ctx) {
    Tensor *coord_tensor = ctx.Input(kIndexZero);
    Tensor *natoms_tensor = ctx.Input(kIndexTwo);
    Tensor *rij_tensor = ctx.Output(kIndexZero);

    // attr
    std::vector<int64_t> sel_a = ctx.GetAttr("sel_a")->GetListInt();
    std::vector<int64_t> sec_a;
    cum_sum(sec_a, sel_a);
    int32_t nnei = sec_a.back();

    // analysis input
    auto natoms = static_cast<int32_t *>(natoms_tensor->GetData());
    int32_t nloc = natoms[0];
    int32_t nsamples = coord_tensor->GetTensorShape()->GetDimSize(0);

    // analysis output
    FPTYPE *p_rij = static_cast<FPTYPE *>(rij_tensor->GetData());
    int32_t batchOutputDataLen = nloc * nnei * coordinateXyzNum;
    for (int32_t ff = 0; ff < nsamples; ff++) {
      FPTYPE *rij = p_rij + ff * batchOutputDataLen;
      prod_env_mat_a_rij_cal(rij, ctx, ff, nnei);
    }
    return KERNEL_STATUS_OK;
  }

  uint32_t ProdEnvMatACalcRijCpuKernel::DoCompute(CpuKernelContext &ctx) {
    DataType params_type = ctx.Input(0)->GetDataType();
    if (params_type == DT_FLOAT) {
      return DoProdEnvMatACalcRijCompute<float>(ctx);
    } else {
      return DoProdEnvMatACalcRijCompute<double>(ctx);
    }
  }

  uint32_t ProdEnvMatACalcRijCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
    Tensor *coord_tensor = ctx.Input(kIndexZero);
    Tensor *type_tensor = ctx.Input(kIndexOne);
    Tensor *natoms_tensor = ctx.Input(kIndexTwo);
    Tensor *mesh_tensor = ctx.Input(kIndexFour);

    // check input dims
    auto coord_dims = coord_tensor->GetTensorShape()->GetDims();
    auto type_dims = type_tensor->GetTensorShape()->GetDims();
    auto natoms_dims = natoms_tensor->GetTensorShape()->GetDims();
    auto mesh_dims = mesh_tensor->GetTensorShape()->GetDims();

    KERNEL_CHECK_FALSE((coord_dims == kIndexTwo), KERNEL_STATUS_PARAM_INVALID,
                       "Dim of coord should be 2, but is [%d]", coord_dims);
    KERNEL_CHECK_FALSE((type_dims == kIndexTwo), KERNEL_STATUS_PARAM_INVALID,
                       "Dim of type should be 2, but is [%d]", type_dims);
    KERNEL_CHECK_FALSE((natoms_dims == 1), KERNEL_STATUS_PARAM_INVALID,
                       "Dim of natoms should be 1, but is [%d]", natoms_dims);
    KERNEL_CHECK_FALSE((mesh_dims == 1), KERNEL_STATUS_PARAM_INVALID,
                       "Dim of mesh should be 1, but is [%d]", mesh_dims);
    KERNEL_CHECK_FALSE((natoms_tensor->GetTensorShape()->GetDimSize(0) >= coordinateXyzNum),
                       KERNEL_STATUS_PARAM_INVALID,
                       "number of atoms should be larger than (or equal to ) 3");
    // check input type
    auto coord_type = static_cast<DataType>(coord_tensor->GetDataType());
    auto type_type = static_cast<DataType>(type_tensor->GetDataType());
    auto mesh_type = static_cast<DataType>(mesh_tensor->GetDataType());
    KERNEL_CHECK_FALSE((coord_type == DT_FLOAT || coord_type == DT_DOUBLE), KERNEL_STATUS_PARAM_INVALID,
                       "Type of coord shoule be DT_FLOAT or DT_DOUBLE, but is [%s]", DTypeStr(coord_type).c_str());
    KERNEL_CHECK_FALSE((type_type == DT_INT32), KERNEL_STATUS_PARAM_INVALID,
                       "Type of type shoule be DT_INT32, but is [%s]", DTypeStr(type_type).c_str());
    KERNEL_CHECK_FALSE((mesh_type == DT_INT32), KERNEL_STATUS_PARAM_INVALID,
                       "Type of mesh shoule be DT_INT32, but is [%s]", DTypeStr(mesh_type).c_str());

    auto natoms = static_cast<int32_t *>(natoms_tensor->GetData());
    int32_t nall = natoms[1];
    int32_t nsamples = coord_tensor->GetTensorShape()->GetDimSize(0);
    KERNEL_CHECK_FALSE((nsamples == type_tensor->GetTensorShape()->GetDimSize(0)), KERNEL_STATUS_PARAM_INVALID,
                       "number of samples should match");
    KERNEL_CHECK_FALSE((nall * coordinateXyzNum == coord_tensor->GetTensorShape()->GetDimSize(1)),
                       KERNEL_STATUS_PARAM_INVALID,
                       "number of coord should match");
    KERNEL_CHECK_FALSE((nall == type_tensor->GetTensorShape()->GetDimSize(1)), KERNEL_STATUS_PARAM_INVALID,
                       "number of type should match");

    return KERNEL_STATUS_OK;
  }

  uint32_t ProdEnvMatACalcRijCpuKernel::Compute(CpuKernelContext &ctx) {
    KERNEL_LOG_INFO("ProdEnvMatACalcRijCpuKernel::Compute start");
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Check ProdEnvMatACalcRij params failed.");
    auto ret = GetInputAndCheck(ctx);
    KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), ret, "GetInputAndCheck failed");
    uint32_t res = DoCompute(ctx);
    KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), res, "Compute failed");
    KERNEL_LOG_INFO("ProdEnvMatACalcRijCpuKernel::Compute end");
    return KERNEL_STATUS_OK;
  }

  template<typename FPTYPE>
  int32_t ProdEnvMatACalcRijCpuKernel::format_nlist_i_cpu(int32_t batchIndex,
      SingleNatomInfo<FPTYPE> &singleNatomInfo, CpuKernelContext &ctx,
      const int32_t &i_idx, const std::vector<int32_t> &nei_idx_a) {
    Tensor *coord_tensor = ctx.Input(kIndexZero);
    Tensor *type_tensor = ctx.Input(kIndexOne);
    int32_t nall =  coord_tensor->GetTensorShape()->GetDimSize(1) / coordinateXyzNum;
    auto rcut = ctx.GetAttr("rcut_r")->GetFloat();
    auto p_coord = static_cast<FPTYPE *>(coord_tensor->GetData());
    auto p_type = static_cast<int32_t *>(type_tensor->GetData());
    const FPTYPE *coord = p_coord + batchIndex * nall * coordinateXyzNum;
    const int32_t *type = p_type + batchIndex * nall;
    std::vector<int64_t> sel_a = ctx.GetAttr("sel_a")->GetListInt();
    std::vector<int64_t> sec_a;
    cum_sum(sec_a, sel_a);
    singleNatomInfo.fmt_nlist_a.resize(sec_a.back());
    fill(singleNatomInfo.fmt_nlist_a.begin(), singleNatomInfo.fmt_nlist_a.end(), -1);
    singleNatomInfo.d_distance_a.resize(sec_a.back());
    FPTYPE defaultDistance = 6.0;
    fill(singleNatomInfo.d_distance_a.begin(), singleNatomInfo.d_distance_a.end(), defaultDistance);
    // gether all neighbors
    std::vector<int32_t> nei_idx(nei_idx_a);
    // get the information for all neighbors
    std::vector<NeighborInfo> sel_nei;
    sel_nei.reserve(nei_idx_a.size());
    for (uint32_t kk = 0; kk < nei_idx.size(); kk++) {
      FPTYPE diff[coordinateXyzNum];
      const int32_t &j_idx = nei_idx[kk];
      for (uint32_t dd = 0; dd < coordinateXyzNum; dd++)
      {
        diff[dd] = coord[j_idx * coordinateXyzNum + dd] - coord[i_idx * coordinateXyzNum + dd];
      }
      FPTYPE rr = sqrt(dot3(diff, diff));
      if (rr < rcut) {
        sel_nei.push_back(NeighborInfo(type[j_idx], rr, j_idx));
      }
    }
    sort(sel_nei.begin(), sel_nei.end());
    std::vector<int64_t> nei_iter = sec_a;
    int32_t overflowed = -1;
    for (uint32_t kk = 0; kk < sel_nei.size(); kk++) {
      const int32_t &nei_type = sel_nei[kk].type;
      if (nei_iter[nei_type] < sec_a[nei_type + 1]) {
        singleNatomInfo.fmt_nlist_a[nei_iter[nei_type]] = sel_nei[kk].index;
        singleNatomInfo.d_distance_a[nei_iter[nei_type]] = sel_nei[kk].dist;
        nei_iter[nei_type]++;
      } else {
        overflowed = nei_type;
      }
    }
    return overflowed;
  }

  template<typename FPTYPE>
  void ProdEnvMatACalcRijCpuKernel::env_mat_a_cpu(std::vector<FPTYPE> &rij_a, const int32_t &i_idx,
      const std::vector<int32_t> &fmt_nlist_a, CpuKernelContext &ctx, int32_t batchIndex) {
      Tensor *coord_tensor = ctx.Input(kIndexZero);
      auto p_coord = static_cast<FPTYPE *>(coord_tensor->GetData());
      const FPTYPE *coord = p_coord + batchIndex * coord_tensor->GetTensorShape()->GetDimSize(1);
      std::vector<int64_t> sel_a = ctx.GetAttr("sel_a")->GetListInt();
      std::vector<int64_t> sec_a;
      cum_sum(sec_a, sel_a);
      // compute the diff of the neighbors
      rij_a.resize(sec_a.back() * coordinateXyzNum);
      fill(rij_a.begin(), rij_a.end(), 0.0);
      for (int32_t ii = 0; ii < int32_t(sec_a.size()) - 1; ii++) {
        for (int64_t jj = sec_a[ii]; jj < sec_a[ii + 1]; jj++) {
          if (fmt_nlist_a[jj] < 0) {
            break;
          }
          const int32_t &j_idx = fmt_nlist_a[jj];
          for (uint32_t dd = 0; dd < coordinateXyzNum; dd++) {
            rij_a[jj * coordinateXyzNum + dd] =
            coord[j_idx * coordinateXyzNum + dd] - coord[i_idx * coordinateXyzNum + dd];
          }
        }
      }
  }

  template <typename FPTYPE>
  void ProdEnvMatACalcRijCpuKernel::prod_env_mat_a_rij_cal(FPTYPE *rij, CpuKernelContext &ctx,
                                                           int32_t batchIndex, int32_t nnei) {
    Tensor *natoms_tensor = ctx.Input(kIndexTwo);
    Tensor *mesh_tensor = ctx.Input(kIndexFour);
    Tensor *nlist_tensor = ctx.Output(kIndexOne);
    Tensor *distance_tensor = ctx.Output(kIndexTwo);
    FPTYPE *p_distance = static_cast<FPTYPE *>(distance_tensor->GetData());
    int32_t *p_nlist = static_cast<int32_t *>(nlist_tensor->GetData());
    auto natoms = static_cast<int32_t *>(natoms_tensor->GetData());
    auto p_mesh = static_cast<int32_t *>(mesh_tensor->GetData());
    int32_t nloc = natoms[0];
    int32_t *nlist = p_nlist + batchIndex * nloc * nnei;
    FPTYPE *distance = p_distance + batchIndex * nloc * nnei;
    // out
    Tensor *rij_x_tensor = ctx.Output(kIndexThree);
    Tensor *rij_y_tensor = ctx.Output(kIndexFour);
    Tensor *rij_z_tensor = ctx.Output(kIndexFive);
    FPTYPE *p_rij_x = static_cast<FPTYPE *>(rij_x_tensor->GetData());
    FPTYPE *p_rij_y = static_cast<FPTYPE *>(rij_y_tensor->GetData());
    FPTYPE *p_rij_z = static_cast<FPTYPE *>(rij_z_tensor->GetData());
    FPTYPE *rij_x = p_rij_x + batchIndex * nloc * nnei;
    FPTYPE *rij_y = p_rij_y + batchIndex * nloc * nnei;
    FPTYPE *rij_z = p_rij_z + batchIndex * nloc * nnei;
    // get meshdata
    InputNlist meshdata;
    meshdata.nlocnum = p_mesh[0];
    meshdata.ilist = &p_mesh[1];
    meshdata.numneigh = &p_mesh[1 + meshdata.nlocnum];
    meshdata.firstneigh = (int32_t(*)[neighborMaxNum]) &p_mesh[1 + kIndexTwo * meshdata.nlocnum];
    int32_t max_nbor_size = max_numneigh(meshdata);
    // build nlist
    std::vector<std::vector<int32_t>> d_nlist_a(nloc);
    auto ComputeRijDistance = [&](int32_t start, int32_t end) {
      for (int32_t ii = start; ii < end; ii++) {
        int32_t coreNatomIndex = meshdata.ilist[ii];
        d_nlist_a[coreNatomIndex].reserve(max_nbor_size);
        for (int32_t jj = 0; jj < meshdata.numneigh[ii]; jj++) {
          int32_t j_idx = meshdata.firstneigh[ii][jj];
          d_nlist_a[coreNatomIndex].push_back(j_idx);
        }

        SingleNatomInfo<FPTYPE> singleNatomInfo;
        format_nlist_i_cpu(batchIndex, singleNatomInfo, ctx, coreNatomIndex, d_nlist_a[coreNatomIndex]);
        std::vector<FPTYPE> d_rij_a;
        env_mat_a_cpu(d_rij_a, coreNatomIndex, singleNatomInfo.fmt_nlist_a, ctx, batchIndex);
        // record outputs
        for (uint32_t jj = 0; jj < nnei * coordinateXyzNum; jj++) {
          rij[ii * nnei * coordinateXyzNum + jj] = d_rij_a[jj];
        }
        for (int32_t jj = 0; jj < nnei; jj++) {
          int32_t index = ii * nnei + jj;
          nlist[index] = singleNatomInfo.fmt_nlist_a[jj];
          distance[index] = singleNatomInfo.d_distance_a[jj];
          rij_x[index] = rij[index * coordinateXyzNum];
          rij_y[index] = rij[index * coordinateXyzNum + kIndexOne];
          rij_z[index] = rij[index * coordinateXyzNum + kIndexTwo];
        }
      }
    };
    CpuKernelUtils::ParallelFor(ctx, nloc, 1, ComputeRijDistance);
  }

  // functions used in custom ops
  void ProdEnvMatACalcRijCpuKernel::cum_sum(std::vector<int64_t> &sec, const std::vector<int64_t> &n_sel) {
    sec.resize(n_sel.size() + 1);
    sec[0] = 0;
    for (uint32_t ii = 1; ii < sec.size(); ii++) {
      sec[ii] = sec[ii - 1] + n_sel[ii - 1];
    }
  }

  int32_t ProdEnvMatACalcRijCpuKernel::max_numneigh(const InputNlist &nlist) {
    int32_t max_num = 0;
    for (int32_t ii = 0; ii < nlist.nlocnum; ii++) {
      if (nlist.numneigh[ii] > max_num) {
        max_num = nlist.numneigh[ii];
      }
    }
    return max_num;
  }

  template<typename FPTYPE>
  inline FPTYPE ProdEnvMatACalcRijCpuKernel::dot3(const FPTYPE *r0, const FPTYPE *r1) {
    return r0[0] * r1[0] + r0[1] * r1[1] + r0[2] * r1[2];
  }

  REGISTER_CPU_KERNEL(kProdEnvMatACalcRij, ProdEnvMatACalcRijCpuKernel);
} // namespace aicpu
