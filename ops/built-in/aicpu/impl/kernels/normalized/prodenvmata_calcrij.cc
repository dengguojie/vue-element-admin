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
  const uint32_t meshDataLength = 1026;
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
      prod_env_mat_a_rij_cal(rij, ctx, ff, nnei, sec_a);
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

  template <typename FPTYPE>
  void ProdEnvMatACalcRijCpuKernel::prod_env_mat_a_rij_cal(
      FPTYPE *rij, CpuKernelContext &ctx, int32_t batchIndex, int32_t nnei,
      std::vector<int64_t> &sec_a) {
    FPTYPE *p_distance = static_cast<FPTYPE *>(ctx.Output(kIndexTwo)->GetData());
    int32_t *p_nlist = static_cast<int32_t *>(ctx.Output(kIndexOne)->GetData());
    auto natoms = static_cast<int32_t *>(ctx.Input(kIndexTwo)->GetData());
    auto p_mesh = static_cast<int32_t *>(ctx.Input(kIndexFour)->GetData());
    int32_t nloc = natoms[0];
    int32_t *nlist = p_nlist + batchIndex * nloc * nnei;
    FPTYPE *distance = p_distance + batchIndex * nloc * nnei;
    // out
    FPTYPE *p_rij_x = static_cast<FPTYPE *>(ctx.Output(kIndexThree)->GetData());
    FPTYPE *p_rij_y = static_cast<FPTYPE *>(ctx.Output(kIndexFour)->GetData());
    FPTYPE *p_rij_z = static_cast<FPTYPE *>(ctx.Output(kIndexFive)->GetData());
    FPTYPE *rij_x = p_rij_x + batchIndex * nloc * nnei;
    FPTYPE *rij_y = p_rij_y + batchIndex * nloc * nnei;
    FPTYPE *rij_z = p_rij_z + batchIndex * nloc * nnei;

    Tensor *coord_tensor = ctx.Input(kIndexZero);
    Tensor *type_tensor = ctx.Input(kIndexOne);
    int32_t nall = coord_tensor->GetTensorShape()->GetDimSize(1) / coordinateXyzNum;
    float rcut = ctx.GetAttr("rcut_r")->GetFloat();
    float rcutsquared = rcut * rcut;
    auto p_coord = static_cast<FPTYPE *>(coord_tensor->GetData());
    auto p_type = static_cast<int32_t *>(type_tensor->GetData());
    const FPTYPE *coord = p_coord + batchIndex * nall * coordinateXyzNum;
    const int32_t *type = p_type + batchIndex * nall;
    // get meshdata
    InputNlist meshdata;
    meshdata.nlocnum = p_mesh[0];
    meshdata.ilist = &p_mesh[1];
    meshdata.numneigh = &p_mesh[1 + meshdata.nlocnum];
    meshdata.firstneigh = (int32_t(*)[neighborMaxNum]) &p_mesh[1 + kIndexTwo * meshdata.nlocnum];
    bool needmap = false;
    if (ctx.Input(kIndexFour)->GetTensorShape()->GetDimSize(0) > (1 + meshDataLength * meshdata.nlocnum)) {
      meshdata.nallmaptable = &p_mesh[1 + meshDataLength * meshdata.nlocnum];
      needmap = true;
    }
    int32_t atomTypes = sec_a.size() - 1;

    auto ComputeRijDistance = [&](int32_t start, int32_t end) {
      int32_t rijaxeslen = (end - start) * nnei * sizeof(FPTYPE);
      (void)memset_s(rij + start * nnei * coordinateXyzNum,
                     rijaxeslen * coordinateXyzNum, 0x00, rijaxeslen * coordinateXyzNum);
      std::fill(nlist + start * nnei, nlist + end * nnei, -1);
      std::fill(distance + start * nnei, distance + end * nnei, rcutsquared + 1.0);
      (void)memset_s(rij_x + start * nnei, rijaxeslen, 0x00, rijaxeslen);
      (void)memset_s(rij_y + start * nnei, rijaxeslen, 0x00, rijaxeslen);
      (void)memset_s(rij_z + start * nnei, rijaxeslen, 0x00, rijaxeslen);
      for (int32_t ii = start; ii < end; ii++) {
        int32_t coreNatomIndex = meshdata.ilist[ii];
        int32_t numNeighbor = meshdata.numneigh[ii];

        // output
        FPTYPE *curRij = rij + ii * nnei * coordinateXyzNum;
        int32_t *curNlist = nlist + ii * nnei;
        FPTYPE *curDist = distance + ii * nnei;
        FPTYPE *curRij_x = rij_x + ii * nnei;
        FPTYPE *curRij_y = rij_y + ii * nnei;
        FPTYPE *curRij_z = rij_z + ii * nnei;

        if (numNeighbor == -1) {
          continue;
        }
        int32_t *neighbors = static_cast<int32_t *>(meshdata.firstneigh[ii]);

        FPTYPE i_x = *(coord + coreNatomIndex * coordinateXyzNum);
        FPTYPE i_y = *(coord + coreNatomIndex * coordinateXyzNum + 1);
        FPTYPE i_z = *(coord + coreNatomIndex * coordinateXyzNum + 2);

        std::vector<std::vector<NeighborInfo>> selNeighbors(atomTypes, std::vector<NeighborInfo>());
        for (uint32_t s_j = 0; s_j < selNeighbors.size(); s_j++) {
          selNeighbors[s_j].reserve(numNeighbor);
        }

        for (int32_t j = 0; j < numNeighbor; j++) {
          int32_t j_idx = *(neighbors + j);

          FPTYPE j_x = *(coord + j_idx * coordinateXyzNum);
          FPTYPE j_y = *(coord + j_idx * coordinateXyzNum + 1);
          FPTYPE j_z = *(coord + j_idx * coordinateXyzNum + 2);

          FPTYPE dx = j_x - i_x;
          FPTYPE dy = j_y - i_y;
          FPTYPE dz = j_z - i_z;

          FPTYPE rr = dx * dx + dy * dy + dz * dz;
          if (rr < rcutsquared) {
            selNeighbors[*(type + j_idx)].push_back(NeighborInfo(rr, j_idx));
          }
        }

        for (uint32_t s_j = 0; s_j < selNeighbors.size(); s_j++) {
          std::sort(selNeighbors[s_j].begin(), selNeighbors[s_j].end());
        }

        for (uint32_t m_k = 0; m_k < selNeighbors.size(); m_k++) {
          uint32_t sortedIdx = 0;
          uint32_t len = selNeighbors[m_k].size();
          uint32_t cntSize = std::min(sec_a[m_k + 1], sec_a[m_k] + len);
          for (uint32_t res_idx = sec_a[m_k]; res_idx < cntSize; res_idx++) {
            int32_t curAtomIdx = selNeighbors[m_k][sortedIdx].index;
            *(curNlist + res_idx) = needmap ? meshdata.nallmaptable[curAtomIdx] : curAtomIdx;
            *(curDist + res_idx) = selNeighbors[m_k][sortedIdx].dist;
            FPTYPE dx = *(coord + curAtomIdx * coordinateXyzNum) - i_x;
            FPTYPE dy = *(coord + curAtomIdx * coordinateXyzNum + 1) - i_y;
            FPTYPE dz = *(coord + curAtomIdx * coordinateXyzNum + 2) - i_z;

            *(curRij + res_idx * coordinateXyzNum) = dx;
            *(curRij + res_idx * coordinateXyzNum + 1) = dy;
            *(curRij + res_idx * coordinateXyzNum + 2) = dz;
            *(curRij_x + res_idx) = dx;
            *(curRij_y + res_idx) = dy;
            *(curRij_z + res_idx) = dz;
            sortedIdx++;
          }
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

  REGISTER_CPU_KERNEL(kProdEnvMatACalcRij, ProdEnvMatACalcRijCpuKernel);
} // namespace aicpu
