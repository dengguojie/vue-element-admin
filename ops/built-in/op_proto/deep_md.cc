/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

/*!
 * \file deep_md.cc
 * \brief
 */
#include "inc/deep_md.h"

#include <cmath>
#include <string>
#include <vector>

#include "util/util.h"
#include "util/error_util.h"
#include "op_log.h"
#include "graph/utils/op_desc_utils.h"
#include "register/infer_data_slice_registry.h"
#include "graph/common_error_codes.h"
#include "graph/debug/ge_attr_define.h"
#include "axis_util.h"
#include "common_shape_fns.h"
#include "util/vector_proto_profiling.h"

namespace ge {
// --------------------------ProdForceSeA Begin---------------------
IMPLEMT_VERIFIER(ProdForceSeA, ProdForceSeAVerify) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("Failed to get op name of ProdForceSeA"), return GRAPH_FAILED);

  auto opDesc = OpDescUtils::GetOpDescFromOperator(op);

  GeTensorDescPtr netDerivDesc = opDesc->MutableInputDesc("net_deriv");
  std::vector<int64_t> netDerivShape = netDerivDesc->MutableShape().GetDims();
  CHECK(netDerivShape.size() != 2, OP_LOGE(opName.GetString(), "Dim of net_deriv should be 2"), return GRAPH_FAILED);
  int64_t nframes = netDerivShape[0];

  GeTensorDescPtr inDerivDesc = opDesc->MutableInputDesc("in_deriv");
  std::vector<int64_t> inDerivShape = inDerivDesc->MutableShape().GetDims();
  CHECK(inDerivShape.size() != 2, OP_LOGE(opName.GetString(), "Dim of in_deriv should be 2"), return GRAPH_FAILED);
  CHECK(inDerivShape[0] != nframes,
        OP_LOGE(opName.GetString(), "shape[0] of in_deriv should match with shape[0] of net_deriv"),
        return GRAPH_FAILED);

  GeTensorDescPtr nlistDesc = opDesc->MutableInputDesc("nlist");
  std::vector<int64_t> nlistShape = nlistDesc->MutableShape().GetDims();
  CHECK(nlistShape.size() != 2, OP_LOGE(opName.GetString(), "Dim of nlist should be 2"), return GRAPH_FAILED);
  CHECK(nlistShape[0] != nframes,
        OP_LOGE(opName.GetString(), "Number of nlist samples should match with net_deriv"), return GRAPH_FAILED);

  GeTensorDescPtr natomsDesc = opDesc->MutableInputDesc("natoms");
  std::vector<int64_t> natomsShape = natomsDesc->MutableShape().GetDims();
  CHECK(natomsShape.size() != 1, OP_LOGE(opName.GetString(), "Dim of natoms should be 1"), return GRAPH_FAILED);
  int64_t natomsSize = natomsShape[0];
  CHECK(natomsSize < 3,
        OP_LOGE(opName.GetString(), "Number of atoms should be larger than (or equal to) 3"), return GRAPH_FAILED);

  CHECK(netDerivDesc->GetDataType() != inDerivDesc->GetDataType(),
        OP_LOGE(opName.GetString(), "Data type of net_deriv and in_deriv are not match"), return GRAPH_FAILED);

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ProdForceSeAInferShape) {
  auto opDesc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr netDerivDesc = opDesc->MutableInputDesc("net_deriv");
  std::vector<int64_t> netDerivShape = netDerivDesc->MutableShape().GetDims();
  std::vector<std::pair<int64_t, int64_t>> netDerivShapeRange;
  opDesc->MutableInputDesc("net_deriv")->GetShapeRange(netDerivShapeRange);
  MakeUpShapeRange(netDerivShape, netDerivShapeRange);
  int64_t nframes = netDerivShape[0];

  GeTensorDescPtr natomsDesc = opDesc->MutableInputDesc("natoms");
  std::vector<int64_t> natomsShape = natomsDesc->MutableShape().GetDims();
  std::vector<std::pair<int64_t, int64_t>> natomsShapeRange;
  opDesc->MutableInputDesc("natoms")->GetShapeRange(natomsShapeRange);
  MakeUpShapeRange(natomsShape, natomsShapeRange);

  int64_t nall = 28328;
  int64_t dim = nall * 3;
  std::vector<int64_t> atomForceShape = {nframes, dim};
  std::vector<std::pair<int64_t, int64_t>> atomForceShapeRange = {netDerivShapeRange[0],
                                                                  std::pair<int64_t, int64_t>(dim, dim)};

  GeTensorDescPtr atomForceDesc = opDesc->MutableOutputDesc("atom_force");
  atomForceDesc->SetShape(ge::GeShape(atomForceShape));
  atomForceDesc->SetShapeRange(atomForceShapeRange);
  atomForceDesc->SetDataType(netDerivDesc->GetDataType());

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ProdForceSeA, ProdForceSeAInferShape);
VERIFY_FUNC_REG(ProdForceSeA, ProdForceSeAVerify);
// --------------------------ProdForceSeA END---------------------

// --------------------------ProdVirialSeA Begin---------------------
IMPLEMT_VERIFIER(ProdVirialSeA, ProdVirialSeAVerify) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("Failed to get op name of ProdVirialSeA"), return GRAPH_FAILED);

  auto opDesc = OpDescUtils::GetOpDescFromOperator(op);

  GeTensorDescPtr netDerivDesc = opDesc->MutableInputDesc("net_deriv");
  std::vector<int64_t> netDerivShape = netDerivDesc->MutableShape().GetDims();
  CHECK(netDerivShape.size() != 2, OP_LOGE(opName.GetString(), "Dim of net_deriv should be 2"), return GRAPH_FAILED);
  int64_t nframes = netDerivShape[0];

  GeTensorDescPtr inDerivDesc = opDesc->MutableInputDesc("in_deriv");
  std::vector<int64_t> inDerivShape = inDerivDesc->MutableShape().GetDims();
  CHECK(inDerivShape.size() != 2, OP_LOGE(opName.GetString(), "Dim of in_deriv should be 2"), return GRAPH_FAILED);
  CHECK(inDerivShape[0] != nframes,
        OP_LOGE(opName.GetString(), "Number of in_deriv samples should match with net_deriv"), return GRAPH_FAILED);

  GeTensorDescPtr rijDesc = opDesc->MutableInputDesc("rij");
  std::vector<int64_t> rijShape = rijDesc->MutableShape().GetDims();
  CHECK(rijShape.size() != 2, OP_LOGE(opName.GetString(), "Dim of rij should be 2"), return GRAPH_FAILED);
  CHECK(rijShape[0] != nframes,
        OP_LOGE(opName.GetString(), "Number of rij samples should match with net_deriv"), return GRAPH_FAILED);

  GeTensorDescPtr nlistDesc = opDesc->MutableInputDesc("nlist");
  std::vector<int64_t> nlistShape = nlistDesc->MutableShape().GetDims();
  CHECK(nlistShape.size() != 2, OP_LOGE(opName.GetString(), "Dim of nlist should be 2"), return GRAPH_FAILED);
  CHECK(nlistShape[0] != nframes,
        OP_LOGE(opName.GetString(), "Number of nlist samples should match with net_deriv"), return GRAPH_FAILED);

  GeTensorDescPtr natomsDesc = opDesc->MutableInputDesc("natoms");
  std::vector<int64_t> natomsShape = natomsDesc->MutableShape().GetDims();
  CHECK(natomsShape.size() != 1, OP_LOGE(opName.GetString(), "Dim of natoms should be 1"), return GRAPH_FAILED);
  CHECK(natomsShape[0] < 3,
        OP_LOGE(opName.GetString(), "Number of atoms should be larger than (or equal to) 3"), return GRAPH_FAILED);

  CHECK(netDerivDesc->GetDataType() != inDerivDesc->GetDataType(),
        OP_LOGE(opName.GetString(), "Data type of net_deriv and in_deriv are not match"), return GRAPH_FAILED);
  CHECK(netDerivDesc->GetDataType() != rijDesc->GetDataType(),
        OP_LOGE(opName.GetString(), "Data type of net_deriv and rij are not match"), return GRAPH_FAILED);

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ProdVirialSeAInferShape) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("Failed to get op name of ProdVirialSeA"), return GRAPH_FAILED);

  const vector<string> depend_names = {"natoms"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  auto opDesc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr netDerivDesc = opDesc->MutableInputDesc("net_deriv");
  std::vector<int64_t> netDerivShape = netDerivDesc->MutableShape().GetDims();
  CHECK(netDerivShape.size() != 2, OP_LOGE(opName.GetString(), "Dim of net_deriv should be 2"), return GRAPH_FAILED);
  int64_t nframes = netDerivShape[0];

  GeTensorDescPtr virialDesc = opDesc->MutableOutputDesc("virial");
  CHECK(virialDesc == nullptr, OP_LOGE(opName.GetString(), "Failed to get virial desc"), return GRAPH_FAILED);
  virialDesc->SetShape(ge::GeShape({nframes, 9}));
  virialDesc->SetOriginShape(ge::GeShape({nframes, 9}));
  virialDesc->SetDataType(netDerivDesc->GetDataType());

  GeTensorDescPtr atomVirialDesc = opDesc->MutableOutputDesc("atom_virial");
  CHECK(atomVirialDesc == nullptr, OP_LOGE(opName.GetString(), "Failed to get atom_virial desc"), return GRAPH_FAILED);

  std::vector<int64_t> atomVirialShape = {nframes};
  Tensor natoms;
  if (op.GetInputConstData("natoms", natoms) != GRAPH_SUCCESS) {
    OP_LOGD(opName.GetString(), "Failed to get natoms tensor from const data");
    atomVirialShape.push_back(-1);

    std::vector<std::pair<int64_t, int64_t>> atomVirialShapeRange;
    MakeUpShapeRange(atomVirialShape, atomVirialShapeRange);
    atomVirialDesc->SetShapeRange(atomVirialShapeRange);
  } else {
    DataType natomsDType = opDesc->MutableInputDesc("natoms")->GetDataType();
    std::vector<int64_t> constVec;
    CHECK(!GetConstValue(op, natoms, natomsDType, constVec), OP_LOGE(opName.GetString(), "Failed to get natoms value"),
          return GRAPH_FAILED);
    CHECK(constVec.size() < 3, OP_LOGE(opName.GetString(), "Failed to check natoms value"), return GRAPH_FAILED);

    int64_t nall = constVec[1];
    CHECK(nall <= 0, OP_LOGE(opName.GetString(), "Failed to get nall"), return GRAPH_FAILED);
    atomVirialShape.push_back(nall * 9);
  }

  atomVirialDesc->SetShape(ge::GeShape(atomVirialShape));
  atomVirialDesc->SetOriginShape(ge::GeShape(atomVirialShape));
  atomVirialDesc->SetDataType(netDerivDesc->GetDataType());

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ProdVirialSeA, ProdVirialSeAInferShape);
VERIFY_FUNC_REG(ProdVirialSeA, ProdVirialSeAVerify);
// --------------------------ProdVirialSeA END---------------------

}  // namespace ge
