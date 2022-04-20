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
#include <algorithm>
#include <numeric>

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
static const int NUM_8 = 8;
static const int NUM_15 = 15;

// --------------------------TabulateFusion Begin-----------------
IMPLEMT_VERIFIER(TabulateFusion, TabulateFusionVerify) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS,
        OP_LOGE("TabulateFusion", "Failed to get op name of TabulateFusion"),
        return GRAPH_FAILED);

  auto opDesc = OpDescUtils::GetOpDescFromOperator(op);

  GeTensorDescPtr tableDesc = opDesc->MutableInputDesc("table");
  std::vector<int64_t> tableShape = tableDesc->MutableShape().GetDims();
  CHECK(tableShape.size() != 2, OP_LOGE(opName.GetString(), "Dim of table Shape should be 2"), return GRAPH_FAILED);

  GeTensorDescPtr tableInfoDesc = opDesc->MutableInputDesc("table_info");
  std::vector<int64_t> tableInfoShape = tableInfoDesc->MutableShape().GetDims();
  CHECK(tableInfoShape.size() != 1, OP_LOGE(opName.GetString(), "Dim of table_info Shape should be 1"),
        return GRAPH_FAILED);
  int64_t tableInfoSize = tableInfoShape[0];
  CHECK(tableInfoSize < 5, OP_LOGE(opName.GetString(), "size of table_info should be >= 5."),
        return GRAPH_FAILED);

  GeTensorDescPtr emXDesc = opDesc->MutableInputDesc("em_x");
  std::vector<int64_t> emXShape = emXDesc->MutableShape().GetDims();
  CHECK(emXShape.size() != 2, OP_LOGE(opName.GetString(), "Dim of  em_x should be 2"), return GRAPH_FAILED);

  GeTensorDescPtr emDesc = opDesc->MutableInputDesc("em");
  std::vector<int64_t> emShape = emDesc->MutableShape().GetDims();
  CHECK(emShape.size() != 3, OP_LOGE(opName.GetString(), "Dim of em should be 3"), return GRAPH_FAILED);
  int64_t lastLayerSize = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("last_layer_size", lastLayerSize)) {
    std::string errMsg = GetInputInvalidErrMsg("last_layer_size");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), errMsg);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(TabulateFusionInferShape) {
  OP_LOGI(op.GetName(), "TabulateFusionInferShape begin");
  auto opDesc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr emDesc = opDesc->MutableInputDesc("em");
  std::vector<int64_t> emShapeVec = emDesc->MutableShape().GetDims();
  int64_t nlocValue = emShapeVec[0];
  int64_t lastLayerSize = 0;
  op.GetAttr("last_layer_size", lastLayerSize);
  std::vector<int64_t> outputShape = {nlocValue, 4, lastLayerSize};
  GeTensorDescPtr outputDesc = opDesc->MutableOutputDesc("descriptor");

  if (IsUnknownVec(emShapeVec)) {
    std::vector<std::pair<int64_t, int64_t>> inputEmShapeRange;
    emDesc->GetShapeRange(inputEmShapeRange);
    MakeUpShapeRange(emShapeVec, inputEmShapeRange);
    std::vector<std::pair<int64_t, int64_t>> outputShapeRange = {inputEmShapeRange[0],
      std::pair<int64_t, int64_t>(4, 4), std::pair<int64_t, int64_t>(lastLayerSize, lastLayerSize)};
    outputDesc->SetShape(ge::GeShape(outputShape));
    outputDesc->SetOriginShape(ge::GeShape(outputShape));
    outputDesc->SetShapeRange(outputShapeRange);
    outputDesc->SetDataType(emDesc->GetDataType());
    return GRAPH_SUCCESS;
  }

  // below is second infershape or static shape scene.
  int32_t splitCount = 1;
  int32_t splitIndex = 0;
  op.GetAttr("split_count", splitCount);
  op.GetAttr("split_index", splitIndex);
  OP_LOGI(op.GetName(), "TabulateFusionInferShape, splitCount=%d, splitIndex=%d", splitCount, splitIndex);
  if (splitCount == 1) {
    outputDesc->SetShape(ge::GeShape(outputShape));
  } else if (splitCount == 2) {
    int64_t baseValue = nlocValue / NUM_15;
    int64_t ceilValue = baseValue * NUM_8;
    if (nlocValue % NUM_15 != 0) {
      ceilValue += (nlocValue % NUM_15);
    }

    if (splitIndex == 0) {
      OP_LOGI(op.GetName(), "TabulateFusionInferShape, splitIndex is 0, dim0=%ld", ceilValue);
      outputShape = {ceilValue, 4, lastLayerSize};
    } else {
      OP_LOGI(op.GetName(), "TabulateFusionInferShape, splitIndex is 1, dim0=%ld", nlocValue - ceilValue);
      outputShape = {nlocValue - ceilValue, 4, lastLayerSize};
    }
    outputDesc->SetShape(ge::GeShape(outputShape));
  } else {
    std::string errMsg = GetInputInvalidErrMsg("not support splitCount > 2");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), errMsg);
    return GRAPH_FAILED;
  }
  outputDesc->SetDataType(emDesc->GetDataType());

  OP_LOGI(op.GetName(), "TabulateFusionInferShape run success");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(TabulateFusion, TabulateFusionInferShape);
VERIFY_FUNC_REG(TabulateFusion, TabulateFusionVerify);
// --------------------------TabulateFusion END---------------------

// --------------------------ProdForceSeA Begin---------------------
IMPLEMT_VERIFIER(ProdForceSeA, ProdForceSeAVerify) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS,
        OP_LOGE("ProdForceSeA", "Failed to get op name of ProdForceSeA"),
        return GRAPH_FAILED);

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
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("ProdForceSeA", "Failed to get op name of ProdForceSeA."),
        return GRAPH_FAILED);
  const vector<string> depend_names = {"natoms"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  auto opDesc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr netDerivDesc = opDesc->MutableInputDesc("net_deriv");
  std::vector<int64_t> netDerivShape = netDerivDesc->MutableShape().GetDims();
  int64_t nframes = netDerivShape[0];
  GeTensorDescPtr atomForceDesc = opDesc->MutableOutputDesc(0);
  if (atomForceDesc == nullptr) {
    OP_LOGE(opName.GetString(), "Failed to get force node.");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> atomForceShape = {nframes};
  Tensor natoms;
  std::vector<std::pair<int64_t, int64_t>> atomForceShapeRange;
  if (op.GetInputConstData("natoms", natoms) != GRAPH_SUCCESS) {
    OP_LOGI(opName.GetString(), "Failed to get natoms node.");
    atomForceShape.push_back(-1);
    atomForceShape.push_back(3);
    MakeUpShapeRange(atomForceShape, atomForceShapeRange);
    atomForceDesc->SetShapeRange(atomForceShapeRange);
  } else {
    OP_LOGI(opName.GetString(), "Success to get natoms node.");
    DataType natomsDType = opDesc->MutableInputDesc("natoms")->GetDataType();
    std::vector<int64_t> constVec;
    if (!GetConstValue(op, natoms, natomsDType, constVec)) {
      OP_LOGE(opName.GetString(), "Get Const Value failed.");
      return GRAPH_FAILED;
    };
    int64_t nall = constVec[1];
    bool secondInfer = false;
    op.GetAttr("second_infer", secondInfer);
    if (secondInfer) {
      atomForceShape.push_back(3);
      atomForceShape.push_back(nall);
      MakeUpShapeRange(atomForceShape, atomForceShapeRange);
      atomForceDesc->SetShapeRange(atomForceShapeRange);
      OP_LOGD(opName.GetString(), "atomForceShapeRange2 %d %d %d %d %d %d.",
              atomForceShapeRange[0].first, atomForceShapeRange[0].second,
              atomForceShapeRange[1].first, atomForceShapeRange[1].second,
              atomForceShapeRange[2].first, atomForceShapeRange[2].second);
    } else {
      atomForceShape.push_back(nall);
      atomForceShape.push_back(3);
      MakeUpShapeRange(atomForceShape, atomForceShapeRange);
      atomForceDesc->SetShapeRange(atomForceShapeRange);
      OP_LOGD(opName.GetString(), "atomForceShapeRange %d %d %d %d %d %d.",
              atomForceShapeRange[0].first, atomForceShapeRange[0].second,
              atomForceShapeRange[1].first, atomForceShapeRange[1].second,
              atomForceShapeRange[2].first, atomForceShapeRange[2].second);
    }
  }
  atomForceDesc->SetShape(ge::GeShape(atomForceShape));
  atomForceDesc->SetOriginShape(ge::GeShape(atomForceShape));
  atomForceDesc->SetDataType(netDerivDesc->GetDataType());

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ProdForceSeA, ProdForceSeAInferShape);
VERIFY_FUNC_REG(ProdForceSeA, ProdForceSeAVerify);
// --------------------------ProdForceSeA END---------------------

// --------------------------ProdVirialSeA Begin---------------------
IMPLEMT_VERIFIER(ProdVirialSeA, ProdVirialSeAVerify) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("ProdVirialSeA", "Failed to get op name of ProdVirialSeA"),
        return GRAPH_FAILED);

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
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("ProdVirialSeA", "Failed to get op name of ProdVirialSeA"),
        return GRAPH_FAILED);

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

// --------------------------ProdEnvMatA Begin---------------------
IMPLEMT_VERIFIER(ProdEnvMatA, ProdEnvMatAVerify) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("ProdEnvMatA", "Failed to get op name of ProdVirialSeA"),
        return GRAPH_FAILED);

  auto opDesc = OpDescUtils::GetOpDescFromOperator(op);

  GeTensorDescPtr coordDesc = opDesc->MutableInputDesc(0);
  std::vector<int64_t> coordShape = coordDesc->MutableShape().GetDims();
  CHECK(coordShape.size() != 2, OP_LOGE(opName.GetString(), "Dim of coordShape should be 2"), return GRAPH_FAILED);
  int64_t nframes = coordShape[0];

  GeTensorDescPtr typeDesc = opDesc->MutableInputDesc(1);
  std::vector<int64_t> typeShape = typeDesc->MutableShape().GetDims();
  CHECK(typeShape.size() != 2, OP_LOGE(opName.GetString(), "Dim of typeShape should be 2"), return GRAPH_FAILED);
  CHECK(typeShape[0] != nframes,
        OP_LOGE(opName.GetString(), "Number of typeShape samples should match with coords"), return GRAPH_FAILED);

  GeTensorDescPtr dstdDesc = opDesc->MutableInputDesc("dstd");
  std::vector<int64_t> dstdShape = dstdDesc->MutableShape().GetDims();
  CHECK(dstdShape.size() != 2, OP_LOGE(opName.GetString(), "Dim of dstdShape should be 2"), return GRAPH_FAILED);

  GeTensorDescPtr davgDesc = opDesc->MutableInputDesc("davg");
  std::vector<int64_t> davgShape = davgDesc->MutableShape().GetDims();
  CHECK(davgShape.size() != 2, OP_LOGE(opName.GetString(), "Dim of davgShape should be 2"), return GRAPH_FAILED);

  GeTensorDescPtr natomsDesc = opDesc->MutableInputDesc("natoms");
  std::vector<int64_t> natomsShape = natomsDesc->MutableShape().GetDims();
  CHECK(natomsShape.size() != 1, OP_LOGE(opName.GetString(), "Dim of natoms should be 1"), return GRAPH_FAILED);
  CHECK(natomsShape[0] < 3,
        OP_LOGE(opName.GetString(), "Number of atoms should be larger than (or equal to) 3"), return GRAPH_FAILED);

  CHECK(coordDesc->GetDataType() != dstdDesc->GetDataType(),
        OP_LOGE(opName.GetString(), "Data type of coords and std are not match"), return GRAPH_FAILED);
  CHECK(davgDesc->GetDataType() != dstdDesc->GetDataType(),
        OP_LOGE(opName.GetString(), "Data type of avg and std are not match"), return GRAPH_FAILED);

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ProdEnvMatAInferShape) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("ProdEnvMatA", "Failed to get op name of ProdEnvMatA"),
        return GRAPH_FAILED);

  const vector<string> depend_names = {"natoms"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  auto opDesc = OpDescUtils::GetOpDescFromOperator(op);
  auto coordDesc = opDesc->MutableInputDesc(0);
  CHECK(coordDesc == nullptr, OP_LOGE(opName.GetString(), "Failed to get coord desc"), return GRAPH_FAILED);
  auto avgDesc = opDesc->MutableInputDesc(5);
  CHECK(avgDesc == nullptr, OP_LOGE(opName.GetString(), "Failed to get avg desc"), return GRAPH_FAILED);

  std::vector<int64_t> dimsCoords = coordDesc->MutableShape().GetDims();
  std::vector<int64_t> dimsAvg = avgDesc->MutableShape().GetDims();
  CHECK(dimsCoords.size() != 2, OP_LOGE(opName.GetString(), "Dim of coords should be 2"), return GRAPH_FAILED);
  CHECK(dimsAvg.size() != 2, OP_LOGE(opName.GetString(), "Dim of avg should be 2"), return GRAPH_FAILED);

  DataType type = coordDesc->GetDataType();

  vector<int64_t> sel_a;

  int64_t nnei = 0;
  if (op.GetAttr("sel_a", sel_a) == GRAPH_SUCCESS) {
    nnei = std::accumulate(sel_a.begin(), sel_a.end(), 0);
  }
  int64_t splitCount = 0;
  int64_t splitIndex = 0;

  op.GetAttr("split_count", splitCount);

  if (op.GetAttr("split_index", splitIndex) != GRAPH_SUCCESS) {
    OP_LOGD(opName.GetString(), "Get split_index name failed.");
  }

  int64_t nsample = dimsCoords[0];
  int64_t nalls = dimsCoords[1];
  int64_t nloc = UNKNOWN_DIM;
  int64_t descrptDimOne = UNKNOWN_DIM;
  int64_t descrptDerivDimOne = UNKNOWN_DIM;
  int64_t rijDimOne = UNKNOWN_DIM;
  int64_t nlistDimOne = UNKNOWN_DIM;
  int64_t totalCores = 15;
  int64_t vectorCores = 7;
  if (nalls != UNKNOWN_DIM) {
    Tensor natoms;
    if (op.GetInputConstData("natoms", natoms) == GRAPH_SUCCESS) {
      DataType natomsDType = opDesc->MutableInputDesc("natoms")->GetDataType();
      std::vector<int64_t> constVec;
      GetConstValue(op, natoms, natomsDType, constVec);
      CHECK(constVec.size() < 3, OP_LOGE(opName.GetString(), "Failed to get natoms value"),
            return GRAPH_FAILED);
      nloc = constVec[0];
    }
    if (splitCount > 1) {
      if (splitIndex == 0) {
        nloc = nloc - ((nloc / totalCores) * vectorCores);
      } else {
        nloc = (nloc / totalCores) * vectorCores;
      }
    }
    descrptDimOne = nloc * nnei * 4;
    descrptDerivDimOne = nloc * nnei * 12;
    rijDimOne = nloc * nnei * 3;
    nlistDimOne = nloc * nnei;
  }

  std::vector<int64_t> dimsDescrptOutput = {nsample, descrptDimOne};
  std::vector<int64_t> dimsDescrptDerivOutput = {nsample, descrptDerivDimOne};
  std::vector<int64_t> dimsRijOutput = {nsample, rijDimOne};
  std::vector<int64_t> dimsNlistOutput = {nsample, nlistDimOne};

  auto descrpt_desc = opDesc->MutableOutputDesc(0);
  CHECK(descrpt_desc == nullptr, OP_LOGE(opName.GetString(), "Failed to get descrpt_desc desc"),
        return GRAPH_FAILED);

  auto descrpt_deriv_desc = opDesc->MutableOutputDesc(1);
  CHECK(descrpt_deriv_desc == nullptr, OP_LOGE(opName.GetString(), "Failed to get descrpt_deriv_desc desc"),
        return GRAPH_FAILED);

  auto rij_desc = opDesc->MutableOutputDesc(2);
  CHECK(rij_desc == nullptr, OP_LOGE(opName.GetString(), "Failed to get rij_desc desc"),
        return GRAPH_FAILED);

  auto nlist_desc = opDesc->MutableOutputDesc(3);
  CHECK(nlist_desc == nullptr, OP_LOGE(opName.GetString(), "Failed to get nlist_desc desc"),
        return GRAPH_FAILED);

  descrpt_desc->SetShape(GeShape(dimsDescrptOutput));
  descrpt_desc->SetOriginShape(GeShape(dimsDescrptOutput));
  descrpt_desc->SetDataType(type);

  descrpt_deriv_desc->SetShape(GeShape(dimsDescrptDerivOutput));
  descrpt_deriv_desc->SetOriginShape(GeShape(dimsDescrptDerivOutput));
  descrpt_deriv_desc->SetDataType(type);

  rij_desc->SetShape(GeShape(dimsRijOutput));
  rij_desc->SetOriginShape(GeShape(dimsRijOutput));
  rij_desc->SetDataType(type);

  nlist_desc->SetShape(GeShape(dimsNlistOutput));
  nlist_desc->SetOriginShape(GeShape(dimsNlistOutput));
  nlist_desc->SetDataType(DT_INT32);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ProdEnvMatA, ProdEnvMatAInferShape);
VERIFY_FUNC_REG(ProdEnvMatA, ProdEnvMatAVerify);
// --------------------------ProdEnvMatA END---------------------

// --------------------------ProdEnvMatACalcRij Begin---------------------
IMPLEMT_VERIFIER(ProdEnvMatACalcRij, ProdEnvMatACalcRijVerify) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("ProdEnvMatACalcRij", "Failed to get op name of ProdEnvMatACalcRij"),
        return GRAPH_FAILED);

  auto opDesc = OpDescUtils::GetOpDescFromOperator(op);

  GeTensorDescPtr coordDesc = opDesc->MutableInputDesc(0);
  std::vector<int64_t> coordShape = coordDesc->MutableShape().GetDims();
  CHECK(coordShape.size() != 2, OP_LOGE(opName.GetString(), "Dim of coordShape should be 2"), return GRAPH_FAILED);
  int64_t nframes = coordShape[0];

  GeTensorDescPtr typeDesc = opDesc->MutableInputDesc(1);
  std::vector<int64_t> typeShape = typeDesc->MutableShape().GetDims();
  CHECK(typeShape.size() != 2, OP_LOGE(opName.GetString(), "Dim of typeShape should be 2"), return GRAPH_FAILED);
  CHECK(typeShape[0] != nframes,
        OP_LOGE(opName.GetString(), "Number of typeShape samples should match with coords"), return GRAPH_FAILED);

  GeTensorDescPtr natomsDesc = opDesc->MutableInputDesc("natoms");
  std::vector<int64_t> natomsShape = natomsDesc->MutableShape().GetDims();
  CHECK(natomsShape.size() != 1, OP_LOGE(opName.GetString(), "Dim of natoms should be 1"), return GRAPH_FAILED);
  CHECK(natomsShape[0] < 3,
        OP_LOGE(opName.GetString(), "Number of atoms should be larger than (or equal to) 3"), return GRAPH_FAILED);

  return GRAPH_SUCCESS;
}

void SetProdEnvMatACalcRijOutputUnknowShape(OpDescPtr op_desc, DataType coord_type) {
  auto output_rij = op_desc->MutableOutputDesc("rij");
  auto output_nlist = op_desc->MutableOutputDesc("nlist");
  auto output_distance = op_desc->MutableOutputDesc("distance");
  auto output_rij_x = op_desc->MutableOutputDesc("rij_x");
  auto output_rij_y = op_desc->MutableOutputDesc("rij_y");
  auto output_rij_z = op_desc->MutableOutputDesc("rij_z");
  
  output_rij->SetDataType(coord_type);
  output_rij->SetShape(ge::GeShape(UNKNOWN_SHAPE));
  output_rij->SetOriginShape(ge::GeShape(UNKNOWN_SHAPE));
  output_nlist->SetDataType(DT_INT32);
  output_nlist->SetShape(ge::GeShape(UNKNOWN_SHAPE));
  output_nlist->SetOriginShape(ge::GeShape(UNKNOWN_SHAPE));
  output_distance->SetDataType(coord_type);
  output_distance->SetShape(ge::GeShape(UNKNOWN_SHAPE));
  output_distance->SetOriginShape(ge::GeShape(UNKNOWN_SHAPE));
  output_rij_x->SetDataType(coord_type);
  output_rij_x->SetShape(ge::GeShape(UNKNOWN_SHAPE));
  output_rij_x->SetOriginShape(ge::GeShape(UNKNOWN_SHAPE));
  output_rij_y->SetDataType(coord_type);
  output_rij_y->SetShape(ge::GeShape(UNKNOWN_SHAPE));
  output_rij_y->SetOriginShape(ge::GeShape(UNKNOWN_SHAPE));
  output_rij_z->SetDataType(coord_type);
  output_rij_z->SetShape(ge::GeShape(UNKNOWN_SHAPE));
  output_rij_z->SetOriginShape(ge::GeShape(UNKNOWN_SHAPE));
  
  return;
}

IMPLEMT_COMMON_INFERFUNC(ProdEnvMatACalcRijInferShape) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto input_coord = op_desc->MutableInputDesc("coord");
  DataType coord_type = op.GetInputDescByName("coord").GetDataType();

  std::vector<int64_t> input_coord_dims = input_coord->GetShape().GetDims();
  const bool coord_known_shape = IsUnknownVec(input_coord_dims);
  if (coord_known_shape) {
    SetProdEnvMatACalcRijOutputUnknowShape(op_desc, coord_type);
    return GRAPH_SUCCESS;
  }

  Tensor natoms;
  if (op.GetInputConstData("natoms", natoms) != GRAPH_SUCCESS) {
    SetProdEnvMatACalcRijOutputUnknowShape(op_desc, coord_type);
    return GRAPH_SUCCESS;
  }

  DataType natomsDType = op_desc->MutableInputDesc("natoms")->GetDataType();
  std::vector<int64_t> constVec;
  GetConstValue(op, natoms, natomsDType, constVec);
  AscendString opName;
  CHECK(constVec.size() < 3, OP_LOGE(opName.GetString(), "Failed to get natoms value"),
        return GRAPH_FAILED);
  int64_t nloc = constVec[0];

  /* Caculate */
  auto nsample = input_coord_dims[0];
  int32_t nnei = 0;
  vector<int32_t> sel_a_arr;
  (void)op.GetAttr("sel_a", sel_a_arr);
  for (size_t i = 0; i < sel_a_arr.size(); i++) {
    nnei += sel_a_arr[i];
  }

  auto output_rij = op_desc->MutableOutputDesc("rij");
  auto output_nlist = op_desc->MutableOutputDesc("nlist");
  auto output_distance = op_desc->MutableOutputDesc("distance");
  auto output_rij_x = op_desc->MutableOutputDesc("rij_x");
  auto output_rij_y = op_desc->MutableOutputDesc("rij_y");
  auto output_rij_z = op_desc->MutableOutputDesc("rij_z");
  std::vector<int64_t> output_rij_dims;
  output_rij_dims.push_back(nsample);
  output_rij_dims.push_back(nloc * nnei * 3);
  output_rij->SetShape(GeShape(output_rij_dims));
  output_rij->SetDataType(coord_type);
  std::vector<int64_t> dims;
  dims.push_back(nsample);
  dims.push_back(nloc * nnei);
  output_nlist->SetShape(GeShape(dims));
  output_nlist->SetDataType(DT_INT32);
  output_distance->SetShape(GeShape(dims));
  output_distance->SetDataType(coord_type);
  output_rij_x->SetShape(GeShape(dims));
  output_rij_x->SetDataType(coord_type);
  output_rij_y->SetShape(GeShape(dims));
  output_rij_y->SetDataType(coord_type);
  output_rij_z->SetShape(GeShape(dims));
  output_rij_z->SetDataType(coord_type);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ProdEnvMatACalcRij, ProdEnvMatACalcRijInferShape);
VERIFY_FUNC_REG(ProdEnvMatACalcRij, ProdEnvMatACalcRijVerify);
// --------------------------ProdEnvMatACalcRij END---------------------

// --------------------------ProdEnvMatACalcDescrpt Begin---------------------
IMPLEMT_VERIFIER(ProdEnvMatACalcDescrpt, ProdEnvMatACalcDescrptVerify) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS,
        OP_LOGE("ProdEnvMatACalcDescrpt", "Failed to get op name of ProdEnvMatACalcDescrpt"),
        return GRAPH_FAILED);

  auto opDesc = OpDescUtils::GetOpDescFromOperator(op);

  GeTensorDescPtr distanceDesc = opDesc->MutableInputDesc("distance");
  std::vector<int64_t> distanceShape = distanceDesc->MutableShape().GetDims();
  CHECK(distanceShape.size() != 2, OP_LOGE(opName.GetString(), "Dim of distanceShape should be 2"), return GRAPH_FAILED);
  int64_t nsamples = distanceShape[0];

  GeTensorDescPtr rijXDesc = opDesc->MutableInputDesc("rij_x");
  std::vector<int64_t> rijXShape = rijXDesc->MutableShape().GetDims();
  CHECK(rijXShape.size() != 2, OP_LOGE(opName.GetString(), "Dim of rij_x shape should be 2"), return GRAPH_FAILED);
  CHECK(rijXShape[0] != nsamples,
        OP_LOGE(opName.GetString(), "Number of rij_x shape samples should match with nsamples"), return GRAPH_FAILED);

  GeTensorDescPtr rijYDesc = opDesc->MutableInputDesc("rij_y");
  std::vector<int64_t> rijYShape = rijYDesc->MutableShape().GetDims();
  CHECK(rijYShape.size() != 2, OP_LOGE(opName.GetString(), "Dim of rij_y shape should be 2"), return GRAPH_FAILED);
  CHECK(rijYShape[0] != nsamples,
        OP_LOGE(opName.GetString(), "Number of rij_y shape samples should match with nsamples"), return GRAPH_FAILED);

  GeTensorDescPtr rijZDesc = opDesc->MutableInputDesc("rij_z");
  std::vector<int64_t> rijZShape = rijZDesc->MutableShape().GetDims();
  CHECK(rijZShape.size() != 2, OP_LOGE(opName.GetString(), "Dim of rij_y shape should be 2"), return GRAPH_FAILED);
  CHECK(rijZShape[0] != nsamples,
        OP_LOGE(opName.GetString(), "Number of rij_y shape samples should match with nsamples"), return GRAPH_FAILED);

  GeTensorDescPtr typeDesc = opDesc->MutableInputDesc("type");
  std::vector<int64_t> typeShape = typeDesc->MutableShape().GetDims();
  CHECK(typeShape.size() != 2, OP_LOGE(opName.GetString(), "Dim of typeShape should be 2"), return GRAPH_FAILED);
  CHECK(typeShape[0] != nsamples,
        OP_LOGE(opName.GetString(), "Number of typeShape samples should match with nsamples"), return GRAPH_FAILED);

  GeTensorDescPtr natomsDesc = opDesc->MutableInputDesc("natoms");
  std::vector<int64_t> natomsShape = natomsDesc->MutableShape().GetDims();
  CHECK(natomsShape.size() != 1, OP_LOGE(opName.GetString(), "Dim of natoms should be 1"), return GRAPH_FAILED);
  CHECK(natomsShape[0] < 3,
        OP_LOGE(opName.GetString(), "Number of atoms should be larger than (or equal to) 3"), return GRAPH_FAILED);

  GeTensorDescPtr davgDesc = opDesc->MutableInputDesc("davg");
  std::vector<int64_t> davgShape = davgDesc->MutableShape().GetDims();
  CHECK(davgShape.size() != 2, OP_LOGE(opName.GetString(), "Dim of davgShape should be 2"), return GRAPH_FAILED);

  GeTensorDescPtr dstdDesc = opDesc->MutableInputDesc("dstd");
  std::vector<int64_t> dstdShape = dstdDesc->MutableShape().GetDims();
  CHECK(dstdShape.size() != 2, OP_LOGE(opName.GetString(), "Dim of dstdShape should be 2"), return GRAPH_FAILED);

  CHECK(distanceDesc->GetDataType() != dstdDesc->GetDataType(),
        OP_LOGE(opName.GetString(), "Data type of distances and std are not match"), return GRAPH_FAILED);
  CHECK(davgDesc->GetDataType() != dstdDesc->GetDataType(),
        OP_LOGE(opName.GetString(), "Data type of avg and std are not match"), return GRAPH_FAILED);

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ProdEnvMatACalcDescrptInferShape) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS,
        OP_LOGE("ProdEnvMatACalcDescrpt", "Failed to get op name of ProdEnvMatACalcDescrpt"),
        return GRAPH_FAILED);

  auto opDesc = OpDescUtils::GetOpDescFromOperator(op);
  auto distanceDesc = opDesc->MutableInputDesc(0);
  CHECK(distanceDesc == nullptr, OP_LOGE(opName.GetString(), "Failed to get distance desc"), return GRAPH_FAILED);
  auto avgDesc = opDesc->MutableInputDesc(5);
  CHECK(avgDesc == nullptr, OP_LOGE(opName.GetString(), "Failed to get avg desc"), return GRAPH_FAILED);

  std::vector<int64_t> distanceDims = distanceDesc->MutableShape().GetDims();
  CHECK(distanceDims.size() != 2, OP_LOGE(opName.GetString(), "Dim of distance should be 2"), return GRAPH_FAILED);
  int64_t nsample = distanceDims[0];
  int64_t descDim = distanceDims[1];
  DataType dataType = distanceDesc->GetDataType();

  std::vector<int64_t> descrptDims = {nsample};
  std::vector<int64_t> descrptDerivDims = {nsample};
  if (descDim > 0) {
    descrptDims.push_back(descDim * 4);
    descrptDerivDims.push_back(descDim * 12);
  } else if (descDim == -1) {
    descrptDims.push_back(-1);
    descrptDerivDims.push_back(-1);
  } else {
    OP_LOGE(opName.GetString(), "No support dim value: %ld", descDim);
    return GRAPH_FAILED;
  }

  auto descrptDesc = opDesc->MutableOutputDesc(0);
  CHECK(descrptDesc == nullptr, OP_LOGE(opName.GetString(), "Failed to get descrpt desc"), return GRAPH_FAILED);
  descrptDesc->SetShape(GeShape(descrptDims));
  descrptDesc->SetOriginShape(GeShape(descrptDims));
  descrptDesc->SetDataType(dataType);
  if (IsUnknownVec(descrptDims)) {
    std::vector<std::pair<int64_t, int64_t>> shapeRange;
    MakeUpShapeRange(descrptDims, shapeRange);
    descrptDesc->SetShapeRange(shapeRange);
  }

  auto descrptDerivDesc = opDesc->MutableOutputDesc(1);
  CHECK(descrptDerivDesc == nullptr, OP_LOGE(opName.GetString(), "Failed to get descrpt_deriv desc"),
        return GRAPH_FAILED);
  descrptDerivDesc->SetShape(GeShape(descrptDerivDims));
  descrptDerivDesc->SetOriginShape(GeShape(descrptDerivDims));
  descrptDerivDesc->SetDataType(dataType);
  if (IsUnknownVec(descrptDerivDims)) {
    std::vector<std::pair<int64_t, int64_t>> shapeRange;
    MakeUpShapeRange(descrptDerivDims, shapeRange);
    descrptDerivDesc->SetShapeRange(shapeRange);
  }

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ProdEnvMatACalcDescrpt, ProdEnvMatACalcDescrptInferShape);
VERIFY_FUNC_REG(ProdEnvMatACalcDescrpt, ProdEnvMatACalcDescrptVerify);
// --------------------------ProdEnvMatACalcDescrpt END---------------------

// --------------------------TabulateFusionGrad Begin---------------------
IMPLEMT_VERIFIER(TabulateFusionGrad, TabulateFusionGradVerify) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS,
        OP_LOGE("TabulateFusionGrad", "Failed to get op name of TabulateFusionGrad"), return GRAPH_FAILED);

  auto opDesc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(opDesc == nullptr, OP_LOGE(opName.GetString(), "Failed to get op desc"), return GRAPH_FAILED);
  GeTensorDescPtr tableDesc = opDesc->MutableInputDesc("table");
  CHECK(tableDesc == nullptr, OP_LOGE(opName.GetString(), "Failed to get table desc"), return GRAPH_FAILED);
  std::vector<int64_t> tableShape = tableDesc->MutableShape().GetDims();
  CHECK(tableShape.size() != 2, OP_LOGE(opName.GetString(), "Dim of table should be 2"), return GRAPH_FAILED);

  GeTensorDescPtr tableInfoDesc = opDesc->MutableInputDesc("table_info");
  CHECK(tableInfoDesc == nullptr, OP_LOGE(opName.GetString(), "Failed to get table info desc"), return GRAPH_FAILED);
  std::vector<int64_t> tableInfoShape = tableInfoDesc->MutableShape().GetDims();
  CHECK(tableInfoShape.size() != 1 || tableInfoShape[0] < 5,
        OP_LOGE(opName.GetString(), "Size of table_info should be greater equal than 5"),
        return GRAPH_FAILED);

  GeTensorDescPtr emXDesc = opDesc->MutableInputDesc("em_x");
  CHECK(emXDesc == nullptr, OP_LOGE(opName.GetString(), "Failed to get em_x desc"), return GRAPH_FAILED);
  std::vector<int64_t> emXShape = emXDesc->MutableShape().GetDims();
  CHECK(emXShape.size() != 2, OP_LOGE(opName.GetString(), "Dim of em_x should be 2"), return GRAPH_FAILED);
  
  GeTensorDescPtr emDesc = opDesc->MutableInputDesc("em");
  CHECK(emDesc == nullptr, OP_LOGE(opName.GetString(), "Failed to get em desc"), return GRAPH_FAILED);
  std::vector<int64_t> emShape = emDesc->MutableShape().GetDims();
  CHECK(emShape.size() != 3, OP_LOGE(opName.GetString(), "Dim of em should be 3"), return GRAPH_FAILED);
  
  GeTensorDescPtr dyDesc = opDesc->MutableInputDesc("dy");
  CHECK(dyDesc == nullptr, OP_LOGE(opName.GetString(), "Failed to get dy desc"), return GRAPH_FAILED);
  std::vector<int64_t> dyShape = dyDesc->MutableShape().GetDims();
  CHECK(dyShape.size() != 3, OP_LOGE(opName.GetString(), "Dim of dy should be 3"), return GRAPH_FAILED);
  
  GeTensorDescPtr descriptorDesc = opDesc->MutableInputDesc("descriptor");
  CHECK(descriptorDesc == nullptr, OP_LOGE(opName.GetString(), "Failed to get descriptor desc"), return GRAPH_FAILED);
  std::vector<int64_t> descriptorShape = descriptorDesc->MutableShape().GetDims();
  CHECK(descriptorShape.size() != 3, OP_LOGE(opName.GetString(), "Dim of descriptor should be 3"), return GRAPH_FAILED);

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(TabulateFusionGradInferShape) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS,
        OP_LOGE("TabulateFusionGrad", "Failed to get op name of TabulateFusionGrad"), return GRAPH_FAILED);

  auto opDesc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr emXDesc = opDesc->MutableInputDesc("em_x");
  std::vector<int64_t> emXDims = emXDesc->MutableShape().GetDims();
  GeTensorDescPtr emDesc = opDesc->MutableInputDesc("em");
  std::vector<int64_t> emDims = emDesc->MutableShape().GetDims();

  GeTensorDescPtr dyDemXDesc = opDesc->MutableOutputDesc("dy_dem_x");
  CHECK(dyDemXDesc == nullptr, OP_LOGE(opName.GetString(), "Failed to get dy_dem_x desc"), return GRAPH_FAILED);
  GeTensorDescPtr dyDemDesc = opDesc->MutableOutputDesc("dy_dem");
  CHECK(dyDemDesc == nullptr, OP_LOGE(opName.GetString(), "Failed to get dy_dem desc"), return GRAPH_FAILED);

  dyDemXDesc->SetDataType(emXDesc->GetDataType());
  dyDemDesc->SetDataType(emDesc->GetDataType());

  if (IsUnknownVec(emXDims)) {
    dyDemXDesc->SetShape(ge::GeShape(emXDims));
    dyDemXDesc->SetOriginShape(ge::GeShape(emXDims));

    std::vector<std::pair<int64_t, int64_t>> dyDemXShapeRange;
    MakeUpShapeRange(emXDims, dyDemXShapeRange);
    dyDemXDesc->SetShapeRange(dyDemXShapeRange);

    dyDemDesc->SetShape(ge::GeShape(emDims));
    dyDemDesc->SetOriginShape(ge::GeShape(emDims));

    std::vector<std::pair<int64_t, int64_t>> dyDemShapeRange;
    MakeUpShapeRange(emDims, dyDemShapeRange);
    dyDemDesc->SetShapeRange(dyDemShapeRange);

    return GRAPH_SUCCESS;
  }

  int32_t splitCount = 1;
  int32_t splitIndex = 0;
  op.GetAttr("split_count", splitCount);
  op.GetAttr("split_index", splitIndex);
  if (splitCount == 1) {
    dyDemXDesc->SetShape(ge::GeShape(emXDims));
    dyDemDesc->SetShape(ge::GeShape(emDims));
  } else if (splitCount == 2){
    int64_t nloc = emDims[0];
    int64_t splitValue = (nloc + splitCount - 1) / splitCount;
    
    if (splitIndex == 0) {
      OP_LOGI(opName.GetString(), "split_index is 0, splitValue=%ld", splitValue);
      dyDemXDesc->SetShape(ge::GeShape({splitValue * emDims[1], 1}));
      dyDemDesc->SetShape(ge::GeShape({splitValue, emDims[1], emDims[2]}));
    } else {
      OP_LOGI(opName.GetString(), "split_index is 1, splitValue=%ld", nloc - splitValue);
      dyDemXDesc->SetShape(ge::GeShape({(nloc - splitValue) * emDims[1], 1}));
      dyDemDesc->SetShape(ge::GeShape({nloc - splitValue, emDims[1], emDims[2]}));
    }
  } else {
    std::string errMsg = GetInputInvalidErrMsg("not support split_count > 2");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), errMsg);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(TabulateFusionGrad, TabulateFusionGradInferShape);
VERIFY_FUNC_REG(TabulateFusionGrad, TabulateFusionGradVerify);
// --------------------------TabulateFusionGrad END---------------------
}  // namespace ge
