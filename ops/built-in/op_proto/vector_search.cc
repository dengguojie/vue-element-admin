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
 * \file vector_search.cc
 * \brief
 */
#include "inc/vector_search.h"

#include <cmath>
#include <string>
#include <vector>

#include "util/util.h"
#include "util/error_util.h"
#include "op_log.h"
#include "graph/utils/op_desc_utils.h"
#include "register/infer_data_slice_registry.h"
#include "util/error_util.h"
#include "graph/common_error_codes.h"
#include "graph/debug/ge_attr_define.h"
#include "axis_util.h"
#include "common_shape_fns.h"
#include "util/vector_proto_profiling.h"

namespace ge {

// --------------------------GenADC---------------------
IMPLEMT_VERIFIER(GenADC, GenADCVerify) {
  auto opDesc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr queryDesc = opDesc->MutableInputDesc("query");
  std::vector<int64_t> queryShape = queryDesc->MutableShape().GetDims();
  if (1 != queryShape.size()) {
    OP_LOGE(TbeGetName(op).c_str(), "Shape of query should be 1 dimensions.");
    return GRAPH_FAILED;
  }
  int64_t dimD = queryShape[0];
  if (0 != dimD % 16) {
    OP_LOGE(TbeGetName(op).c_str(), "Dimesion d should be multiple of 16.");
    return GRAPH_FAILED;
  }

  GeTensorDescPtr codeBookDesc = opDesc->MutableInputDesc("code_book");
  std::vector<int64_t> codeBookShape = codeBookDesc->MutableShape().GetDims();
  if (3 != codeBookShape.size()) {
    OP_LOGE(TbeGetName(op).c_str(), "Shape of code book should be 3 dimensions.");
    return GRAPH_FAILED;
  }
  if (dimD != codeBookShape[0] * codeBookShape[2]) {
    OP_LOGE(TbeGetName(op).c_str(), "Failed to check dimensions: d = M * dsub.");
    return GRAPH_FAILED;
  }

  GeTensorDescPtr centroidsDesc = opDesc->MutableInputDesc("centroids");
  std::vector<int64_t> centroidsShape = centroidsDesc->MutableShape().GetDims();
  if (2 != centroidsShape.size()) {
    OP_LOGE(TbeGetName(op).c_str(), "Shape of centroids should be 2 dimensions.");
    return GRAPH_FAILED;
  }
  if (dimD != centroidsShape[1]) {
    OP_LOGE(TbeGetName(op).c_str(), "The 2nd dimension of centroids should be equal to the 1st dimension of query.");
    return GRAPH_FAILED;
  }

  GeTensorDescPtr bucketListDesc = opDesc->MutableInputDesc("bucket_list");
  std::vector<int64_t> bucketListShape = bucketListDesc->MutableShape().GetDims();
  if (1 != bucketListShape.size()) {
    OP_LOGE(TbeGetName(op).c_str(), "Shape of bucket list should be 1 dimensions.");
    return GRAPH_FAILED;
  }

  std::vector<std::string> inputs{"code_book", "centroids"};
  if (!CheckInputDtypeSame(op, inputs)) {
    OP_LOGE(TbeGetName(op).c_str(), "Input dtypes are not the same.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(GenADCInferShape) {
  auto opDesc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr bucketListDesc = opDesc->MutableInputDesc("bucket_list");
  std::vector<int64_t> bucketListShape = bucketListDesc->MutableShape().GetDims();
  std::vector<std::pair<int64_t, int64_t>> bucketListShapeRange;
  opDesc->MutableInputDesc("bucket_list")->GetShapeRange(bucketListShapeRange);
  MakeUpShapeRange(bucketListShape, bucketListShapeRange);

  GeTensorDescPtr codeBookDesc = opDesc->MutableInputDesc("code_book");
  std::vector<int64_t> codeBookShape = codeBookDesc->MutableShape().GetDims();
  std::vector<std::pair<int64_t, int64_t>> codeBookShapeRange;
  MakeUpShapeRange(codeBookShape, codeBookShapeRange);

  GeTensorDescPtr adcTablesDesc = opDesc->MutableOutputDesc("adc_tables");
  std::vector<int64_t> adcTablesShape;
  std::vector<std::pair<int64_t, int64_t>> adcTablesShapeRange;

  adcTablesShape.push_back(bucketListShape[0]);
  adcTablesShapeRange.push_back(bucketListShapeRange[0]);
  for (size_t i = 0; i < codeBookShape.size() - 1; i++) {
    adcTablesShape.push_back(codeBookShape[i]);
    adcTablesShapeRange.push_back(codeBookShapeRange[i]);
  }

  adcTablesDesc->SetShape(ge::GeShape(adcTablesShape));
  adcTablesDesc->SetShapeRange(adcTablesShapeRange);
  adcTablesDesc->SetDataType(ge::DT_FLOAT16);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(GenADC, GenADCInferShape);
VERIFY_FUNC_REG(GenADC, GenADCVerify);
// --------------------------GenADC END---------------------

// ----------------TopKPQDistance Begin-------------------
IMPLEMT_COMMON_INFERFUNC(TopKPQDistanceInferShape) {
  int32_t topK = 0;
  if (op.GetAttr("k", topK) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "get attr k from op failed");
    return GRAPH_FAILED;
  }
  vector<int64_t> outputDims = {topK};

  ge::TensorDesc inputPqDistanceTensorDesc = op.GetDynamicInputDesc("pq_distance", 0);
  ge::TensorDesc inputPqivfTensorDesc = op.GetDynamicInputDesc("pq_ivf", 0);
  ge::TensorDesc inputPqindexTensorDesc = op.GetDynamicInputDesc("pq_index", 0);
  DataType distanceDtype = inputPqDistanceTensorDesc.GetDataType();
  DataType pqivfDtype = inputPqivfTensorDesc.GetDataType();
  DataType pqindexDtype = inputPqindexTensorDesc.GetDataType();

  ge::TensorDesc outputDistanceDesc = op.GetOutputDescByName("topk_distance");
  outputDistanceDesc.SetShape(ge::Shape(outputDims));
  outputDistanceDesc.SetDataType(distanceDtype);
  ge::TensorDesc outputIvfDesc = op.GetOutputDescByName("topk_ivf");
  outputIvfDesc.SetShape(ge::Shape(outputDims));
  outputIvfDesc.SetDataType(pqivfDtype);
  ge::TensorDesc outputIndexDesc = op.GetOutputDescByName("topk_index");
  outputIndexDesc.SetShape(ge::Shape(outputDims));
  outputIndexDesc.SetDataType(pqindexDtype);

  CHECK(op.UpdateOutputDesc("topk_distance", outputDistanceDesc) != GRAPH_SUCCESS,
        OP_LOGE(TbeGetName(op).c_str(), "Update topk_distance outputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("topk_ivf", outputIvfDesc) != GRAPH_SUCCESS,
        OP_LOGE(TbeGetName(op).c_str(), "Update topk_ivf outputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("topk_index", outputIndexDesc) != GRAPH_SUCCESS,
        OP_LOGE(TbeGetName(op).c_str(), "Update topk_index outputDesc failed."), return GRAPH_FAILED);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(TopKPQDistance, TopKPQDistanceVerify) {
  OP_LOGI(TbeGetName(op).c_str(), "TopKPQDistanceVerify begin");
  int32_t groupSize = 0;
  if (op.GetAttr("group_size", groupSize) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "get attr group_size from op failed");
    return GRAPH_FAILED;
  }
  if(groupSize <= 0){
    OP_LOGE(TbeGetName(op).c_str(), "groupSize[%d] must greater than 0", groupSize);
    return GRAPH_FAILED;
  }

  int32_t k = 0;
  if (op.GetAttr("k", k) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "get attr k from op failed");
    return GRAPH_FAILED;
  }
  size_t inputSize = op.GetInputsSize();
  OP_LOGI(TbeGetName(op).c_str(), "inputSize is [%d], groupSize is [%d], k is [%d]", inputSize, groupSize, k);

  constexpr int32_t INPUT_N = 5;
  if (inputSize / INPUT_N == 0) {
    string msg = ConcatString("op inputSize error, inputSize is", inputSize);
    std::string err_msg = OtherErrMsg(msg);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  for (int32_t i = 0; i < static_cast<int32_t>(inputSize) / INPUT_N; i++) {
    std::vector<int64_t> pqDistanceDims = op.GetDynamicInputDesc("pq_distance", i).GetShape().GetDims();
    std::vector<int64_t> pqIvfDims = op.GetDynamicInputDesc("pq_ivf", i).GetShape().GetDims();
    std::vector<int64_t> pqIndexDims = op.GetDynamicInputDesc("pq_index", i).GetShape().GetDims();
    string msg = ConcatString("The shape of pq_distance is:", DebugString(pqDistanceDims),
                              "The shape of pq_ivf is:", DebugString(pqIvfDims),
                              "The shape of pq_index is:", DebugString(pqIndexDims), ".They must be the same");
    OP_LOGI(TbeGetName(op).c_str(), "input shape:[%d]", OtherErrMsg(msg).c_str());
    if (!(pqDistanceDims == pqIvfDims && pqIvfDims == pqIndexDims)) {
      std::string err_msg = OtherErrMsg(msg);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
    std::vector<int64_t> extremeDistanceDims =
        op.GetDynamicInputDesc("grouped_extreme_distance", i).GetShape().GetDims();
    OP_LOGI(TbeGetName(op).c_str(), "extremeDistanceDims shape:[%s]", DebugString(extremeDistanceDims).c_str());

    if (extremeDistanceDims.size() <= 0) {
      std::string err_msg = OtherErrMsg("extremeDistanceDims is empty");
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }

  OP_LOGI(TbeGetName(op).c_str(), "TopKPQDistanceVerify end");
  return GRAPH_SUCCESS;
}
// Registered inferfunction
COMMON_INFER_FUNC_REG(TopKPQDistance, TopKPQDistanceInferShape);
// Registered verify function
VERIFY_FUNC_REG(TopKPQDistance, TopKPQDistanceVerify);
// ----------------TopKPQDistance End---------------------

// ----------------ScanPQCodes Begin-------------------
const int64_t IVF_LAST_DIM_1 = 16;
const int64_t IVF_LAST_DIM_2 = 32;
const int64_t IVF_LAST_DIM_3 = 64;
const int64_t BUCKET_SHAPE = 1;
const int64_t ADC_TABLE_SHAPE = 256;
const int64_t GROUP_SIZE_BASE = 64;
const int64_t EXTREME_MODE_NUM = 2;
graphStatus ScanPQCodesVerifyAttrs(op::ScanPQCodes op){
  int32_t totalLimit = 0;
  CHECK(ge::GRAPH_SUCCESS != op.GetAttr("total_limit", totalLimit),
        OP_LOGE(TbeGetName(op).c_str(), "ScanPQCodes GetOpAttr total_limit failed."),
        return GRAPH_FAILED);
  int32_t groupSize = 0;
  CHECK(ge::GRAPH_SUCCESS != op.GetAttr("group_size", groupSize),
        OP_LOGE(TbeGetName(op).c_str(), "ScanPQCodes GetOpAttr group_size failed."),
        return GRAPH_FAILED);
  CHECK(groupSize % GROUP_SIZE_BASE != 0 || groupSize > totalLimit,
        OP_LOGE(TbeGetName(op).c_str(), "ScanPQCodes GetOpAttr group_size is unavalible."),
        return GRAPH_FAILED);
  int32_t extremeMode = 0;
  CHECK(ge::GRAPH_SUCCESS != op.GetAttr("extreme_mode", extremeMode),
        OP_LOGE(TbeGetName(op).c_str(), "ScanPQCodes GetOpAttr extreme_mode failed."),
        return GRAPH_FAILED);
  CHECK(extremeMode >= EXTREME_MODE_NUM,
        OP_LOGE(TbeGetName(op).c_str(), "ScanPQCodes GetOpAttr extreme_mode more than 2."),
        return GRAPH_FAILED);
  int32_t splitCount = 0;
  CHECK(ge::GRAPH_SUCCESS != op.GetAttr("split_count", splitCount),
        OP_LOGE(TbeGetName(op).c_str(), "ScanPQCodes GetOpAttr split_count failed."),
        return GRAPH_FAILED);
  CHECK(splitCount <= 0,
        OP_LOGE(TbeGetName(op).c_str(), "ScanPQCodes GetOpAttr split_count is unavaliable."),
        return GRAPH_FAILED);
  int32_t splitIndex = 0;
  CHECK(ge::GRAPH_SUCCESS != op.GetAttr("split_index", splitIndex),
        OP_LOGE(TbeGetName(op).c_str(), "ScanPQCodes GetOpAttr split_index failed."),
        return GRAPH_FAILED);
  CHECK(splitIndex > splitCount - 1,
        OP_LOGE(TbeGetName(op).c_str(), "ScanPQCodes GetOpAttr split_index is more than splitCount."),
        return GRAPH_FAILED);
  return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(ScanPQCodes, ScanPQCodesVerify) {
  auto opDesc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr ivfDesc = opDesc->MutableInputDesc("ivf");
  std::vector<int64_t> ivfShape = ivfDesc->MutableShape().GetDims();
  int64_t dimLast = ivfShape[ivfShape.size() - 1];
  std::vector<int64_t> ivfLastDims {IVF_LAST_DIM_1, IVF_LAST_DIM_2, IVF_LAST_DIM_3};
  CHECK((find(ivfLastDims.begin(), ivfLastDims.end(), dimLast) == ivfLastDims.end()),
        OP_LOGE(TbeGetName(op).c_str(), "Last dimesion %ld of ivf should be equl to 16, 32, 64.", dimLast),
        return GRAPH_FAILED);
  GeTensorDescPtr bucketListDesc = opDesc->MutableInputDesc("bucket_list");
  std::vector<int64_t> bucketListShape = bucketListDesc->MutableShape().GetDims();
  CHECK(BUCKET_SHAPE != bucketListShape.size(),
        OP_LOGE(TbeGetName(op).c_str(), "Shape of code bucket_list should be 1 dimension."),
        return GRAPH_FAILED);
  GeTensorDescPtr bucketBaseDistanceDesc = opDesc->MutableInputDesc("bucket_base_distance");
  std::vector<int64_t> bucketBaseDistanceShape = bucketBaseDistanceDesc->MutableShape().GetDims();
  CHECK(BUCKET_SHAPE != bucketBaseDistanceShape.size(),
        OP_LOGE(TbeGetName(op).c_str(), "Shape of code bucketBaseDistanceShape should be 1 dimension."),
        return GRAPH_FAILED);
  GeTensorDescPtr bucketLimitsDesc = opDesc->MutableInputDesc("bucket_limits");
  std::vector<int64_t> bucketLimitsShape = bucketLimitsDesc->MutableShape().GetDims();
  CHECK(BUCKET_SHAPE != bucketLimitsShape.size(),
        OP_LOGE(TbeGetName(op).c_str(), "Shape of code bucketLimitsShape should be 1 dimension."),
        return GRAPH_FAILED);
  GeTensorDescPtr bucketOffsetsDesc = opDesc->MutableInputDesc("bucket_offsets");
  std::vector<int64_t> bucketOffsetsShape = bucketOffsetsDesc->MutableShape().GetDims();
  CHECK(BUCKET_SHAPE != bucketOffsetsShape.size(),
        OP_LOGE(TbeGetName(op).c_str(), "Shape of code bucketOffsetsShape should be 1 dimension."),
        return GRAPH_FAILED);
  GeTensorDescPtr adcTablesDesc = opDesc->MutableInputDesc("adc_tables");
  std::vector<int64_t> adcTablesShape = adcTablesDesc->MutableShape().GetDims();
  int64_t dimM = adcTablesShape[1];
  int64_t dimKsub = adcTablesShape[2];
  CHECK((find(ivfLastDims.begin(), ivfLastDims.end(), dimM) == ivfLastDims.end()),
        OP_LOGE(TbeGetName(op).c_str(), "M dimesion of adc_tables should be equl to 16, 32, 64."),
        return GRAPH_FAILED);
  CHECK(ADC_TABLE_SHAPE != dimKsub,
        OP_LOGE(TbeGetName(op).c_str(), "ksub dimesion of adc_tables should be equl to 256."),
        return GRAPH_FAILED);
  return ScanPQCodesVerifyAttrs(op);
}

const int64_t SLICE_SIZE = 1024;
IMPLEMT_COMMON_INFERFUNC(ScanPQCodesShape) {
  auto opDest = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr bucketListDesc = opDest->MutableInputDesc("bucket_list");
  std::vector<int64_t> bucketListShape = bucketListDesc->MutableShape().GetDims();
  int64_t bucketNumbers = bucketListShape[0];
  opDest->MutableOutputDesc("actual_count")->SetDataType(ge::DT_INT32);
  opDest->MutableOutputDesc("pq_distance")->SetDataType(ge::DT_FLOAT16);
  opDest->MutableOutputDesc("grouped_extreme_distance")->SetDataType(ge::DT_FLOAT16);
  opDest->MutableOutputDesc("pq_ivf")->SetDataType(ge::DT_INT32);
  opDest->MutableOutputDesc("pq_index")->SetDataType(ge::DT_INT32);
  int32_t totalLimit = 0;
  op.GetAttr("total_limit", totalLimit);
  int32_t groupSize = 0;
  op.GetAttr("group_size", groupSize);
  int32_t spliteCount = 0;
  op.GetAttr("split_count", spliteCount);
  totalLimit = totalLimit / spliteCount + bucketNumbers * SLICE_SIZE;
  std::vector<int64_t> outShape;
  std::vector<int64_t> outShapeExtremeDistance;
  std::vector<int64_t> outShapeIndex;
  outShape.push_back(totalLimit);
  if (groupSize == 0) {
    OP_LOGE(TbeGetName(op).c_str(), "group_size can not be 0");
    return GRAPH_FAILED;
  }
  outShapeExtremeDistance.push_back(totalLimit / groupSize);
  outShapeIndex.push_back(totalLimit);
  op.SetAttr("total_limit", totalLimit);
  opDest->MutableOutputDesc("actual_count")->SetShape(GeShape({1}));
  opDest->MutableOutputDesc("pq_distance")->SetShape(GeShape(outShape));
  opDest->MutableOutputDesc("grouped_extreme_distance")->SetShape(GeShape(outShapeExtremeDistance));
  opDest->MutableOutputDesc("pq_ivf")->SetShape(GeShape(outShapeIndex));
  opDest->MutableOutputDesc("pq_index")->SetShape(GeShape(outShapeIndex));
  return GRAPH_SUCCESS;
}

// Registered verify function
COMMON_INFER_FUNC_REG(ScanPQCodes, ScanPQCodesShape);
VERIFY_FUNC_REG(ScanPQCodes, ScanPQCodesVerify);
// ----------------ScanPQCodes END---------------------

// ----------------CalcBucketsLimitAndOffset Begin-------------------
IMPLEMT_COMMON_INFERFUNC(CalcBucketsLimitAndOffsetInferShape) {
  TensorDesc td = op.GetInputDesc("bucket_list");
  (void)op.UpdateOutputDesc("buckets_limit", td);
  TensorDesc td_ivf_offset = op.GetInputDesc("ivf_offset");
  td.SetDataType(td_ivf_offset.GetDataType());
  (void)op.UpdateOutputDesc("buckets_offset", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(CalcBucketsLimitAndOffset, CalcBucketsLimitAndOffsetInferShape);
// ----------------CalcBucketsLimitAndOffset END---------------------

}  // namespace ge
