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
    OP_LOGE(op.GetName().c_str(), "Shape of query should be 1 dimensions.");
    return GRAPH_FAILED;
  }
  int64_t dimD = queryShape[0];
  if (0 != dimD % 16) {
    OP_LOGE(op.GetName().c_str(), "Dimesion d should be multiple of 16.");
    return GRAPH_FAILED;
  }

  GeTensorDescPtr codeBookDesc = opDesc->MutableInputDesc("code_book");
  std::vector<int64_t> codeBookShape = codeBookDesc->MutableShape().GetDims();
  if (3 != codeBookShape.size()) {
    OP_LOGE(op.GetName().c_str(), "Shape of code book should be 3 dimensions.");
    return GRAPH_FAILED;
  }
  if (dimD != codeBookShape[0] * codeBookShape[2]) {
    OP_LOGE(op.GetName().c_str(), "Failed to check dimensions: d = M * dsub.");
    return GRAPH_FAILED;
  }

  GeTensorDescPtr centroidsDesc = opDesc->MutableInputDesc("centroids");
  std::vector<int64_t> centroidsShape = centroidsDesc->MutableShape().GetDims();
  if (2 != centroidsShape.size()) {
    OP_LOGE(op.GetName().c_str(), "Shape of centroids should be 2 dimensions.");
    return GRAPH_FAILED;
  }
  if (dimD != centroidsShape[1]) {
    OP_LOGE(op.GetName().c_str(), "The 2nd dimension of centroids should be equal to the 1st dimension of query.");
    return GRAPH_FAILED;
  }

  GeTensorDescPtr bucketListDesc = opDesc->MutableInputDesc("bucket_list");
  std::vector<int64_t> bucketListShape = bucketListDesc->MutableShape().GetDims();
  if (1 != bucketListShape.size()) {
    OP_LOGE(op.GetName().c_str(), "Shape of bucket list should be 1 dimensions.");
    return GRAPH_FAILED;
  }

  std::vector<std::string> inputs{"code_book", "centroids"};
  if (!CheckInputDtypeSame(op, inputs)) {
    OP_LOGE(op.GetName().c_str(), "Input dtypes are not the same.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(GenADCInferShape) {
  auto opDesc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr bucketListDesc = opDesc->MutableInputDesc("bucket_list");
  std::vector<int64_t> bucketListShape = bucketListDesc->MutableShape().GetDims();
  DataType bucketListDtype = op.GetInputDesc("bucket_list").GetDataType();
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
    OP_LOGE(op.GetName().c_str(), "get attr k from op failed");
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
        OP_LOGE(op.GetName().c_str(), "Update topk_distance outputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("topk_ivf", outputIvfDesc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "Update topk_ivf outputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("topk_index", outputIndexDesc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "Update topk_index outputDesc failed."), return GRAPH_FAILED);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(TopKPQDistance, TopKPQDistanceVerify) {
  OP_LOGI(op.GetName().c_str(), "TopKPQDistanceVerify begin");
  int32_t groupSize = 0;
  if (op.GetAttr("group_size", groupSize) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attr group_size from op failed");
    return GRAPH_FAILED;
  }
  if(groupSize <= 0){
    OP_LOGE(op.GetName().c_str(), "groupSize[%d] must greater than 0", groupSize);
    return GRAPH_FAILED;
  }

  int32_t k = 0;
  if (op.GetAttr("k", k) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attr k from op failed");
    return GRAPH_FAILED;
  }
  size_t inputSize = op.GetInputsSize();
  OP_LOGI(op.GetName().c_str(), "inputSize is [%d], groupSize is [%d], k is [%d]", inputSize, groupSize, k);

  constexpr int32_t INPUT_N = 5;
  if (inputSize / INPUT_N == 0) {
    string msg = ConcatString("op inputSize error, inputSize is", inputSize);
    std::string err_msg = OtherErrMsg(msg);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int32_t extremeDistanceSum = 0;
  for (int32_t i = 0; i < inputSize / INPUT_N; i++) {
    std::vector<int64_t> pqDistanceDims = op.GetDynamicInputDesc("pq_distance", i).GetShape().GetDims();
    std::vector<int64_t> pqIvfDims = op.GetDynamicInputDesc("pq_ivf", i).GetShape().GetDims();
    std::vector<int64_t> pqIndexDims = op.GetDynamicInputDesc("pq_index", i).GetShape().GetDims();
    string msg = ConcatString("The shape of pq_distance is:", DebugString(pqDistanceDims),
                              "The shape of pq_ivf is:", DebugString(pqIvfDims),
                              "The shape of pq_index is:", DebugString(pqIndexDims), ".They must be the same");
    OP_LOGI(op.GetName().c_str(), "input shape:[%d]", OtherErrMsg(msg).c_str());
    if (!(pqDistanceDims == pqIvfDims && pqIvfDims == pqIndexDims)) {
      std::string err_msg = OtherErrMsg(msg);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    std::vector<int64_t> extremeDistanceDims =
        op.GetDynamicInputDesc("grouped_extreme_distance", i).GetShape().GetDims();
    OP_LOGI(op.GetName().c_str(), "extremeDistanceDims shape:[%s]", DebugString(extremeDistanceDims).c_str());

    if (extremeDistanceDims.size() > 0) {
      extremeDistanceSum += extremeDistanceDims[0];
    } else {
      std::string err_msg = OtherErrMsg("extremeDistanceDims is empty");
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }
  if (k > extremeDistanceSum) {
        string msg = ConcatString("extremeDistanceSum:", extremeDistanceSum, "k is:",k ,
                                  "extremeDistanceDims[0] must greater than or equal to k.");
        std::string err_msg = OtherErrMsg(msg);
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
        return GRAPH_FAILED;
  }
  OP_LOGI(op.GetName().c_str(), "TopKPQDistanceVerify end");
  return GRAPH_SUCCESS;
}
// Registered inferfunction
COMMON_INFER_FUNC_REG(TopKPQDistance, TopKPQDistanceInferShape);
// Registered verify function
VERIFY_FUNC_REG(TopKPQDistance, TopKPQDistanceVerify);
// ----------------TopKPQDistance End---------------------

}  // namespace ge
