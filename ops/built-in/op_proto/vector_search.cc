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

}  // namespace ge
