/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file parsing_ops.cpp
 * \brief
 */
#include "inc/parsing_ops.h"
#include "common/inc/op_log.h"
#include "common_shape_fns.h"
#include "graph/operator.h"
#include "util/util.h"
#include "util/common_shape_fns.h"
#include "util/error_util.h"
#include "op_log.h"
#include "graph/utils/node_utils.h"
#include <string>

namespace ge {

IMPLEMT_INFERFUNC(StringToNumber, StringToNumberInfer) {
  TensorDesc out_desc = op.GetOutputDesc("y");
  DataType out_type;
  if (op.GetAttr("out_type", out_type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attribute failed");
    return GRAPH_FAILED;
  }
  if ((out_type != DT_FLOAT) || (out_type != DT_DOUBLE) || (out_type != DT_INT32) || (out_type != DT_INT64)) {
    OP_LOGE(op.GetName().c_str(), "out_type type not supported");
    return GRAPH_FAILED;
  }

  out_desc.SetDataType(out_type);
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update y failed");
    return GRAPH_FAILED;
  }
  return UnchangedShape(op, "x", "y");
}
INFER_FUNC_REG(StringToNumber, StringToNumberInfer);

IMPLEMT_INFERFUNC(DecodeRaw, DecodeRawInfer) {
  auto x1_tensor = op.GetInputDesc(0);
  Shape s = x1_tensor.GetShape();
  std::vector<int64_t> dims;
  for (size_t i = 0; i< s.GetDimNum(); i++) {
    dims.push_back(s.GetDim(i));
  }
  dims.push_back(UNKNOWN_DIM);
  Shape output_shape(dims);
  DataType dtype;
  if (op.GetAttr("out_type", dtype) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get attr dtype failed");
    return GRAPH_FAILED;
  }
  TensorDesc y_tensor = op.GetOutputDesc("output");
  y_tensor.SetDataType(dtype);
  y_tensor.SetShape(output_shape);
  if (op.UpdateOutputDesc("output", y_tensor) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update output failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(DecodeRaw, DecodeRawInfer);

IMPLEMT_INFERFUNC(ParseTensor, ParseTensorInfer) {
  Shape output_shape(ge::UNKNOWN_RANK);
  DataType out_type;
  if (op.GetAttr("out_type", out_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
                                       string("get attr[out_type] failed"));
    return GRAPH_FAILED;
  }

  TensorDesc output_desc = op.GetOutputDesc("output");
  output_desc.SetShape(output_shape);
  output_desc.SetDataType(out_type);
  if (op.UpdateOutputDesc("output", output_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[output] desc failed"));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(ParseTensor, ParseTensorInfer);

IMPLEMT_INFERFUNC(ParseSingleExample, ParseSingleExampleInfer) {
  auto x1_tensor = op.GetInputDesc("serialized");
  Shape x1_shape;
  if (WithRank(x1_tensor, 0, x1_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call WithRank function, input[serialized] rank must be 0, "
        "got rank[",
        x1_shape.GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<ge::DataType> sparse_types;
  if (op.GetAttr("sparse_types", sparse_types) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
                                       string("get attr[sparse_types] failed"));
    return GRAPH_FAILED;
  }
  std::vector<std::string> sparse_keys;
  if (op.GetAttr("sparse_keys", sparse_keys) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
                                       string("get attr[sparse_keys] failed"));
    return GRAPH_FAILED;
  }
  int32_t num_sparse = 0;
  if (op.GetAttr("num_sparse", num_sparse) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
                                       string("get attr[num_sparse] failed"));
    return GRAPH_FAILED;
  }
  if (num_sparse != static_cast<int32_t>(sparse_keys.size())){
    std::string err_msg = GetAttrSizeErrMsg(
        "num_sparse", std::to_string(num_sparse),
        std::to_string(sparse_keys.size()));
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  
  for (int i = 0; i < num_sparse; ++i) {
    TensorDesc output_desc = op.GetDynamicOutputDesc("sparse_indices", i);
    output_desc.SetShape(Shape({UNKNOWN_DIM,1}));
    output_desc.SetDataType(DT_INT64);
    op.UpdateDynamicOutputDesc("sparse_indices", i, output_desc);
    
    output_desc = op.GetDynamicOutputDesc("sparse_values", i);
    output_desc.SetShape(Shape({UNKNOWN_DIM}));
    output_desc.SetDataType(sparse_types[i]);
    op.UpdateDynamicOutputDesc("sparse_values", i, output_desc);
    
    output_desc = op.GetDynamicOutputDesc("sparse_shapes", i);
    output_desc.SetShape(Shape({1}));
    output_desc.SetDataType(DT_INT64);
    op.UpdateDynamicOutputDesc("sparse_shapes", i, output_desc);
  }

  std::vector<std::string> dense_keys;
  if (op.GetAttr("dense_keys", dense_keys) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
                                       string("get attr[dense_keys] failed"));
    return GRAPH_FAILED;
  }
  int size = dense_keys.size();
  std::vector<std::vector<int64_t>> dense_shapes;
  if (op.GetAttr("dense_shapes", dense_shapes) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
                                       string("get attr[dense_shapes] failed"));
    return GRAPH_FAILED;
  }
  
  std::vector<ge::DataType> tdense_type;
  if (op.GetAttr("Tdense", tdense_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
                                       string("get attr[Tdense] failed"));
    return GRAPH_FAILED;
  }

  for (int i = 0; i < size; ++i) {
    TensorDesc output_desc = op.GetDynamicOutputDesc("dense_values", i);
    output_desc.SetShape(Shape(std::vector<int64_t>({dense_shapes[i]})));
    output_desc.SetDataType(tdense_type[i]);
    op.UpdateDynamicOutputDesc("dense_values", i, output_desc);
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(ParseSingleExample, ParseSingleExampleInfer);

IMPLEMT_INFERFUNC(DecodeCSV, DecodeCSVInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc == nullptr) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("get op desc failed, op desc is nullptr"));
    return GRAPH_FAILED;
  }

  size_t inputs_size = op_desc->GetInputsSize();
  size_t record_defaults_size = inputs_size - 1;
  for (size_t i = 0; i < record_defaults_size; ++i) {
    GeShape record_default_shape;
    auto temp_record_default_desc =
        op_desc->MutableInputDesc("record_defaults" + std::to_string(i));
    if (temp_record_default_desc == nullptr) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(
          op.GetName(), string("get input[record_default] desc failed, "
                               "input[record_default] desc is nullptr"));
      return GRAPH_FAILED;
    }

    std::string err_msg;
    if (WithRankAtMost(temp_record_default_desc, 1, record_default_shape,
        op.GetName().c_str()) != GRAPH_SUCCESS) {
      err_msg = GetShapeErrMsg(
          i, DebugString(temp_record_default_desc->GetShape().GetDims()),
          "at most 1D");
      err_msg = string(
                    "failed to call WithRankAtMost function, dynamic input "
                    "record_defaults") +
                err_msg;
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    if (record_default_shape.GetDimNum() == 1 &&
        record_default_shape.GetDim(0) > 1) {
      err_msg =
          ConcatString("shape of dynamic input record_defaults[", i,
                       "] must be at length-0 or length-1 vector or a "
                       "scalar, got record_defaults[",
                       i, "][0] dim[", record_default_shape.GetDim(0), "]");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_PARAM_INVALID;
    }
  }

  size_t outputs_size = op_desc->GetOutputsSize();
  auto temp_records_desc = op.GetInputDesc("records");
  for (size_t i = 0; i < outputs_size; ++i) {
    auto temp_record_default_desc =
        op.GetDynamicInputDesc("record_defaults", i);
    auto temp_output_desc = op.GetDynamicOutputDesc("output", i);
    (void)FillOpDesc(temp_output_desc, temp_records_desc.GetShape(),
                     temp_record_default_desc.GetDataType());
    op.UpdateDynamicOutputDesc("output", i, temp_output_desc);
  }

  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(DecodeCSV, DecodeCSVInfer);

IMPLEMT_INFERFUNC(ParseExample, ParseExampleInfer) {
  std::vector<bool> variable_length;
  std::vector<size_t> elements_per_stride;
  int32_t num_sparse;
  if (op.GetAttr("Nsparse", num_sparse) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
                                       string("failed to get attr[Nsparse]."));
    return GRAPH_FAILED;
  }
  std::string err_msg;
  if (num_sparse < 0) {
    err_msg = ConcatString("attr[Nsparse] must be non-negative, got[",
                           num_sparse, "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int32_t num_dense;
  if (op.GetAttr("Ndense", num_dense) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
                                       string("failed to get attr[Ndense]."));
    return GRAPH_FAILED;
  }
  if (num_dense < 0) {
    err_msg =
        ConcatString("attr[Ndense] must be non-negative, got[", num_dense, "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  Operator::OpListType sparse_types;
  if (op.GetAttr("sparse_types", sparse_types) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("failed to get attr[sparse_types]."));
    return GRAPH_FAILED;
  }

  Operator::OpListType dense_types;
  if (op.GetAttr("Tdense", dense_types) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
                                       string("failed to get attr[Tdense]."));
    return GRAPH_FAILED;
  }

  Operator::OpListListInt dense_shapes;
  if (op.GetAttr("dense_shapes", dense_shapes) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("failed to get attr[dense_shapes]."));
    return GRAPH_FAILED;
  }

  for (size_t i = 0; i < dense_shapes.size(); ++i) {
    GeShape temp_dense_shape(dense_shapes[i]);
    if (!ShapeFullyDefined(temp_dense_shape)) {
      OP_LOGW(op.GetName().c_str(), "dense_shapes[%ld] is not fully defined.",
              i);
    }
    std::vector<int64_t> dense_shape;
    if (dense_shapes[i].size() > 0 && dense_shapes[i][0] == -1) {
      variable_length.push_back(true);
      for (size_t d = 1; d < dense_shapes[i].size(); ++d) {
        dense_shape.push_back(dense_shapes[i][d]);
      }
    } else {
      variable_length.push_back(false);
      dense_shape = dense_shapes[i];
    }
    int64_t dense_shape_size = 1;
    for (size_t d = 0; d < dense_shape.size(); ++d) {
      if (dense_shape[i] < 0) {
        dense_shape_size = -1;
        break;
      }
      dense_shape_size *= dense_shape[i];
    }
    elements_per_stride.push_back(dense_shape_size);
  }

  if (num_sparse != static_cast<int32_t>(sparse_types.size())) {
    err_msg = ConcatString("attr[Nsparse] value[", num_sparse,
                           "] is not equal to attr[sparse_types] size[",
                           sparse_types.size(), "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (num_dense != static_cast<int32_t>(dense_types.size())) {
    err_msg = ConcatString("attr[Ndense] value[", num_dense,
                           "] is not equal to attr[dense_types] size[",
                           dense_types.size(), "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (num_dense != static_cast<int32_t>(dense_shapes.size())) {
    err_msg = ConcatString("attr[Ndense] value[", num_dense,
                           "] is not equal to attr[dense_shapes] size[",
                           dense_shapes.size(), "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (num_dense > std::numeric_limits<int32_t>::max()) {
    err_msg = ConcatString("attr[Ndense] value[", num_dense,
                           "] is bigger than in32 max");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  for (auto type : sparse_types) {
    if (type != DT_INT64 && type != DT_FLOAT && type != DT_STRING) {
      std::string sparse_dt = DTypeStr(type);
      err_msg = ConcatString(
          "invalid data type[", sparse_dt,
          "] of attr[sparse_types], should be DT_INT64, DT_FLOAT or DT_STRING");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }

  for (auto type : dense_types) {
    if (type != DT_INT64 && type != DT_FLOAT && type != DT_STRING) {
      std::string dense_dt = DTypeStr(type);
      err_msg = ConcatString(
          "invalid data type[", dense_dt,
          "] of attr[Tdense], should be DT_INT64, DT_FLOAT or DT_STRING");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  GeShape serialized_shape;
  auto serialized_desc = op_desc->MutableInputDesc(0);
  if (WithRank(serialized_desc, 1, serialized_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(
        0, DebugString(serialized_desc->GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  GeShape unused_shape;
  auto unused_desc = op_desc->MutableInputDesc(1);
  if (WithRank(unused_desc, 1, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    err_msg =
        GetShapeErrMsg(1, DebugString(unused_desc->GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  Shape sparse_indices_shape({UNKNOWN_DIM, 1});
  for (int i = 0; i < num_sparse; ++i) {
    auto temp_sparse_indices_desc =
        op.GetDynamicOutputDesc("sparse_indices", i);
    (void)FillOpDesc(temp_sparse_indices_desc, sparse_indices_shape, DT_INT64);
    op.UpdateDynamicOutputDesc("sparse_indices", i, temp_sparse_indices_desc);
  }

  Shape sparse_values_shape({UNKNOWN_DIM});
  for (int i = 0; i < num_sparse; ++i) {
    auto temp_sparse_values_desc = op.GetDynamicOutputDesc("sparse_values", i);
    (void)FillOpDesc(temp_sparse_values_desc, sparse_values_shape,
                     sparse_types[i]);
    op.UpdateDynamicOutputDesc("sparse_values", i, temp_sparse_values_desc);
  }

  Shape sparse_shapes_shape({1});
  for (int i = 0; i < num_sparse; ++i) {
    auto temp_sparse_shapes_desc = op.GetDynamicOutputDesc("sparse_shapes", i);
    (void)FillOpDesc(temp_sparse_shapes_desc, sparse_shapes_shape, DT_INT64);
    op.UpdateDynamicOutputDesc("sparse_shapes", i, temp_sparse_shapes_desc);
  }

  for (int i = 0; i < num_dense; ++i) {
    Shape dense_value_shape(std::vector<int64_t>({dense_shapes[i]}));
    auto dense_values_desc = op.GetDynamicOutputDesc("dense_values", i);
    (void)FillOpDesc(dense_values_desc, dense_value_shape, dense_types[i]);
    op.UpdateDynamicOutputDesc("dense_values", i, dense_values_desc);
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ParseExample, ParseExampleInfer);

IMPLEMT_INFERFUNC(ParseSingleSequenceExample, ParseSingleSequenceExampleInfer) {
  int32_t num_context_sparse;
  if (op.GetAttr("Ncontext_sparse", num_context_sparse) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("failed to get attr[Ncontext_sparse]."));
    return GRAPH_FAILED;
  }

  int32_t num_context_dense;
  if (op.GetAttr("Ncontext_dense", num_context_dense) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("failed to get attr[Ncontext_dense]."));
    return GRAPH_FAILED;
  }

  std::vector<std::vector<int64_t>> feature_list_dense_shapes;
  if (op.GetAttr("feature_list_dense_shapes", feature_list_dense_shapes) !=
      GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("failed to get attr[feature_list_dense_shapes]."));
    return GRAPH_FAILED;
  }

  int32_t num_feature_list_sparse;
  if (op.GetAttr("Nfeature_list_sparse", num_feature_list_sparse) !=
      GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("failed to get attr[Nfeature_list_sparse]."));
    return GRAPH_FAILED;
  }

  int32_t num_feature_list_dense;
  if (op.GetAttr("Nfeature_list_dense", num_feature_list_dense) !=
      GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("failed to get attr[Nfeature_list_dense]."));
    return GRAPH_FAILED;
  }

  std::vector<ge::DataType> context_dense_types;
  if (op.GetAttr("Tcontext_dense", context_dense_types) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("failed to get attr[Tcontext_dense]."));
    return GRAPH_FAILED;
  }

  std::vector<ge::DataType> feature_list_sparse_types;
  if (op.GetAttr("feature_list_sparse_types", feature_list_sparse_types) !=
      GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("failed to get attr[context_sparse_types]."));
    return GRAPH_FAILED;
  }

  std::vector<ge::DataType> feature_list_dense_types;
  if (op.GetAttr("feature_list_dense_types", feature_list_dense_types) !=
      GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("failed to get attr[feature_list_dense_types]."));
    return GRAPH_FAILED;
  }

  std::vector<std::vector<int64_t>> context_dense_shapes;
  if (op.GetAttr("context_dense_shapes", context_dense_shapes) !=
      GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("failed to get attr[context_dense_shapes]."));
    return GRAPH_FAILED;
  }

  std::string err_msg;
  for (const ge::DataType& type : context_dense_types) {
    if (type != DT_INT64 && type != DT_FLOAT && type != DT_STRING) {
      std::string context_dense_dt = DTypeStr(type);
      err_msg = ConcatString("invalid data type[", context_dense_dt,
                             "] of attr[Tcontext_dense], should be "
                             "DT_INT64, DT_FLOAT or DT_STRING");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }

  for (const ge::DataType& type : feature_list_sparse_types) {
    if (type != DT_INT64 && type != DT_FLOAT && type != DT_STRING) {
      std::string feature_list_sparse_dt = DTypeStr(type);
      err_msg = ConcatString("invalid data type[", feature_list_sparse_dt,
                             "] of attr[feature_list_sparse_types], should be "
                             "DT_INT64, DT_FLOAT or DT_STRING");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  GeShape serialized_shape;
  auto serialized_desc = op_desc->MutableInputDesc(0);
  if (WithRank(serialized_desc, 0, serialized_shape, op.GetName().c_str())
      != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(
        0, DebugString(serialized_desc->GetShape().GetDims()), "scalar ");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  int64_t num_examples = serialized_shape.GetDim(0);

  GeShape unused_shape;
  auto unused_desc = op_desc->MutableInputDesc(1);
  if (WithRank(unused_desc, 1, unused_shape, op.GetName().c_str())
      != GRAPH_SUCCESS) {
    err_msg =
        GetShapeErrMsg(1, DebugString(unused_desc->GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  Shape contex_sparse_indices_shape({UNKNOWN_DIM});
  for (int i = 0; i < num_context_sparse; ++i) {
    auto temp_contex_sparse_indices_desc =
        op.GetDynamicOutputDesc("contex_sparse_indices", i);
    (void)FillOpDesc(temp_contex_sparse_indices_desc,
                     contex_sparse_indices_shape, DT_STRING);
    op.UpdateDynamicOutputDesc("contex_sparse_indices", i,
                               temp_contex_sparse_indices_desc);
  }

  Shape context_sparse_values_shape({UNKNOWN_DIM});
  for (int i = 0; i < num_context_sparse; ++i) {
    auto temp_context_sparse_values_desc =
        op.GetDynamicOutputDesc("context_sparse_values", i);
    (void)FillOpDesc(temp_context_sparse_values_desc,
                     context_sparse_values_shape, DT_STRING);
    op.UpdateDynamicOutputDesc("context_sparse_values", i,
                               temp_context_sparse_values_desc);
  }

  Shape context_sparse_shapes_shape({1});
  for (int i = 0; i < num_context_sparse; ++i) {
    auto temp_context_sparse_shapes_desc =
        op.GetDynamicOutputDesc("context_sparse_shapes", i);
    (void)FillOpDesc(temp_context_sparse_shapes_desc,
                     context_sparse_shapes_shape, DT_STRING);
    op.UpdateDynamicOutputDesc("context_sparse_shapes", i,
                               temp_context_sparse_shapes_desc);
  }

  for (int i = 0; i < num_context_dense; ++i) {
    Shape context_dense_shape(std::vector<int64_t>({context_dense_shapes[i]}));
    Shape num_examples_shape({num_examples});
    auto temp_context_dense_values_desc =
        op.GetDynamicOutputDesc("context_dense_values", i);
    (void)FillOpDesc(temp_context_dense_values_desc, context_dense_shape,
                     context_dense_types[i]);
    op.UpdateDynamicOutputDesc("context_dense_values", i,
                               temp_context_dense_values_desc);
  }

  Shape feature_list_sparse_indices_shape({UNKNOWN_DIM, 2});
  for (int i = 0; i < num_feature_list_sparse; ++i) {
    auto temp_feature_list_sparse_indices_desc =
        op.GetDynamicOutputDesc("feature_list_sparse_indices", i);
    (void)FillOpDesc(temp_feature_list_sparse_indices_desc,
                     feature_list_sparse_indices_shape, DT_INT64);
    op.UpdateDynamicOutputDesc("feature_list_sparse_indices", i,
                               temp_feature_list_sparse_indices_desc);
  }

  Shape feature_list_sparse_values_shape({UNKNOWN_DIM});
  for (int i = 0; i < num_feature_list_sparse; ++i) {
    auto temp_feature_list_sparse_values_desc =
        op.GetDynamicOutputDesc("feature_list_sparse_values", i);
    (void)FillOpDesc(temp_feature_list_sparse_values_desc,
                     feature_list_sparse_values_shape,
                     feature_list_sparse_types[i]);
    op.UpdateDynamicOutputDesc("feature_list_sparse_values", i,
                               temp_feature_list_sparse_values_desc);
  }

  Shape feature_list_sparse_shapes_shape({2});
  for (int i = 0; i < num_feature_list_sparse; ++i) {
    auto temp_feature_list_sparse_shapes_desc =
        op.GetDynamicOutputDesc("feature_list_sparse_shapes", i);
    (void)FillOpDesc(temp_feature_list_sparse_shapes_desc,
                     feature_list_sparse_shapes_shape, DT_INT64);
    op.UpdateDynamicOutputDesc("feature_list_sparse_shapes", i,
                               temp_feature_list_sparse_shapes_desc);
  }

  for (int i = 0; i < num_feature_list_dense; ++i) {
    Shape num_examples_shape({UNKNOWN_DIM});
    Shape feature_list_dense_shape(
        std::vector<int64_t>({feature_list_dense_shapes[i]}));
    if (Concatenate(num_examples_shape, feature_list_dense_shape,
                    feature_list_dense_shape) != GRAPH_SUCCESS) {
      ConcatString(
          "failed to call Concatenate function to concatenate num_examples[",
          num_examples, "] and ", i, "th shape[",
          DebugString(feature_list_dense_shapes[i]),
          "] of attr[feature_list_dense_shapes]");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    auto temp_feature_list_dense_values_desc =
        op.GetDynamicOutputDesc("feature_list_dense_values", i);
    (void)FillOpDesc(temp_feature_list_dense_values_desc,
                     feature_list_dense_shape, context_dense_types[i]);
    op.UpdateDynamicOutputDesc("feature_list_dense_values", i,
                               temp_feature_list_dense_values_desc);
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ParseSingleSequenceExample, ParseSingleSequenceExampleInfer);

}  // namespace ge
