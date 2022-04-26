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
 * \file lookup_ops.cpp
 * \brief
 */
#include "inc/lookup_ops.h"
#include "graph/operator.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "util/error_util.h"
#include "util/common_shape_fns.h"
#include "util/lookup_ops_shape_fns.h"

namespace ge {

IMPLEMT_INFERFUNC(LookupTableFind, LookupTableFindInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);


  auto handle_desc = op_desc->MutableInputDesc(0);
  GeShape handle_shape;
  if (WithRank(handle_desc, 0, handle_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(
        0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "scalar");
    err_msg = std::string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto default_value_desc = op_desc->MutableInputDesc(2);
  GeShape default_value_shape;
  // Default value must be scalar or vector.
  if (WithRankAtMost(default_value_desc, 1, default_value_shape,
                     TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(
        2, DebugString(default_value_desc->GetShape().GetDims()), "scalar or 1D");
    err_msg = std::string("failed to call WithRankAtMost, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto keys_desc = op_desc->MutableInputDesc(1);
  DataType Tin = keys_desc->GetDataType();
  DataType Tout = DT_FLOAT;
  if (op.GetAttr("Tout", Tout) != GRAPH_SUCCESS) {
    OP_LOGW(TbeGetName(op).c_str(),
            "get attr[Tout] failed, use default type DT_FLOAT");
  }

  ShapeAndType output_shape_and_type;
  // ShapeAndType only support old version tensor shape
  Shape keys_shape = op.GetInputDesc(1).GetShape();
  if (ValidateTableResourceHandle(op, keys_shape, Tin, Tout, true,
                                  output_shape_and_type,
                                  TbeGetName(op).c_str()) == GRAPH_FAILED) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(
        TbeGetName(op),
        std::string("failed to call ValidateTableResourceHandle"));
    return GRAPH_FAILED;
  }
  GeShape values_shape(output_shape_and_type.GetShape().GetDims());

  auto values_desc = op_desc->MutableOutputDesc(0);
  (void)FillOpDesc(values_desc, values_shape, Tout);
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(LookupTableFind, LookupTableFindInfer);

IMPLEMT_INFERFUNC(LookupTableExport, LookupTableExportInfer) {
  Shape shape;
  if (WithRank(op.GetInputDesc(0), 0, shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(
        0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "scalar");
    err_msg = std::string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  Shape keys = UnknownShapeOfRank(1);
  auto p_context = op.GetInferenceContext();
  const std::vector<std::vector<ShapeAndType>> &value_shape_and_types =
      p_context->GetInputHandleShapesAndTypes();
  Shape output_values_shape(UNKNOWN_SHAPE);
  if (value_shape_and_types.size() != 0) {
    if (value_shape_and_types[0].size() != 2) {
      std::string err_msg = ConcatString(
          "invalid size of value and type context for op, should be 2, got ",
          value_shape_and_types[0].size());
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }

    std::vector<ShapeAndType> handle_data;
    handle_data.emplace_back(value_shape_and_types[0][0]);
    handle_data.emplace_back(value_shape_and_types[0][1]);

    ShapeAndType output_shape_and_type;
    if (ValidateTableResourceHandle(shape, handle_data, output_shape_and_type,
                                    false,
                                    TbeGetName(op).c_str()) == GRAPH_FAILED) {
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(
          TbeGetName(op),
          std::string("failed to call ValidateTableResourceHandle"));
      return GRAPH_FAILED;
    }
    output_values_shape = output_shape_and_type.GetShape();
  }

  ge::DataType Tkeys;
  if (op.GetAttr("Tkeys", Tkeys) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       std::string("get attr[Tkeys] failed"));
    return GRAPH_PARAM_INVALID;
  }
  ge::DataType Tvalues;
  if (op.GetAttr("Tvalues", Tvalues) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       std::string("get attr[Tvalues] failed"));
    return GRAPH_PARAM_INVALID;
  }

  TensorDesc y_desc = op.GetOutputDesc("keys");
  y_desc.SetShape(keys);
  y_desc.SetDataType(Tkeys);
  op.UpdateOutputDesc("keys", y_desc);

  y_desc = op.GetOutputDesc("values");
  y_desc.SetShape(output_values_shape);
  y_desc.SetDataType(Tvalues);
  op.UpdateOutputDesc("values", y_desc);
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(LookupTableExport, LookupTableExportInfer);

IMPLEMT_INFERFUNC(LookupTableImport, LookupTableImportInfer) {
  Shape shape;
  if (WithRank(op.GetInputDesc(0), 0, shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(
        0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "scalar");
    err_msg = std::string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  Shape keys;
  if (WithRank(op.GetInputDesc(1), 1, keys, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(
        1, DebugString(op.GetInputDesc(1).GetShape().GetDims()), "1D");
    err_msg = std::string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  Shape values_shape = op.GetInputDesc(2).GetShape();
  if (Merge(keys, values_shape, keys, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg =
        ConcatString("failed to call Merge, can not merge input[1] shape",
                     DebugString(keys.GetDims()), " and input[2] shape",
                     DebugString(values_shape.GetDims()));
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_PARAM_INVALID;
  }

  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(LookupTableImport, LookupTableImportInfer);

IMPLEMT_INFERFUNC(LookupTableInsert, LookupTableInsertInfer) {
  Shape shape;
  if (WithRank(op.GetInputDesc(0), 0, shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(
        0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "scalar");
    err_msg = std::string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(LookupTableInsert, LookupTableInsertInfer);

IMPLEMT_INFERFUNC(LookupTableSize, LookupTableSizeInfer) {
  Shape shape;
  (void)Scalar(shape);
  TensorDesc output_desc = op.GetOutputDesc("size");
  output_desc.SetShape(shape);
  output_desc.SetDataType(DT_INT64);
  op.UpdateOutputDesc("size", output_desc);
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(LookupTableSize, LookupTableSizeInfer);

IMPLEMT_INFERFUNC(HashTable, HashTableInfer) {
  DataType key_type;
  DataType value_type;
  if (op.GetAttr("key_dtype", key_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), std::string("get attr[key_dtype] failed"));
    return GRAPH_PARAM_INVALID;
  }
  if (op.GetAttr("value_dtype", value_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), std::string("get attr[value_dtype] failed"));
    return GRAPH_PARAM_INVALID;
  }
  if (key_type == DT_INT64) {
    if (!((value_type == DT_INT64) || (value_type == DT_INT32) ||
          (value_type == DT_FLOAT) || (value_type == DT_DOUBLE) ||
          (value_type == DT_STRING))) {
      std::string err_msg = ConcatString(
          "when attr[key_type] is DT_INT64, attr[value_type] should in ["
          "DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE, DT_STRING], but got [",
          TypeUtils::DataTypeToSerialString(value_type), "]");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_PARAM_INVALID;
    }
  }
  if (key_type == DT_INT32) {
    if (!((value_type == DT_INT32) || (value_type == DT_FLOAT) ||
          (value_type == DT_DOUBLE) || (value_type == DT_STRING))) {
      std::string err_msg = ConcatString(
          "when attr[key_type] is DT_INT32, attr[value_type] should in "
          "[DT_INT32, DT_FLOAT, DT_DOUBLE, DT_STRING], but got [",
          TypeUtils::DataTypeToSerialString(value_type), "]");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_PARAM_INVALID;
    }
  }
  if (key_type == DT_STRING) {
    if (!((value_type == DT_BOOL) || (value_type == DT_INT32) ||
          (value_type == DT_INT64) || (value_type == DT_FLOAT) ||
          (value_type == DT_DOUBLE))) {
      std::string err_msg = ConcatString(
          "when attr[key_type] is DT_STRING, attr[value_type] should in "
          "[DT_BOOL, DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE], but got [",
          TypeUtils::DataTypeToSerialString(value_type), "]");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_PARAM_INVALID;
    }
  }
  Shape scalar_shape;
  (void)Scalar(scalar_shape);
  TensorDesc handle_desc = op.GetOutputDesc("handle");
  handle_desc.SetShape(scalar_shape);
  handle_desc.SetDataType(DT_RESOURCE);
  if (op.UpdateOutputDesc("handle", handle_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op),
        std::string("update description for output[handle] failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(HashTable, HashTableInfer);

IMPLEMT_INFERFUNC(InitializeTable, InitializeTableInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc(0), 0, unused_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(
        0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "scalar");
    err_msg = std::string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  Shape key_shape;
  if (WithRank(op.GetInputDesc(1), 1, key_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(
        1, DebugString(op.GetInputDesc(1).GetShape().GetDims()), "1D");
    err_msg = std::string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  Shape value_shape = op.GetInputDesc(2).GetShape();
  if ((Merge(key_shape, value_shape, unused_shape, TbeGetName(op).c_str())) !=
      GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call Merge, the shape", DebugString(key_shape.GetDims()),
        " of input[1] must same with shape", DebugString(value_shape.GetDims()),
        " of input[2]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(InitializeTable, InitializeTableInfer);

IMPLEMT_INFERFUNC(MutableDenseHashTable, MutableDenseHashTableInfer) {
  std::vector<int64_t> value_p;
  if (op.GetAttr("value_shape", value_p) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  Shape value_s(std::move(value_p));

  TensorDesc desc_handle = op.GetOutputDesc("handle");
  desc_handle.SetShape(Shape());
  desc_handle.SetDataType(DT_RESOURCE);

  ge::DataType key_t;
  ge::DataType value_t;
  key_t = op.GetInputDesc(0).GetDataType();

  if (op.GetAttr("value_dtype", value_t) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
      TbeGetName(op), string("failed to get attr[value_dtype]"));
    return GRAPH_FAILED;
  }

  if (key_t == DT_INT64) {
    if (!((value_t == DT_BOOL) || (value_t == DT_INT64) || (value_t == DT_INT32) || (value_t == DT_FLOAT) ||
          (value_t == DT_DOUBLE))) {
      std::string err_msg = ConcatString(
        "when attr[key_type] is DT_INT64, attr[value_type] should in ["
        "DT_INT64, DT_INT32, DT_FLOAT, DT_BOOL], but got [",
      TypeUtils::DataTypeToSerialString(value_t), "]");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  } else if (key_t == DT_INT32) {
    if (!((value_t == DT_INT32) || (value_t == DT_FLOAT) || (value_t == DT_DOUBLE))) {
      std::string err_msg = ConcatString(
        "when attr[key_type] is DT_INT64, attr[value_type] should in ["
        "DT_INT64, DT_INT32, DT_FLOAT], but got [",
      TypeUtils::DataTypeToSerialString(value_t), "]");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }

  auto empty_key = op.GetInputDesc("empty_key");

  std::vector<std::vector<ShapeAndType>> key_value_vec;
  std::vector<ShapeAndType> key_value;
  ShapeAndType key(empty_key.GetShape(), key_t);
  ShapeAndType value(value_s, value_t);
  key_value.emplace_back(key);
  key_value.emplace_back(value);
  key_value_vec.emplace_back(key_value);

  auto pcontext = op.GetInferenceContext();
  pcontext->SetOutputHandleShapesAndTypes(std::move(key_value_vec));
  if (op.UpdateOutputDesc("handle", desc_handle) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
      TbeGetName(op), string("fail to update output[handle] desc."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MutableDenseHashTable, MutableDenseHashTableInfer);

IMPLEMT_INFERFUNC(MutableHashTableOfTensors, MutableHashTableOfTensorsInfer) {
  std::vector<int64_t> value_p;
  if (op.GetAttr("value_shape", value_p) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  Shape value_s(std::move(value_p));

  TensorDesc desc_handle = op.GetOutputDesc("handle");
  desc_handle.SetShape(Shape());
  desc_handle.SetDataType(DT_RESOURCE);

  ge::DataType key_t;
  ge::DataType value_t;

  if (op.GetAttr("key_dtype", key_t) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
      TbeGetName(op), string("failed to get attr[key_dtype]"));
    return GRAPH_FAILED;
  }
  if (op.GetAttr("value_dtype", value_t) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  if (key_t == DT_INT64) {
    if (!((value_t == DT_INT64) || (value_t == DT_INT32) || (value_t == DT_FLOAT) || (value_t == DT_DOUBLE))) {
      std::string err_msg = ConcatString(
        "when attr[key_type] is DT_INT64, attr[value_type] should in ["
        "DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE], but got [",
        TypeUtils::DataTypeToSerialString(value_t), "]");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  } else if (key_t == DT_INT32) {
    if (!((value_t == DT_INT32) || (value_t == DT_FLOAT) || (value_t == DT_DOUBLE))) {
      std::string err_msg = ConcatString(
        "when attr[key_type] is DT_INT64, attr[value_type] should in ["
        "DT_INT32, DT_FLOAT, DT_DOUBLE], but got [",
        TypeUtils::DataTypeToSerialString(value_t), "]");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }
  std::vector<std::vector<ShapeAndType>> key_value_vec;
  std::vector<ShapeAndType> key_value;
  ShapeAndType key(Shape(), key_t);
  ShapeAndType value(value_s, value_t);
  key_value.emplace_back(key);
  key_value.emplace_back(value);
  key_value_vec.emplace_back(key_value);
  auto pcontext = op.GetInferenceContext();
  pcontext->SetOutputHandleShapesAndTypes(std::move(key_value_vec));

  if (op.UpdateOutputDesc("handle", desc_handle) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
      TbeGetName(op), string("fail to update output[handle] desc."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MutableHashTableOfTensors, MutableHashTableOfTensorsInfer);

IMPLEMT_INFERFUNC(MutableHashTable, MutableHashTableInfer) {
  TensorDesc desc_handle = op.GetOutputDesc("handle");
  desc_handle.SetShape(Shape());
  desc_handle.SetDataType(DT_RESOURCE);

  ge::DataType key_t;
  ge::DataType value_t;

  if (op.GetAttr("key_dtype", key_t) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
      TbeGetName(op), string("failed to get attr[key_dtype]"));
    return GRAPH_FAILED;
  }
  if (op.GetAttr("value_dtype", value_t) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  if (key_t == DT_INT64) {
    if (!((value_t == DT_INT64) || (value_t == DT_INT32) || (value_t == DT_FLOAT) || (value_t == DT_DOUBLE))) {
      std::string err_msg = ConcatString(
        "when attr[key_type] is DT_INT64, attr[value_type] should in ["
        "DT_INT64, DT_INT32, DT_FLOAT, DT_DOUBLE], but got [",
      TypeUtils::DataTypeToSerialString(value_t), "]");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  } else if (key_t == DT_INT32) {
    if (!((value_t == DT_INT32) || (value_t == DT_FLOAT) || (value_t == DT_DOUBLE))) {
      std::string err_msg = ConcatString(
        "when attr[key_type] is DT_INT64, attr[value_type] should in ["
        "DT_INT32, DT_FLOAT, DT_DOUBLE], but got [",
      TypeUtils::DataTypeToSerialString(value_t), "]");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }
  std::vector<std::vector<ShapeAndType>> key_value_vec;
  std::vector<ShapeAndType> key_value;
  ShapeAndType key(Shape(), key_t);
  ShapeAndType value(Shape(), value_t);
  key_value.emplace_back(key);
  key_value.emplace_back(value);
  key_value_vec.emplace_back(key_value);
  auto pcontext = op.GetInferenceContext();
  pcontext->SetOutputHandleShapesAndTypes(std::move(key_value_vec));

  if (op.UpdateOutputDesc("handle", desc_handle) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MutableHashTable, MutableHashTableInfer);

}  // namespace ge
