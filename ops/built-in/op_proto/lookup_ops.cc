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
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "util/lookup_ops_shape_fns.h"

namespace ge {
IMPLEMT_INFERFUNC(LookupTableFind, LookupTableFindInfer) {
  Shape shape;
  if (WithRank(op.GetInputDesc(0), 0, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input handle rank must be 0");
    return GRAPH_FAILED;
  }

  Shape shape_default_value;
  if (WithRankAtMost(op.GetInputDesc(2), 0, shape_default_value, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input default_value rank must be 0");
    return GRAPH_FAILED;
  }

  auto p_context = op.GetInferenceContext();
  const std::vector<std::vector<ShapeAndType>>& value_shape_and_types = p_context->GetInputHandleShapesAndTypes();

  Shape output_shape(UNKNOWN_SHAPE);
  if (value_shape_and_types.size() != 0) {
    if (value_shape_and_types[0].size() != 2) {
      OP_LOGE(op.GetName().c_str(), "value_shape_and_types[0].size's rank must be 2");
      return GRAPH_FAILED;
    }

    std::vector<ShapeAndType> handle_data;
    handle_data.emplace_back(value_shape_and_types[0][0]);
    handle_data.emplace_back(value_shape_and_types[0][1]);

    ShapeAndType output_shape_and_type;
    if (ValidateTableResourceHandle(shape, handle_data, output_shape_and_type, true, op.GetName().c_str()) ==
        GRAPH_FAILED) {
      return GRAPH_FAILED;
    }
    output_shape = output_shape_and_type.GetShape();
  }

  ge::DataType Tout;
  if (op.GetAttr("Tout", Tout) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get Attr dtypes error.");
  }

  TensorDesc y_desc = op.GetOutputDesc("values");
  y_desc.SetShape(output_shape);
  y_desc.SetDataType(Tout);
  op.UpdateOutputDesc("values", y_desc);

  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(LookupTableFind, LookupTableFindInfer);

IMPLEMT_INFERFUNC(LookupTableExport, LookupTableExportInfer) {
  Shape shape;
  if (WithRank(op.GetInputDesc(0), 0, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input handle rank must be 0");
    return GRAPH_FAILED;
  }
  Shape keys = UnknownShapeOfRank(1);
  auto p_context = op.GetInferenceContext();
  const std::vector<std::vector<ShapeAndType>>& value_shape_and_types = p_context->GetInputHandleShapesAndTypes();
  Shape output_values_shape(UNKNOWN_SHAPE);
  if (value_shape_and_types.size() != 0) {
    if (value_shape_and_types[0].size() != 2) {
      return GRAPH_FAILED;
    }

    std::vector<ShapeAndType> handle_data;
    handle_data.emplace_back(value_shape_and_types[0][0]);
    handle_data.emplace_back(value_shape_and_types[0][1]);

    ShapeAndType output_shape_and_type;
    if (ValidateTableResourceHandle(shape, handle_data, output_shape_and_type, false, op.GetName().c_str()) ==
        GRAPH_FAILED) {
      return GRAPH_FAILED;
    }
    output_values_shape = output_shape_and_type.GetShape();
  }

  ge::DataType Tkeys;
  if (op.GetAttr("Tkeys", Tkeys) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get attr Tkeys failed.");
  }
  ge::DataType Tvalues;
  if (op.GetAttr("Tvalues", Tvalues) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get attr Tvalues failed.");
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
  if (WithRank(op.GetInputDesc(0), 0, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input handle rank must be 0");
    return GRAPH_FAILED;
  }

  Shape keys;
  if (WithRank(op.GetInputDesc(1), 1, keys, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input keys rank must be 1");
    return GRAPH_FAILED;
  }

  Shape values_shape = op.GetInputDesc(2).GetShape();
  if (Merge(keys, values_shape, keys, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "keys values_shape can not merge.");
    return GRAPH_PARAM_INVALID;
  }

  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(LookupTableImport, LookupTableImportInfer);

IMPLEMT_INFERFUNC(LookupTableInsert, LookupTableInsertInfer) {
  Shape shape;
  if (WithRank(op.GetInputDesc(0), 0, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input handle rank must be 0");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(LookupTableInsert, LookupTableInsertInfer);

IMPLEMT_INFERFUNC(LookupTableSize, LookupTableSizeInfer) {
  Shape shape;
  if (Scalar(shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "create Scalar fail");
    return GRAPH_FAILED;
  }

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
    OP_LOGE(op.GetName().c_str(), "get key type tesnor fail!");
    return GRAPH_PARAM_INVALID;
  }
  if (op.GetAttr("value_dtype", value_type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get value type tesnor fail!");
    return GRAPH_PARAM_INVALID;
  }
  if (key_type == DT_INT64) {
    if (!((value_type == DT_INT64) || (value_type == DT_INT32) || (value_type == DT_FLOAT) ||
          (value_type == DT_DOUBLE))) {
      OP_LOGE(op.GetName().c_str(), "value_type illegal when key_type is DT_INT64");
      return GRAPH_PARAM_INVALID;
    }
  }
  if (key_type == DT_INT32) {
    if (!((value_type == DT_INT32) || (value_type == DT_FLOAT) || (value_type == DT_DOUBLE))) {
      OP_LOGE(op.GetName().c_str(), "value_type illegal when key_type is DT_INT32");
      return GRAPH_PARAM_INVALID;
    }
  }
  Shape scalar_shape;
  (void)Scalar(scalar_shape);
  TensorDesc handle_desc = op.GetOutputDesc("handle");
  handle_desc.SetShape(scalar_shape);
  handle_desc.SetDataType(DT_RESOURCE);
  if (op.UpdateOutputDesc("handle", handle_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update handle desc failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(HashTable, HashTableInfer);

IMPLEMT_INFERFUNC(InitializeTable, InitializeTableInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc(0), 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input handle must be 0-D");
    return GRAPH_FAILED;
  }
  Shape key_shape;
  if (WithRank(op.GetInputDesc(1), 1, key_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input keys must be 1-D");
    return GRAPH_FAILED;
  }
  Shape value_shape = op.GetInputDesc(2).GetShape();
  if ((Merge(key_shape, value_shape, unused_shape, op.GetName().c_str())) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "shape of keys must same with shape of values !");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(InitializeTable, InitializeTableInfer);

IMPLEMT_INFERFUNC(MutableDenseHashTable, MutableDenseHashTableInfer) {
  std::vector<int64_t> value_p;
  if (op.GetAttr("value_shape", value_p) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "getOpAttr value_shape failed!");
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
    OP_LOGE(op.GetName().c_str(), "getOpAttr value_dtype failed!");
    return GRAPH_FAILED;
  }

  if (key_t == DT_INT64) {
    if (!((value_t == DT_BOOL) || (value_t == DT_INT64) || (value_t == DT_INT32) || (value_t == DT_FLOAT) ||
          (value_t == DT_DOUBLE))) {
      OP_LOGE(op.GetName().c_str(), "valueType illegal with keyType is DT_INT64");
      return GRAPH_FAILED;
    }
  } else if (key_t == DT_INT32) {
    if (!((value_t == DT_INT32) || (value_t == DT_FLOAT) || (value_t == DT_DOUBLE))) {
      OP_LOGE(op.GetName().c_str(), "valueType illegal with keyType is DT_INT32");
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
    OP_LOGE(op.GetName().c_str(), "update handle desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MutableDenseHashTable, MutableDenseHashTableInfer);

IMPLEMT_INFERFUNC(MutableHashTableOfTensors, MutableHashTableOfTensorsInfer) {
  std::vector<int64_t> value_p;
  if (op.GetAttr("value_shape", value_p) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "getOpAttr value_shape failed!");
    return GRAPH_FAILED;
  }

  Shape value_s(std::move(value_p));

  TensorDesc desc_handle = op.GetOutputDesc("handle");
  desc_handle.SetShape(Shape());
  desc_handle.SetDataType(DT_RESOURCE);

  ge::DataType key_t;
  ge::DataType value_t;

  if (op.GetAttr("key_dtype", key_t) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "getOpAttr key_dtype failed!");
    return GRAPH_FAILED;
  }
  if (op.GetAttr("value_dtype", value_t) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "getOpAttr value_dtype failed!");
    return GRAPH_FAILED;
  }
  if (key_t == DT_INT64) {
    if (!((value_t == DT_INT64) || (value_t == DT_INT32) || (value_t == DT_FLOAT) || (value_t == DT_DOUBLE))) {
      OP_LOGE(op.GetName().c_str(), "valueType illegal with keyType is DT_INT64");
      return GRAPH_FAILED;
    }
  } else if (key_t == DT_INT32) {
    if (!((value_t == DT_INT32) || (value_t == DT_FLOAT) || (value_t == DT_DOUBLE))) {
      OP_LOGE(op.GetName().c_str(), "valueType illegal with keyType is DT_INT32");
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
    OP_LOGE(op.GetName().c_str(), "update handle desc failed.");
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
    OP_LOGE(op.GetName().c_str(), "getOpAttr key_dtype failed!");
    return GRAPH_FAILED;
  }
  if (op.GetAttr("value_dtype", value_t) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "getOpAttr value_dtype failed!");
    return GRAPH_FAILED;
  }
  if (key_t == DT_INT64) {
    if (!((value_t == DT_INT64) || (value_t == DT_INT32) || (value_t == DT_FLOAT) || (value_t == DT_DOUBLE))) {
      OP_LOGE(op.GetName().c_str(), "valueType illegal with keyType is DT_INT64");
      return GRAPH_FAILED;
    }
  } else if (key_t == DT_INT32) {
    if (!((value_t == DT_INT32) || (value_t == DT_FLOAT) || (value_t == DT_DOUBLE))) {
      OP_LOGE(op.GetName().c_str(), "valueType illegal with keyType is DT_INT32");
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
    OP_LOGE(op.GetName().c_str(), "update handle desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MutableHashTable, MutableHashTableInfer);

}  // namespace ge
