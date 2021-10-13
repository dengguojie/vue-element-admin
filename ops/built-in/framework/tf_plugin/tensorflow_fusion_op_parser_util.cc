/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
 * \file tensorflow_fusion_op_parser_util.cpp
 * \brief
 */
#include "tensorflow_fusion_op_parser_util.h"
#include "proto/tensorflow/attr_value.pb.h"
#include "proto/tensorflow/types.pb.h"

#include "op_log.h"

using domi::tensorflow::DataType;

namespace domi {
static const char* const TENSORFLOW_ATTR_TYPE_TENSOR = "tensor";
static const char* const TENSORFLOW_ATTR_VALUE = "value";

#define GET_CONST_VALUE(tensor, param, index, FIELD)                                 \
  do {                                                                               \
    google::protobuf::RepeatedField<FIELD> val_vec;                                  \
    int32_t val_size = 0;                                                            \
    val_vec = tensor.FIELD##_val();                                                  \
    val_size = val_vec.size();                                                       \
    if (index < val_size) {                                                          \
      param = val_vec.Get(index);                                                    \
    } else if (tensor.has_tensor_shape()) {                                          \
      const std::string tensor_content = tensor.tensor_content();                    \
      char* buf = const_cast<char*>(tensor_content.data());                          \
      FIELD* buf_v = reinterpret_cast<FIELD*>(buf);                                  \
      if (static_cast<uint32_t>(index) >= tensor_content.length() / sizeof(FIELD)) { \
        OP_LOGE("Const data size is smaller than index :%d,not supported!", index);  \
        return FAILED;                                                               \
      }                                                                              \
      param = buf_v[index];                                                          \
    } else {                                                                         \
      OP_LOGE("Const data size is smaller than index :%d,not supported!", index);    \
      return PARAM_INVALID;                                                          \
    }                                                                                \
  } while (false)

Status ParseParamFromConst(const domi::tensorflow::NodeDef* node_def, int32_t& param, int index) {
  domi::tensorflow::TensorProto tensor;
  TensorflowFusionOpParserUtil::GetTensorFromNode(node_def, tensor);
  GET_CONST_VALUE(tensor, param, index, int);
  return SUCCESS;
}

Status ParseParamFromConst(const domi::tensorflow::NodeDef* node_def, float& param, int index) {
  domi::tensorflow::TensorProto tensor;
  TensorflowFusionOpParserUtil::GetTensorFromNode(node_def, tensor);
  GET_CONST_VALUE(tensor, param, index, float);
  return SUCCESS;
}

Status TensorflowFusionOpParserUtil::GetTensorFromNode(const domi::tensorflow::NodeDef* node_def,
                                                       domi::tensorflow::TensorProto& tensor) {
  if (node_def == nullptr) {
    return FAILED;
  }

  string node_name = node_def->name();

  domi::tensorflow::AttrValue attr_value;
  // Check that the attribute value must exist and get the value.
  if (!FindAttrValue(node_def, TENSORFLOW_ATTR_VALUE, attr_value)) {
    OP_LOGE("NodeDef %s Attr %s is not exist.", node_name.c_str(), TENSORFLOW_ATTR_VALUE);
    return FAILED;
  }
  // Check that the value attribute must be tensor.
  if (CheckAttrHasType(attr_value, TENSORFLOW_ATTR_TYPE_TENSOR) != SUCCESS) {
    OP_LOGE("check Attr %s failed", TENSORFLOW_ATTR_VALUE);
    return FAILED;
  }
  tensor = attr_value.tensor();
  return SUCCESS;
}

// Convert tensorflow property to ge property
bool TensorflowFusionOpParserUtil::FindAttrValue(const domi::tensorflow::NodeDef* nodeDef, const string& attr_name,
                                                 domi::tensorflow::AttrValue& attr_value) {
  if (nodeDef == nullptr) {
    return FAILED;
  }
  const google::protobuf::Map<std::string, domi::tensorflow::AttrValue>& attr = nodeDef->attr();
  const google::protobuf::Map<std::string, domi::tensorflow::AttrValue>::const_iterator it = attr.find(attr_name);
  if (it != attr.end()) {
    attr_value = it->second;
    return true;
  }
  return false;
}

Status TensorflowFusionOpParserUtil::CheckAttrHasType(const domi::tensorflow::AttrValue& attr_value,
                                                      const string& type) {
  uint32_t num_set = 0;
#define VALIDATE_FIELD(name, type_string, oneof_case)                                                         \
  do {                                                                                                        \
    if (attr_value.has_list()) {                                                                              \
      if (attr_value.list().name##_size() > 0) {                                                              \
        if (type != "list(" type_string ")") {                                                                \
          OP_LOGE("GeAttrValue has value with type 'list(" type_string ")'when '%s' expected", type.c_str()); \
          return FAILED;                                                                                      \
        }                                                                                                     \
        ++num_set;                                                                                            \
      }                                                                                                       \
    } else if (attr_value.value_case() == domi::tensorflow::AttrValue::oneof_case) {                          \
      if (type != (type_string)) {                                                                              \
        OP_LOGE("GeAttrValue has value with type '" type_string "' when '%s' expected", type.c_str());        \
        return FAILED;                                                                                        \
      }                                                                                                       \
      ++num_set;                                                                                              \
    }                                                                                                         \
  } while (false)

  VALIDATE_FIELD(s, "string", kS);
  VALIDATE_FIELD(i, "int", kI);
  VALIDATE_FIELD(f, "float", kF);
  VALIDATE_FIELD(b, "bool", kB);
  VALIDATE_FIELD(type, "type", kType);
  VALIDATE_FIELD(shape, "shape", kShape);
  VALIDATE_FIELD(tensor, "tensor", kTensor);
  VALIDATE_FIELD(func, "func", kFunc);

#undef VALIDATE_FIELD

  if (attr_value.value_case() == domi::tensorflow::AttrValue::kPlaceholder) {
    OP_LOGE("GeAttrValue has value with unexpected type 'placeholder'");
    return FAILED;
  }

  // Okay to have an empty list, but not to be missing a non-list value.
  string str_x = "list(";
  bool bStart = (type.size() >= str_x.size()) && (type.compare(0, str_x.size(), str_x) == 0);
  if ((num_set == 0) && !bStart) {
    OP_LOGE("GeAttrValue missing value with expected type '%s'", type.c_str());
    return FAILED;
  }

  // Ref types and DT_INVALID are illegal, and DataTypes must
  // be a valid enum type.
  if (type == "type") {
    if (!domi::tensorflow::DataType_IsValid(attr_value.type())) {
      OP_LOGE("GeAttrValue has invalid DataType enum: %d", attr_value.type());
      return FAILED;
    }
    if (attr_value.type() == domi::tensorflow::DT_INVALID) {
      OP_LOGE("GeAttrValue has invalid DataType");
      return FAILED;
    }
  } else if (type == "list(type)") {
    for (auto& as_int : attr_value.list().type()) {
      const DataType dtype = static_cast<DataType>(as_int);
      if (!domi::tensorflow::DataType_IsValid(dtype)) {
        OP_LOGE("GeAttrValue has invalid DataType enum: %d", as_int);
        return FAILED;
      }
      if (dtype == domi::tensorflow::DT_INVALID) {
        OP_LOGE("GeAttrValue contains invalid DataType");
        return FAILED;
      }
    }
  }

  return SUCCESS;
}
}  // namespace domi
