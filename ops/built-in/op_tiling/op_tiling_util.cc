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
 * \file op_tiling_util.cc
 * \brief tiling function of op
 */

#include <functional>
#include <ge_error_codes.h>
#include <graph/utils/type_utils.h>
#include "graph/utils/op_desc_utils.h"
#include "op_tiling_util.h"

namespace {
constexpr int32_t DATA_BLOCK_4 = 4;
constexpr int32_t DATA_BLOCK_8 = 8;
constexpr int32_t DATA_BLOCK_16 = 16;
constexpr int32_t DATA_BLOCK_32 = 32;
}  // namespace

namespace optiling {
using namespace std;

/*
 * @brief: get datatype string from enum
 * @param [in] type: enum datatype
 * @return string: datatype string
 */
std::string to_string(const ge::DataType& type) {
  return ge::TypeUtils::DataTypeToSerialString(type);
}

/*
 * @brief: get format string from enum
 * @param [in] format: enum format
 * @return string: format string
 */
std::string to_string(const ge::Format& format) {
  return ge::TypeUtils::FormatToSerialString(format);
}

int64_t GetTensorSize(const GeShape& shape) {
  int64_t shapeNum = 1;
  if (!shape.IsScalar()) {
    shapeNum = shape.GetShapeSize();
  }
  return shapeNum;
}

int64_t GetByteLenByString(const std::string& op_type) {
  auto find_it = STR_TO_DATATYPE.find(op_type);
  if (find_it != STR_TO_DATATYPE.end()) {
    return GetSizeByDataType(find_it->second);
  }
  OP_LOGW("GetByteLen", "con not get the dtype[%s] in ge::DataType list. will return 0", op_type.c_str());
  return 0;
}

vector<vector<int64_t>> GetInputShapes(const ge::Operator& paras) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(paras);
  if (op_desc == nullptr)
    return {};

  vector<vector<int64_t>> shapes;
  int count = op_desc->GetInputsSize();
  for (int i = 0; i < count; i++) {
    auto ptr = op_desc->MutableInputDesc(i);
    shapes.emplace_back(ptr->MutableShape().GetDims());
  }

  return shapes;
}

int64_t GetDataBlockElems(const ge::DataType& dtype) {
  int64_t dataBlock = 0;
  if (dtype == DT_FLOAT || dtype == DT_INT32 || dtype == DT_UINT32) {
    dataBlock = DATA_BLOCK_8;
  } else if (dtype == DT_FLOAT16 || dtype == DT_INT16 || dtype == DT_UINT16) {
    dataBlock = DATA_BLOCK_16;
  } else if (dtype == DT_INT8 || dtype == DT_UINT8) {
    dataBlock = DATA_BLOCK_32;
  } else if (dtype == DT_INT64 || dtype == DT_UINT64) {
    dataBlock = DATA_BLOCK_4;
  }
  return dataBlock;
}

static bool TransJsonToVector(const ge::Operator& op, const nlohmann::json& compile_info_json,
                              const std::vector<std::string>& compile_info_key,
                              const std::map<std::string, int64_t>& optional_key,
                              std::vector<int64_t>& compile_info_vec) {
  using namespace nlohmann;
  const nlohmann::json& all_vars = compile_info_json["vars"];
  compile_info_vec.resize(compile_info_key.size(), 0);
  for (size_t i = 0; i < compile_info_key.size(); i++) {
    auto it = optional_key.find(compile_info_key[i]);
    if (it == optional_key.end()) {
      OP_TILING_CHECK(!GetCompileValue(all_vars, compile_info_key[i], compile_info_vec[i]),
                      VECTOR_INNER_ERR_REPORT_TILIING(TbeGetOpType(op).c_str(), "GetCompileParams, get %s error",
                                                      compile_info_key[i].c_str()),
                      return false);
    } else {
      const int64_t default_value = it->second;
      GetCompileValue(all_vars, compile_info_key[i], compile_info_vec[i], default_value);
    }
    OP_LOGD(TbeGetOpType(op).c_str(), "TransJsonToVector key:value = %s:%ld", compile_info_key[i].c_str(),
            compile_info_vec[i]);
  }

  OP_LOGD(TbeGetOpType(op).c_str(), "TransJsonToVector end");
  return true;
}

void* ParseCompileToInt64Vec(const ge::Operator& op, const ge::AscendString compile_info,
                             const std::vector<std::string>& compile_info_key,
                             const std::map<std::string, int64_t>& optional_key) {
  auto json_object = std::make_shared<nlohmann::json>(nlohmann::json::parse(compile_info.GetString()));
  std::vector<int64_t>* parsed_vector_ptr = new std::vector<int64_t>(compile_info_key.size(), 0);
  bool bsucc = TransJsonToVector(op, *json_object, compile_info_key, optional_key, *parsed_vector_ptr);
  OP_TILING_CHECK(!bsucc, delete parsed_vector_ptr, return nullptr);
  return static_cast<void*>(parsed_vector_ptr);
}

bool ParseCompileToInt64Vec(const ge::Operator& op, const ge::AscendString compile_info,
                            const std::vector<std::string>& compile_info_key,
                            const std::map<std::string, int64_t>& optional_key, std::vector<int64_t>& compile_vec) {
  std::shared_ptr<nlohmann::json> json_object =
      ops::make_shared_nothrow<nlohmann::json>(nlohmann::json::parse(compile_info.GetString()));
  OP_TILING_CHECK(json_object == nullptr, OP_LOGW(TbeGetOpType(op), "Parse the compile info failed, will return false"),
                  return false);
  return TransJsonToVector(op, *json_object, compile_info_key, optional_key, compile_vec);
}
}  // namespace optiling
