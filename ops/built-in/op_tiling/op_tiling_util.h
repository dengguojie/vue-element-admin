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
 * \file op_tiling_util.h
 * \brief
 */

#ifndef CANN_OPS_BUILT_IN_OP_TILING_OP_TILING_UTIL_H_
#define CANN_OPS_BUILT_IN_OP_TILING_OP_TILING_UTIL_H_

#include <vector>
#include <nlohmann/json.hpp>
#include "error_log.h"
#include "op_tiling.h"
#include "op_attr.h"
#include "op_const.h"
#include "external/graph/operator.h"
#include "graph/utils/op_desc_utils.h"
#include "vector_tiling_profiling.h"

#define REGISTER_OP_TILING_V3_WITH_VECTOR(optype, opfunc, vector_key, optional_key)                                \
  bool Tbe##optype##TilingV3WithVec(const ge::Operator& para, const void* op_info_void,                            \
                                    optiling::utils::OpRunInfo& rinfo) {                                           \
    OP_TILING_CHECK(op_info_void == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(#optype, "op_info_void is nullptr."), \
                    return false);                                                                                 \
    return opfunc(#optype, para, *(const std::vector<int64_t>*)op_info_void, rinfo);                               \
  }                                                                                                                \
  void* Tbe##optype##TilingV3WithVecParsefunc(const ge::Operator& para, const ge::AscendString& compile_info) {    \
    return ParseCompileToInt64Vec(para, compile_info, vector_key, optional_key);                                   \
  }                                                                                                                \
  REGISTER_OP_TILING_V3(optype, Tbe##optype##TilingV3WithVec, Tbe##optype##TilingV3WithVecParsefunc)

#define REGISTER_OP_TILING_V3_CUSTOM(optype, opfunc, parse_func, struct_name)                                         \
  bool Tbe##optype##TilingV3Custom(const ge::Operator& para, const void* op_info_void,                                \
                                   optiling::utils::OpRunInfo& rinfo) {                                               \
    OP_TILING_CHECK(op_info_void == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(#optype, "op_info_void is nullptr."),    \
                    return false);                                                                                    \
    return opfunc(#optype, para, *static_cast<const struct_name*>(op_info_void), rinfo);                              \
  }                                                                                                                   \
  void* Tbe##optype##TilingV3CustomParsefunc(const ge::Operator& para, const ge::AscendString& compile_info) {        \
    std::shared_ptr<nlohmann::json> json_object(new nlohmann::json(nlohmann::json::parse(compile_info.GetString()))); \
    if (json_object == nullptr) {                                                                                     \
      return nullptr;                                                                                                 \
    }                                                                                                                 \
    struct_name* parsed_void_ptr = new (struct_name)();                                                               \
    bool parse_ret = parse_func(#optype, *json_object, *parsed_void_ptr);                                             \
    if (parse_ret) {                                                                                                  \
      return static_cast<void*>(parsed_void_ptr);                                                                     \
    }                                                                                                                 \
    delete parsed_void_ptr;                                                                                           \
    return nullptr;                                                                                                   \
  }                                                                                                                   \
  REGISTER_OP_TILING_V3(optype, Tbe##optype##TilingV3Custom, Tbe##optype##TilingV3CustomParsefunc)

namespace optiling {
using optiling::ByteBuffer;
using namespace ge;

const std::string PATTERN_REDUCE = "CommReduce";
const std::string PATTERN_ELEMWISE = "ElemWise";
const std::string PATTERN_BROADCAST = "Broadcast";
const std::string PATTERN_NORM = "Norm";
const std::string PATTERN_TRANSPOSE = "Transpose";

const std::map<std::string, std::int64_t> NO_OPTIONAL_VALUE;
const std::map<std::string, DataType> STR_TO_DATATYPE = {{"float", DT_FLOAT},
                                                         {"float32", DT_FLOAT},
                                                         {"float16", DT_FLOAT16},
                                                         {"int8", DT_INT8},
                                                         {"int16", DT_INT16},
                                                         {"int32", DT_INT32},
                                                         {"int64", DT_INT64},
                                                         {"uint8", DT_UINT8},
                                                         {"uint16", DT_UINT16},
                                                         {"uint32", DT_UINT32},
                                                         {"uint64", DT_UINT64},
                                                         {"bool", DT_BOOL},
                                                         {"double", DT_DOUBLE},
                                                         {"dual", DT_DUAL},
                                                         {"dual_sub_int8", DT_DUAL_SUB_INT8},
                                                         {"dual_sub_uint8", DT_DUAL_SUB_UINT8},
                                                         {"int4", DT_INT4},
                                                         {"bfloat16", DT_BF16}};

/*
 * @brief: read input shapes from paras
 * @param [in] paras: ge::Operator
 * @return vector<vector<int64_t>>: shapes vector of inputs
 */
vector<vector<int64_t>> GetInputShapes(const ge::Operator& paras);

/*
 * @brief: get datatype string from enum
 * @param [in] type: enum datatype
 * @return string: datatype string
 */

std::string to_string(const ge::DataType& type);
/*
 * @brief: get format string from enum
 * @param [in] format: enum format
 * @return string: format string
 */
std::string to_string(const ge::Format& format);
/*
 * @brief: if shape is empty set {1}
 * @param [in] shape: std::vector<int64_t>
 * @return : void
 */
inline void ScalarToShape(std::vector<int64_t>& shape) {
  if (shape.empty())
    shape.push_back(1);
}

template <typename T>
string to_string(const ByteBuffer& tiling_data) {
  auto data = tiling_data.str();
  string result = "(";
  const T* data_addr = reinterpret_cast<const T*>(data.c_str());
  for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
    result += std::to_string(*data_addr);
    data_addr++;
    result += ",";
  }
  result += ")";

  return result;
}

/*
 * @brief: get Byte size base on dtype(string)
 * @param [in] op_type: string dtype
 * @return int64_t: byte len
 */
int64_t GetByteLenByString(const std::string& op_type);

/*
 * @brief: get data block elements
 * @param [in] dtype: ge DataType
 * @return Int: dataBlock;
 */
int64_t GetDataBlockElems(const ge::DataType& dtype);

template <typename T>
bool GetCompileValue(const nlohmann::json& all_vars, const std::string& name, T& value) {
  if (all_vars.empty()) {
    return false;
  }

  if (all_vars.count(name) == 0) {
    return false;
  }

  value = all_vars[name].get<T>();
  return true;
}

template <typename T1, typename T2>
bool GetCompileValue(const nlohmann::json& all_vars, const std::string& name, T1& value, const T2 default_value) {
  if (!GetCompileValue(all_vars, name, value)) {
    value = static_cast<T1>(default_value);
  }
  return true;
}

/*
 * @brief: transfor the json to vector_int64, with the json string key
 * @param [in] op_type: op type
 * @param [in] compile_info_json: the compile info json class
 * @param [in] compile_info_key: the string vector, inclue the key value for op_type
 * @param [in] compile_info_vec: the result vector of int64_t, base on the compile_info_key
 * @return bool: true or false;
 */
void* ParseCompileToInt64Vec(const ge::Operator& op, const ge::AscendString compile_info,
                             const std::vector<std::string>& compile_info_key,
                             const std::map<std::string, int64_t>& optional_key);

}  // namespace optiling
#endif  // CANN_OPS_BUILT_IN_OP_TILING_OP_TILING_UTIL_H_