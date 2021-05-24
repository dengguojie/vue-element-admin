/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file vector_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_VECTOR_TILING_H_
#define OPS_BUILT_IN_OP_TILING_VECTOR_TILING_H_

#include <string>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "vector_tiling_log.h"

namespace optiling {
/*
 * @brief: tiling function of reduce operator
 * @param [in] op_type: op_type of the reduce operator
 * @param [in] op_paras: inputs/outputs/atts of the reduce operator
 * @param [in] op_info: compile time generated info of the reduce operator
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool ReduceTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                  OpRunInfo& run_info);

/*
 * @brief: tiling function of elementwise operator
 * @param [in] op_type: op_type of the elementwise operator
 * @param [in] op_paras: inputs/outputs/atts of the elementwise operator
 * @param [in] op_info: compile time generated info of the elementwise operator
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool EletwiseTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                    OpRunInfo& run_info);

/*
 * @brief: tiling function of norm operator
 * @param [in] op_type: op_type of the norm operator
 * @param [in] op_paras: inputs/outputs/atts of the norm operator
 * @param [in] op_info: compile time generated info of the norm operator
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool NormTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                OpRunInfo& run_info);

#define REGISTER_OP_TILING_FUNC_BUFFERED_CUSTOM(optype, opfunc, parserfunc)                                       \
bool g_##optype##_TilingEntry(const TeOpParas& para, const OpCompileInfo& cinfo, OpRunInfo& rinfo) {              \
    static std::map<std::string, ParsedOpCompileInfo> parsed_compile_info_storage;                                \
    std::string hash_key = cinfo.key;                                                                             \
    std::string cinfo_str = cinfo.str;                                                                            \
    if (!hash_key.empty() && parsed_compile_info_storage.find(hash_key) != parsed_compile_info_storage.end()) {   \
        ParsedOpCompileInfo& parsed_compile_info = parsed_compile_info_storage.at(hash_key);                      \
        void* parsed_object_ptr = parsed_compile_info.parsed_object;                                              \
        return opfunc(para.op_type, para, parsed_object_ptr, rinfo);                                              \
    }                                                                                                             \
    void* parsed_object_ptr = parserfunc(cinfo_str);                                                              \
    ParsedOpCompileInfo* parsed_compile_info = new ParsedOpCompileInfo();                                         \
    parsed_compile_info->value = cinfo_str;                                                                       \
    parsed_compile_info->parsed_object = parsed_object_ptr;                                                       \
    parsed_compile_info_storage.emplace(hash_key, *parsed_compile_info);                                          \
    return opfunc(para.op_type, para, parsed_object_ptr, rinfo);                                                  \
}                                                                                                                 \
REGISTER_OP_TILING(optype, g_##optype##_TilingEntry)

}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_VECTOR_TILING_H_
