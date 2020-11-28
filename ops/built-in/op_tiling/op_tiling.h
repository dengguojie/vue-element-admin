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
 * \file example.h
 * \brief
 */

#ifndef CANN_OPS_COMMON_OP_TILING_H_
#define CANN_OPS_COMMON_OP_TILING_H_

#include <map>
#include <chrono>
#include <memory>
#include <string>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include "register/op_tiling_registry.h"
#include "op_log.h"

using namespace std;

namespace optiling {


const static bool prof_switch = std::getenv("OPTILING_PROF") != nullptr;

struct ParsedOpCompileInfo {
    std::string value;
    std::shared_ptr<void> parsed_object;
};

#define REGISTER_OP_TILING_FUNC_BUFFERED(optype, opfunc)                                                              \
bool g_##optype##_TilingEntry(const TeOpParas& para, const OpCompileInfo& cinfo, OpRunInfo& rinfo) {                  \
    std::chrono::time_point<std::chrono::steady_clock> before_tiling, after_tiling;                                   \
    if (prof_switch) {                                                                                                \
        before_tiling = std::chrono::steady_clock::now();                                                             \
    }                                                                                                                 \
    static std::map<std::string, std::shared_ptr<ParsedOpCompileInfo>> parsed_compile_info_storage;                   \
    const std::string& hash_key = cinfo.key;                                                                          \
    if (!hash_key.empty() && parsed_compile_info_storage.find(hash_key) != parsed_compile_info_storage.end()) {       \
        std::shared_ptr<ParsedOpCompileInfo> parsed_compile_info = parsed_compile_info_storage.at(hash_key);          \
        std::shared_ptr<void> parsed_object_ptr = parsed_compile_info->parsed_object;                                 \
        nlohmann::json* parsed_object = static_cast<nlohmann::json*>(parsed_object_ptr.get());                        \
	bool result = opfunc(para.op_type, para, *parsed_object, rinfo);                                              \
	if (prof_switch) {                                                                                            \
            after_tiling = std::chrono::steady_clock::now();                                                          \
	    uint64_t t = std::chrono::duration_cast<std::chrono::microseconds>(after_tiling - before_tiling).count(); \
	    GE_OP_LOGEVT("OPTILING_PROF: op_name: %s, time_cost: %d", cinfo.str.c_str(), t);                          \
	}                                                                                                             \
        return result;                                                                                                \
    }                                                                                                                 \
    const std::string& cinfo_str = cinfo.str;                                                                         \
    std::shared_ptr<nlohmann::json> parsed_object(new nlohmann::json(nlohmann::json::parse(cinfo_str)));              \
    std::shared_ptr<ParsedOpCompileInfo> parsed_compile_info(new ParsedOpCompileInfo());                              \
    parsed_compile_info->value = cinfo_str;                                                                           \
    parsed_compile_info->parsed_object = std::static_pointer_cast<void>(parsed_object);                               \
    parsed_compile_info_storage.emplace(hash_key, parsed_compile_info);                                               \
    if (prof_switch) {                                                                                                \
        after_tiling = std::chrono::steady_clock::now();                                                              \
        uint64_t t = std::chrono::duration_cast<std::chrono::microseconds>(after_tiling - before_tiling).count();     \
	GE_OP_LOGEVT("OPTILING_PROF: op_name: %s, time_cost: %d", cinfo.str.c_str(), t);                              \
    }                                                                                                                 \
    return result;                                                                                                    \
}                                                                                                                     \
REGISTER_OP_TILING_FUNC_NEW(optype, g_##optype##_TilingEntry)

}  // namespace optiling
#endif // CANN_OPS_COMMON_OP_TILING_H_
