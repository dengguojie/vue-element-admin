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
#include <memory>
#include <string>
#include <nlohmann/json.hpp>
#include "register/op_tiling_registry.h"

using namespace std;

namespace optiling {

struct ParsedOpCompileInfo {
    std::string value;
    std::shared_ptr<void> parsed_object;
};

#define REGISTER_OP_TILING_FUNC_BUFFERED(optype, opfunc)                                                          \
bool g_##optype##_TilingEntry(const TeOpParas& para, const OpCompileInfo& cinfo, OpRunInfo& rinfo) {              \
    static std::map<std::string, std::shared_ptr<ParsedOpCompileInfo>> parsed_compile_info_storage;               \
    std::string hash_key = cinfo.key;                                                                             \
    std::string cinfo_str = cinfo.str;                                                                            \
    if (!hash_key.empty() && parsed_compile_info_storage.find(hash_key) != parsed_compile_info_storage.end()) {   \
        std::shared_ptr<ParsedOpCompileInfo> parsed_compile_info = parsed_compile_info_storage.at(hash_key);      \
        std::shared_ptr<void> parsed_object_ptr = parsed_compile_info->parsed_object;                             \
        nlohmann::json* parsed_object = static_cast<nlohmann::json*>(parsed_object_ptr.get());                    \
        return opfunc(para.op_type, para, *parsed_object, rinfo);                                                 \
    }                                                                                                             \
    std::shared_ptr<nlohmann::json> parsed_object(new nlohmann::json(nlohmann::json::parse(cinfo_str)));          \
    std::shared_ptr<ParsedOpCompileInfo> parsed_compile_info(new ParsedOpCompileInfo());                          \
    parsed_compile_info->value = cinfo_str;                                                                       \
    parsed_compile_info->parsed_object = std::static_pointer_cast<void>(parsed_object);                           \
    parsed_compile_info_storage.emplace(hash_key, parsed_compile_info);                                           \
    return opfunc(para.op_type, para, *parsed_object, rinfo);                                                     \
}                                                                                                                 \
REGISTER_OP_TILING_FUNC_NEW(optype, g_##optype##_TilingEntry)

}  // namespace optiling
#endif // CANN_OPS_COMMON_OP_TILING_H_