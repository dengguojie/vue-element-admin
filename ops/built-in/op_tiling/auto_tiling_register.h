/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
 * \file auto_tiling_register.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_AUTO_TILING_REGISTER_H_
#define OPS_BUILT_IN_OP_TILING_AUTO_TILING_REGISTER_H_

#include "vector_tiling_rt2.h"

#include <vector>
#include <unordered_map>

using AutoTilingFunc = bool (*) (gert::TilingContext*, const optiling::OpInfoImpl*);
using AutoTilingParseFunc = optiling::AutoTilingCompileInfo*
    (*) (const char* op_type, const nlohmann::json& json_compile_info);

#define REGISTER_AUTO_TILING(pattern, tilingfunc, parsefunc)                                                           \
  static AutoTilingRegister g_auto_tiling_register_##__COUNTER__(pattern, tilingfunc, parsefunc);

class AutoTilingRegister {
public:
  AutoTilingRegister(optiling::SchPattern _pattern, AutoTilingFunc _tiling_func, AutoTilingParseFunc _parser) {
    auto& register_parser = RegisterParser();
    register_parser[_pattern] = _parser;
    auto& register_tiling = RegisterTiling();
    register_tiling[_pattern] = _tiling_func;
  };
  ~AutoTilingRegister() = default;
  static std::unordered_map<optiling::SchPattern, AutoTilingParseFunc>& RegisterParser() {
    static std::unordered_map<optiling::SchPattern, AutoTilingParseFunc> g_auto_tiling_parsers;
    return g_auto_tiling_parsers;
  }
  static std::unordered_map<optiling::SchPattern, AutoTilingFunc>& RegisterTiling() {
    static std::unordered_map<optiling::SchPattern, AutoTilingFunc> g_auto_tiling_funcs;
    return g_auto_tiling_funcs;
  }
};

#endif  // OPS_BUILT_IN_OP_TILING_AUTO_TILING_REGISTER_H_
