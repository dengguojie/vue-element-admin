/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file auto_tiling_rt2.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_AUTO_TILING_RT2_H
#define OPS_BUILT_IN_OP_TILING_AUTO_TILING_RT2_H

#include <string>
#include <nlohmann/json.hpp>
#include "exe_graph/runtime/tiling_context.h"
#include "vector_op_info.h"

namespace optiling {
/*
 * @brief: tiling parser function of custom ops
 * @param [in] op_type: op_type of ops
 * @param [in] json_compile_info: compile_info of ops
 * @return std::shared_ptr<AutoTilingCompileInfo>: AutoTilingCompileInfo pointer
 */
std::shared_ptr<optiling::AutoTilingCompileInfo> ParseAutoTiling(
    const char* op_type, const nlohmann::json& json_compile_info);

/*
 * @brief: tiling function of custom ops
 * @param [in] context: inputs/outputs/attrs of ops
 * @param [in] op_info: inputs/in_type/attrs/compile_info of ops
 * @return bool: success or not
 */
bool DoAutoTiling(gert::TilingContext* context, const optiling::OpInfo* op_info);
}

#endif  // OPS_BUILT_IN_OP_TILING_AUTO_TILING_RT2_H