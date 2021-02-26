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
 * \file cube_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_TILING_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_TILING_H_

#include <vector>
#include <string>

#include <nlohmann/json.hpp>
#include "graph/debug/ge_log.h"
#include "op_tiling.h"

namespace optiling {
/*
 * @brief: tiling function of cube category operators
 * @param [in] curShape: execution time shape info
 * @param [in] opInfo: compile time generated info of operator
 * @param [out] runInfo: result data
 * @return int: tiling id
 */
int32_t CubeTiling(const std::string& opType, const std::vector<int32_t>& curShape, const nlohmann::json& opInfo,
                   OpRunInfo& runInfo);

/*
* @brief: tiling function of conv3d forward and backprop
* @param [in] op_type: op_type of the conv3d forward and backprop
* @param [in] input_shape: input shape of the conv3d forward and backprop
* @param [in] output_shape: output shape of the conv3d forward and backprop
* @param [in] compile_info: compile time generated info of the conv3d forward and backprop
* @param [out] run_info: result data
* @return bool: success or not
*/
bool Conv3DCommonTiling(const std::string& op_type,
                        const std::vector<int64_t>& input_shape,
                        const std::vector<int64_t>& output_shape,
                        const nlohmann::json& compile_info,
                        OpRunInfo& run_info);
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_TILING_H_