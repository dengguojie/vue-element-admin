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
 * \file conv_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CONV_TILING_H_
#define OPS_BUILT_IN_OP_TILING_CONV_TILING_H_

#include <vector>
#include <string>

#include <nlohmann/json.hpp>
#include "graph/debug/ge_log.h"
#include "op_tiling.h"

namespace optiling {
/*
 * @brief: tiling function of conv category operators
 * @param [in] curShape: execution time shape info
 * @param [in] opInfo: compile time generated info of operator
 * @param [out] runInfo: result data
 * @return int: tiling id
 */
int32_t ConvTiling(const std::vector<int32_t>& curShape, const std::string& dynamicMode, const nlohmann::json& opInfo,
                   OpRunInfo& runInfo);
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CONV_TILING_H_
