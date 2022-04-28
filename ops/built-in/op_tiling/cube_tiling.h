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

#include "op_tiling.h"

#include <vector>
#include <string>

#include <nlohmann/json.hpp>
#include "graph/debug/ge_log.h"
#include "external/graph/operator.h"

namespace optiling {
/*
 * @brief: The struct that Conv2D/DepthwiseConv2D Tiling needs to save compile info from the info in json format.
 */
struct Conv2DTilingParseInfo {
    int32_t fmapC1 = 0;
    bool correctRangeFlag = false;
    std::string tilingType = "";
    std::vector<std::string> varMap;
    std::vector<std::string> tilingKeyList;
    std::vector<std::vector<std::string>> customVarsList;
    std::vector<std::vector<int64_t>> defaultRangeList;
    std::vector<std::vector<int64_t>> tilingRangeList;
    std::vector<int32_t> blockDimList;
    std::vector<std::vector<int32_t>> repoSeedsList;
    std::vector<std::vector<int64_t>> repoRangeList;
    std::vector<std::vector<int64_t>> costRangeList;
    // hardware info
    uint32_t aicoreNum = 0;
    uint64_t l2Size = 0;
    uint64_t l1Size = 0;
    uint64_t l0aSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0cSize = 0;
    uint64_t ubSize = 0;
    uint64_t btSize = 0;
    uint32_t ddrReadRate = 0;
    uint32_t ddrWriteRate = 0;
    uint32_t l2Rate = 0;
    uint32_t l2ReadRate = 0;
    uint32_t l2WriteRate = 0;
    uint32_t l1ToL0aRate = 0;
    uint32_t l1ToL0bRate = 0;
    uint32_t l1ToUbRate = 0;
    uint32_t l0cToUbRate = 0;
    uint32_t ubToL2Rate = 0;
    uint32_t ubToDdrRate = 0;
    uint32_t ubToL1Rate = 0;
    uint32_t cubeBandwidth = 0;
    uint32_t vectorBandwidth = 0;
    bool cubeVectorSplit = false;
    std::string socVersion = "";
    // fusion utilize info
    float preFusionUbUtilize = 0;
    int64_t preFusionVectorUtilize = 0;
    float postFusionUbUtilize = 0;
    int64_t postFusionVectorUtilize = 0;
};

/*
 * @brief: tiling function of cube category operators
 * @param [in] curShape: execution time shape info
 * @param [in] opInfo: compile time generated info of operator
 * @param [out] runInfo: result data
 * @return int: tiling id
 */
bool cube_tiling1(const std::string &op_type, const std::vector<int64_t> &input_shape, const std::string &x_format,
                  const std::vector<int64_t> &var_value, const nlohmann::json &compile_info,
                  utils::OpRunInfo &run_info);
// For op tiling registry version 4
bool cube_tiling1_v4(const std::string& opType, const std::vector<int64_t>& inputShape,
                     const std::string& xFormat, const std::vector<int64_t>& varValue,
                     const optiling::Conv2DTilingParseInfo& compileInfo, utils::OpRunInfo& runInfo);

/*
 * @brief: tiling function of cube operators
 * @param [in] compile_info: compile time generated info of operator
 * @param [in] opInfo: merge compile time generated info of operator
 * @param [out] runInfo: result data
 * @return : void
 */
void deal_with_compile_info(const nlohmann::json& compile_info,
                            nlohmann::json &opInfo);

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
                        utils::OpRunInfo& run_info);
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_TILING_H_