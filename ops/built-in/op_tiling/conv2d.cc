/*
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <string>
#include <nlohmann/json.hpp>
#include "register/op_tiling.h"
#include "graph/debug/ge_log.h"

namespace optiling {
/*
 * @brief: tiling function of conv2d
 * @param [in] op_type: op_type of the conv2d
 * @param [in] op_paras: inputs/outputs/atts of the conv2d
 * @param [in] op_compile_info: compile time generated info of the conv2d
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool Conv2DTiling(const std::string &opType,
                  const TeOpParas &opParas,
                  const nlohmann::json &opCompileInfo,
                  OpRunInfo &runInfo)
{
    if (opParas.inputs.empty() || opParas.outputs.empty() ||
        opParas.inputs[0].tensor.empty() || opParas.outputs[0].tensor.empty() ||
        opParas.inputs[0].tensor[0].shape.empty() ||
        opParas.outputs[0].tensor[0].shape.empty()) {
        return false;
    }
    std::string mode = opCompileInfo["dynamic_mode"].get<std::string>();
    GELOGD("dynamic_mode is [%s]", mode.c_str());
    GELOGD("Current format is %s, Ori format is %s",
        opParas.inputs[0].tensor[0].format.c_str(),
        opParas.inputs[0].tensor[0].ori_format.c_str());
    int32_t tilingID = 0;
    uint32_t blockDim = 0;
    if (mode == "dynamic_hw") {
        auto h = opParas.inputs[0].tensor[0].shape[2];
        auto w = opParas.inputs[0].tensor[0].shape[3];
        auto& tilingSeeds = opCompileInfo.at("repo_seeds");
        for (auto it = tilingSeeds.begin(); it != tilingSeeds.end(); it++) {
            auto& seed = it.value();
            if (h == seed[0] && w == seed[1]) {
                tilingID = std::stoi(it.key());
                blockDim = (uint32_t)opCompileInfo["block_dim"][it.key()];
            }
        }
        if (tilingID == 0) {
            auto& tilingCases = opCompileInfo.at("tiling_range");
            for (auto it = tilingCases.begin(); it != tilingCases.end(); it++) {
                auto& ranges = it.value();
                for (auto itRange = ranges.begin(); itRange != ranges.end(); itRange++) {
                    if (h >= (*itRange)[0] && h <= (*itRange)[1] &&
                        w >= (*itRange)[2] && w <= (*itRange)[3]) {
                        tilingID = std::stoi(it.key());
                        blockDim = (uint32_t)opCompileInfo["block_dim"][it.key()];
                    }
                }
            }
        }

        runInfo.block_dim = blockDim;
        ByteBufferPut(runInfo.tiling_data, tilingID);
        ByteBufferPut(runInfo.tiling_data, (int32_t)h);
        ByteBufferPut(runInfo.tiling_data, (int32_t)w);
        auto outH = opParas.outputs[0].tensor[0].shape[2];
        auto outW = opParas.outputs[0].tensor[0].shape[3];
        ByteBufferPut(runInfo.tiling_data, (int32_t)outH);
        ByteBufferPut(runInfo.tiling_data, (int32_t)outW);

        GELOGD("tiling_data is %d, %d, %d, %d, %d", tilingID, (int32_t)h, (int32_t)w,
               (int32_t)outH, (int32_t)outW);
    } else if (mode == "dynamic_batch") {
        auto curB = opParas.inputs[0].tensor[0].shape[0];
        auto& tilingCase = opCompileInfo.at("tiling_range");
        for (auto it = tilingCase.begin(); it != tilingCase.end(); it++) {
            auto& range = it.value();
            if (curB >= range[0] && curB <= range[1]) {
                tilingID = std::stoi(it.key());
                runInfo.block_dim = (uint32_t)opCompileInfo["block_dim"][it.key()];
            }
        }
        ByteBufferPut(runInfo.tiling_data, tilingID);
        ByteBufferPut(runInfo.tiling_data, (int32_t)curB);
        GELOGD("Input info is %d, %d.", tilingID, (int32_t)curB);

    } else {
        GE_LOGE("mode: %s is not supported", mode.c_str());
        return false;
    }

    if (tilingID == 0) {
        GE_LOGE("This shape is not covered by any tiling, "
            "please modify range and recompile");
        return false;
    }
    return true;
}

// register tiling interface of the conv2d
REGISTER_OP_TILING_FUNC(Conv2D, Conv2DTiling);
}