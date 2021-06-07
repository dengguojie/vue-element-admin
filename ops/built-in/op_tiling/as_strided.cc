/*
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * dynamic TransposeD op tiling
 */

#include "as_strided.h"

#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>
#include <math.h>

#include "register/op_tiling.h"
#include "op_log.h"
#include "op_tiling.h"
#include "securec.h"
#include <nlohmann/json.hpp>

using namespace std;

namespace optiling {

static void Serialize(OpRunInfo& runInfo, const AsStridedInfo& asInfo) {
    vector<int64_t> tilingData;
    tilingData.push_back(asInfo.axisLen);
    tilingData.push_back(asInfo.stride);
    for (int64_t i = 0; i < tilingData.size(); i++) {
        ByteBufferPut(runInfo.tiling_data, tilingData[i]);
    }
    runInfo.block_dim = 96;
};

bool AsStridedTiling(const std::string &opType,
                     const TeOpParas &opParas,
                     const nlohmann::json &opInfo,
                     OpRunInfo &runInfo) {
    OP_LOGI(opType.c_str(), "Tiling is running.");
    AsStridedInfo asInfo;
    asInfo.axisLen = 1000;
    asInfo.stride = 10;
    Serialize(runInfo, asInfo);

    return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(AsStrided, AsStridedTiling);

};
