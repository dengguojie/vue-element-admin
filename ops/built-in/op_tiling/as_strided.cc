/*
 * Copyright (c) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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

#include <math.h>
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include "op_log.h"
#include "op_tiling.h"
#include "register/op_tiling.h"
#include "external/graph/operator.h"
#include "securec.h"

using namespace std;

using namespace ge;
namespace optiling {
static void Serialize(utils::OpRunInfo& runInfo, const AsStridedInfo& asInfo) {
    vector<int64_t> tilingData;
    tilingData.push_back(asInfo.axisLen);
    tilingData.push_back(asInfo.stride);
    for (int64_t i = 0; i < static_cast<int64_t>(tilingData.size()); i++) {
        runInfo.AddTilingData(tilingData[i]);
    }
    runInfo.SetBlockDim(96);
};

bool AsStridedTiling(const std::string &opType,
                     const ge::Operator &opParas,
                     const nlohmann::json &opInfo,
                     utils::OpRunInfo &runInfo) {
    OP_LOGI(opType.c_str(), "Tiling is running.");
    AsStridedInfo asInfo;
    asInfo.axisLen = 1000;
    asInfo.stride = 10;
    Serialize(runInfo, asInfo);

    return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED_V2(AsStrided, AsStridedTiling);
};
