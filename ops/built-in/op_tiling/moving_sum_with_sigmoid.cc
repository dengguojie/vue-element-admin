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

#include <math.h>
#include <stdio.h>

#include <nlohmann/json.hpp>
#include <string>

#include "op_log.h"
#include "op_tiling.h"
#include "error_log.h"

namespace optiling {
    const int64_t TILING_MODE_1 = 1;
    struct MovingSumWithSigmoidTilingParam {
        int32_t tiling_mode;
        int32_t core_num;
    };
    void InitRunningParams(MovingSumWithSigmoidTilingParam& params) {
        params.tiling_mode = TILING_MODE_1;
        params.core_num = 0;
    }

    bool GetCompileParams(const std::string& op_type, const nlohmann::json& op_compile_info,
                          int32_t& core_num) {
        OP_LOGD("GetCompileParams is running.");
        using namespace nlohmann;

        if (op_compile_info == nullptr)
        {
          VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_compile_info is null");
          return false;
        }

        const auto &all_vars = op_compile_info["vars"];
        core_num = all_vars["core_num"].get<std::int32_t>();

        return true;
    }

    void SetRunningInfo(const MovingSumWithSigmoidTilingParam & tiling_params, OpRunInfo & run_info)
    {
        OP_LOGD("SetRunningInfo is running");
        ByteBufferPut(run_info.tiling_data, tiling_params.tiling_mode);
        ByteBufferPut(run_info.tiling_data, tiling_params.core_num);
    }

    void PrintTilingParams(const MovingSumWithSigmoidTilingParam &tiling_params)
    {
        OP_LOGD("PrintTilingParams is running");
        OP_LOGD("op [RnnGenMaskV2Tiling] : cal_mode=%d.", tiling_params.cal_mode);
        OP_LOGD("op [RnnGenMaskV2Tiling] : core_used=%d.", tiling_params.core_used);
    }

    bool MovingSumWithSigmoidTiling(const std::string & op_type, const TeOpParas & op_paras,
                                    const nlohmann::json & op_compile_info, OpRunInfo & run_info)
    {
        OP_LOGD("MovingSumWithSigmoidTiling is running.");
        MovingSumWithSigmoidTilingParam run_params;
        InitRunningParams(run_params);
        int32_t core_num = 0;

        bool can_get_params = GetCompileParams(op_type, op_compile_info, core_num);
        if (!can_get_params) {
            return false;
        }

        run_params.core_num = core_num;
        SetRunningInfo(run_params, run_info);
        PrintTilingParams(run_params);
        std::vector<int64_t> workspace;
        run_info.workspaces = workspace;

        run_info.block_dim = core_num;
        OP_LOGI(op_type.c_str(), "MovingSumWithSigmoidTiling run success.");

        return true;
    }
    REGISTER_OP_TILING_FUNC_BUFFERED(MovingSumWithSigmoid, MovingSumWithSigmoidTiling);
}
// namespace optiling.
