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
 * \file dyn_seq_outer.cc
 * \brief
 */
#include <math.h>
#include <string>

#include "op_log.h"
#include "error_log.h"
#include "op_tiling_util.h"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"

namespace optiling {
    const int64_t OFFSET_NUMS = 3;
    const int64_t BATCHSIZE_MAX = 256;
    const int64_t INT_BTYES = 4;
    struct DynSeqOuterTilingParam {
        int32_t batch_size;
        int32_t feature_dim;
    };
    void InitRunningParams(DynSeqOuterTilingParam& params) {
        params.batch_size = 1;
        params.feature_dim = 1;
    }

    bool GetCoreNum(const std::string& op_type, const std::vector<int64_t>& compile_info,
                    int32_t& core_num) {
        OP_LOGD("GetCompileParams is running.");
        OP_TILING_CHECK(compile_info.size() < 1,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "the compile info num must == 1, is %zu",
                                                  compile_info.size()),
                  return false);
        core_num = compile_info[0];

        return true;
    }

    void SetRunningInfo(const DynSeqOuterTilingParam & tiling_params, utils::OpRunInfo& run_info)
    {
        OP_LOGD("SetRunningInfo is running.");
        run_info.AddTilingData(tiling_params.batch_size);
        run_info.AddTilingData(tiling_params.feature_dim);
    }

    void PrintTilingParams(const DynSeqOuterTilingParam &tiling_params)
    {
        OP_LOGD("PrintTilingParams is running.");
        OP_LOGD("op [DynSeqOuterTilingParam] : batch_size=%d.", tiling_params.batch_size);
        OP_LOGD("op [DynSeqOuterTilingParam] : feature_dim=%d.", tiling_params.feature_dim);
    }

    bool DynSeqOuterTiling(const std::string& op_type, const ge::Operator& op_paras, const std::vector<int64_t>& op_compile_info,
                           utils::OpRunInfo& run_info)
    {
        OP_LOGD("DynSeqOuterTiling is running.");
        DynSeqOuterTilingParam run_params;
        InitRunningParams(run_params);
        int32_t core_num = 0;

        OP_TILING_CHECK(!GetCoreNum(op_type, op_compile_info, core_num),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile info from nlohmann json failed."),
                  return false);

        auto op_desc = OpDescUtils::GetOpDescFromOperator(op_paras);
        auto x1_desc = op_desc->GetInputDescPtr(0);
        auto x1_shape = x1_desc->GetShape();
        run_params.feature_dim = x1_shape.GetDim(1);

        auto offset_desc = op_desc->GetInputDescPtr(2);
        auto offse_shape = offset_desc->GetShape();
        run_params.batch_size = offse_shape.GetDim(0);
        if (run_params.batch_size > BATCHSIZE_MAX)
        {
          VECTOR_INNER_ERR_REPORT_TILIING(op_type, "batch_size is over 256.");
          return false;
        }
        SetRunningInfo(run_params, run_info);
        PrintTilingParams(run_params);

        int64_t workspace = INT_BTYES * BATCHSIZE_MAX * OFFSET_NUMS;
        run_info.AddWorkspace(workspace);
        run_info.SetBlockDim(core_num);
 
        OP_LOGI(op_type.c_str(), "DynSeqOuterTiling run success.");

        return true;
    }

    static const std::vector<std::string> DynSeqOuter_COMPILE_INFO_KEY = {"core_num"};
    REGISTER_OP_TILING_V3_WITH_VECTOR(DynSeqOuter, DynSeqOuterTiling,
                                      DynSeqOuter_COMPILE_INFO_KEY, NO_OPTIONAL_VALUE);
}
// namespace optiling.
