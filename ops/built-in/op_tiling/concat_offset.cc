/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file concat_offset.cpp
 * \brief
 */
#include <string>
#include <math.h>
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"
#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "error_log.h"

namespace optiling {

bool ConcatOffsetTiling(const std::string& op_type,
                        const TeOpParas& op_paras,
                        const nlohmann::json& op_info,
                        OpRunInfo& run_info) {
    using namespace ge;
    if (op_paras.inputs.empty() || op_paras.inputs[1].tensor.empty()) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op [ConcatOffsetTiling] : input shape error");
      return false;
    }
    const std::vector<int64_t>& x_shape = op_paras.inputs[1].tensor[0].shape;
    int64_t input_num = x_shape.size() == 0 ? 1 : x_shape[0];
    ByteBufferPut(run_info.tiling_data, input_num);

    GELOGD("op [ConcatOffsetTiling] : CompileParams, input_num = %d.", input_num);

    run_info.block_dim = 1;
    std::vector<int64_t> workspace;
    run_info.workspaces = workspace;

    GELOGI("op[%s] tiling run success.", op_type.c_str());

    return true;
}

// register tiling interface of the ConcatOffset op.
REGISTER_OP_TILING_FUNC_BUFFERED(ConcatOffset, ConcatOffsetTiling);
}  // namespace optiling
