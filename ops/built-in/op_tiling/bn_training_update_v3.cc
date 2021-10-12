/* Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#include <algorithm>
#include "error_log.h"
#include "vector_tiling.h"
#include "graph/utils/op_desc_utils.h"

namespace optiling {
    bool BNTrainingUpdateV3Tiling(const std::string& op_type, const ge::Operator& op_paras,
                                const nlohmann::json& op_info, utils::OpRunInfo& run_info) {
        auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
        OP_TILING_CHECK(operator_info == nullptr,
                        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetOpDescFromOperator return nullptr"), return false);
        auto input_x_desc = operator_info->MutableInputDesc(0);
        OP_TILING_CHECK(input_x_desc == nullptr,
                        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_x opdesc failed"), return false);
        
        std::vector<int64_t> shape_x{std::vector<int64_t>(5, 0)};
        shape_x = input_x_desc->MutableShape().GetDims();
        int32_t N = shape_x[0];
        int32_t H = shape_x[2];
        int32_t W = shape_x[3];

        float batch_var_scalar = 1.0;
        float num_rec = 1.0;
        int32_t num = N*H*W;
        if (op_info.count("bn_update_num_rec_dtype") > 0) {
            num_rec = 1.0 / (float)num;
        }
        if (op_info.count("bn_update_batch_var_scaler_dtype") > 0) {
            batch_var_scalar = (float)(num) / (float)((num) - 1);
        }

        EletwiseTiling(op_type, op_paras, op_info, run_info);
        run_info.AddTilingData(static_cast<float>(num_rec));
        run_info.AddTilingData(static_cast<float>(batch_var_scalar));
        return true;
    }

    REGISTER_OP_TILING_FUNC_BUFFERED_V2(BNTrainingUpdateV3, BNTrainingUpdateV3Tiling);
}   // namespace optiling