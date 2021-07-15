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
#include <algorithm>
#include "error_log.h"
#include "vector_tiling.h"

namespace optiling{
    bool BNTrainingUpdateTiling(const std::string& op_type, const TeOpParas& op_paras,
                                const nlohmann::json& op_info, OpRunInfo& run_info){
        OP_TILING_CHECK(op_paras.inputs.empty(),
                        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs cannot be empty"), return false);
        OP_TILING_CHECK(op_paras.inputs[0].tensor.empty(),
                        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs[0].tensor cannot be empty"),
                        return false);
        std::vector<int64_t> shape_x = op_paras.inputs[0].tensor[0].shape;
        int32_t N = shape_x[0];
        int32_t C1 = shape_x[1];
        int32_t H = shape_x[2];
        int32_t W = shape_x[3];

        float batch_var_scalar = 1.0;
        float num_rec = 1.0;
        int32_t num = N*H*W;
        if(op_info.count("bn_update_num_rec_dtype") > 0){
            num_rec = 1.0 / (float)num;
        }
        if(op_info.count("bn_update_batch_var_scaler_dtype") > 0){
            batch_var_scalar = (float)(num) / (float)((num) - 1);
        }

        bool ret = EletwiseTiling(op_type, op_paras, op_info, run_info);
        ByteBufferPut(run_info.tiling_data, (float)num_rec);
        ByteBufferPut(run_info.tiling_data, (float)batch_var_scalar);
        return true;
    }

    REGISTER_OP_TILING_FUNC_BUFFERED(BNTrainingUpdate, BNTrainingUpdateTiling);
}   //namespace optiling