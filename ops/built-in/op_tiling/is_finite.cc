/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file is_finite.cc
 * \brief dynamic IsFinite tiling
 */
#include <string>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "external/graph/operator.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "error_log.h"

using namespace ge;
namespace optiling {

    const int64_t BLOCK_SIZE = 32;
    const int64_t OUTPUT_BYTE = 1;

    static int64_t GetFloorDiv(const int64_t l_value, const int64_t r_value) {
        if (r_value == 0) {
            return l_value;
        }

        return l_value / r_value;
    }

    static int64_t GetCeilDiv(const int64_t l_value, const int64_t r_value) {
        if (r_value == 0) {
            return l_value;
        }

        return (l_value + r_value - 1) / r_value;
    }

    static int64_t GetMod(const int64_t l_value, const int64_t r_value) {
        if (r_value == 0) {
            return l_value;
        }

        return l_value % r_value;
    }

    static bool CheckTensorShape(const std::string& op_type, const ge::Operator& op_paras) {
        ge::Shape input_shape = op_paras.GetInputDesc(0).GetShape();
        int64_t input_dims = input_shape.GetDimNum();
        if (input_dims <= 0) {
            VECTOR_INNER_ERR_REPORT_TILIING(op_type, "IsFiniteTiling: shape of input tensor x is invalid.");
            return false;
        }

        for (int64_t axis = 0; axis < input_dims; axis++) {
            int64_t dim = input_shape.GetDim(axis);
            if (dim <= 0) {
                VECTOR_INNER_ERR_REPORT_TILIING(op_type, "IsFiniteTiling: dim of input tensor x'shape is invalid.");
                return false;
            }
        }
        return true;
    }

    static bool
    GetCompileParams(const std::string& op_type, const nlohmann::json& op_compile_info_json, int64_t& core_num,
                     int64_t& ub_size, int64_t& input_data_byte) {
        using namespace nlohmann;

        auto all_vars = op_compile_info_json["vars"];
        if (all_vars.count("core_num") == 0) {
            VECTOR_INNER_ERR_REPORT_TILIING(op_type, "IsFiniteTiling: GetCompileParams, get core_num error.");
            return false;
        }
        core_num = all_vars["core_num"].get<std::int64_t>();

        if (all_vars.count("ub_size") == 0) {
            VECTOR_INNER_ERR_REPORT_TILIING(op_type, "IsFiniteTiling: GetCompileParams, get ub_size error.");
            return false;
        }
        ub_size = all_vars["ub_size"].get<std::int64_t>();

        if (all_vars.count("input_data_byte") == 0) {
            VECTOR_INNER_ERR_REPORT_TILIING(op_type, "IsFiniteTiling: GetCompileParams, get input_data_byte error.");
            return false;
        }
        input_data_byte = all_vars["input_data_byte"].get<std::int64_t>();

        OP_LOGD(op_type.c_str(),
                "IsFiniteTiling: GetCompileParams, core_num[%lld], ub_size[%lld], input_data_byte[%lld].", core_num,
                ub_size, input_data_byte);
        return true;
    }

    static void SetLastCoreTilingData(const int64_t& ub_max_size, const int64_t& element_size, int64_t& need_core_num,
                                      std::vector<int64_t>& core_parm_vector) {
        need_core_num = 1;
        int64_t per_core_size = 0;
        int64_t per_core_loop_cnt = 0;
        int64_t per_core_left_size = 0;
        int64_t last_core_size = element_size;
        int64_t last_core_loop_cnt = GetFloorDiv(element_size, ub_max_size);
        int64_t last_core_left_size = GetMod(element_size, ub_max_size);

        core_parm_vector.push_back(per_core_size);
        core_parm_vector.push_back(per_core_loop_cnt);
        core_parm_vector.push_back(per_core_left_size);
        core_parm_vector.push_back(last_core_size);
        core_parm_vector.push_back(last_core_loop_cnt);
        core_parm_vector.push_back(last_core_left_size);
    }

    static void
    SetIsFiniteTiling(int64_t& core_num, int64_t& ub_max_size, int64_t& need_core_num, int64_t& total_element_size,
                      std::vector<int64_t>& core_parm_vector) {
        int64_t unit_out = GetFloorDiv(BLOCK_SIZE, OUTPUT_BYTE);
        int64_t per_core_size = GetFloorDiv(total_element_size, core_num);
        if (GetMod(total_element_size, core_num) > 0) {
            per_core_size += 1;
        }
        if (GetMod(per_core_size, unit_out) > 0) {
            per_core_size = (GetFloorDiv(per_core_size, unit_out) + 1) * unit_out;
        }

        need_core_num = GetCeilDiv(total_element_size, per_core_size);
        int64_t per_core_loop_cnt = GetFloorDiv(per_core_size, ub_max_size);
        int64_t per_core_left_size = GetMod(per_core_size, ub_max_size);
        int64_t last_core_size = GetMod(total_element_size, per_core_size);
        int64_t last_core_loop_cnt = GetFloorDiv(last_core_size, ub_max_size);
        int64_t last_core_left_size = GetMod(last_core_size, ub_max_size);

        core_parm_vector.push_back(per_core_size);
        core_parm_vector.push_back(per_core_loop_cnt);
        core_parm_vector.push_back(per_core_left_size);
        core_parm_vector.push_back(last_core_size);
        core_parm_vector.push_back(last_core_loop_cnt);
        core_parm_vector.push_back(last_core_left_size);
    }

    static void
    ComputeIsFiniteTiling(int64_t& core_num, int64_t& ub_x_size, int64_t& need_core_num, int64_t& total_element_size,
                          std::vector<int64_t>& core_parm_vector) {
        // dma move 32byte unit
        int64_t ub_max_size = ub_x_size - GetMod(ub_x_size, BLOCK_SIZE / OUTPUT_BYTE);

        // one core
        if (need_core_num == 1) {
            SetLastCoreTilingData(ub_max_size, total_element_size, need_core_num, core_parm_vector);
            return;
        }

        need_core_num = core_num;
        SetIsFiniteTiling(core_num, ub_max_size, need_core_num, total_element_size, core_parm_vector);
    }

    /*
     * @brief: tiling function of op
     * @param [in] opType: opType of the op
     * @param [in] opParas: inputs/outputs/attrs of the op
     * @param [in] op_info: compile stage generated info of the op
     * @param [out] runInfo: result data
     * @return bool: success or not
     */
    bool IsFiniteTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                        utils::OpRunInfo& run_info) {
        OP_LOGD(op_type.c_str(), "IsFiniteTiling start running.");

        // get compile info
        int64_t input_data_byte = 0;
        int64_t core_num = 0;
        int64_t ub_size = 0;
        int64_t need_core_num = 0;
        int64_t total_element_size = 1;
        std::vector<int64_t> core_parm_vector;

        if (!GetCompileParams(op_type, op_info, core_num, ub_size, input_data_byte)) {
            VECTOR_INNER_ERR_REPORT_TILIING(op_type, "IsFiniteTiling: GetCompileParams error.");
            return false;
        }

        if (!CheckTensorShape(op_type, op_paras)) {
            VECTOR_INNER_ERR_REPORT_TILIING(op_type, "IsFiniteTiling: CheckTensorShape error.");
            return false;
        }

        ge::Shape input_shape = op_paras.GetInputDesc(0).GetShape();
        for (int64_t axis = 0; axis < input_shape.GetDimNum(); axis++) {
            total_element_size *= input_shape.GetDim(axis);
        }

        int64_t ub_part = 2;
        int64_t ub_usable_size = GetFloorDiv(ub_size, ub_part);
        int64_t data_one_block = GetFloorDiv(BLOCK_SIZE, input_data_byte);
        int64_t real_move_size = GetCeilDiv(total_element_size, data_one_block) * data_one_block;

        // all element can move into ub
        if (ub_usable_size >= real_move_size) {
            need_core_num = 1;
        }

        ComputeIsFiniteTiling(core_num, ub_usable_size, need_core_num, total_element_size, core_parm_vector);

        int64_t per_core_size = core_parm_vector[0];
        int64_t per_core_loop_cnt = core_parm_vector[1];
        int64_t per_core_left_size = core_parm_vector[2];
        int64_t last_core_size = core_parm_vector[3];
        int64_t last_core_loop_cnt = core_parm_vector[4];
        int64_t last_core_left_size = core_parm_vector[5];

        OP_LOGD(op_type.c_str(), "IsFiniteTiling: total_element_size=%lld", total_element_size);
        OP_LOGD(op_type.c_str(), "IsFiniteTiling: need_core_num=%lld", need_core_num);
        OP_LOGD(op_type.c_str(), "IsFiniteTiling: per_core_size=%lld, per_core_loop_cnt=%lld, per_core_left_size=%lld",
                per_core_size, per_core_loop_cnt, per_core_left_size);
        OP_LOGD(op_type.c_str(), "IsFiniteTiling: last_core_size=%lld, last_core_loop_cnt=%lld, last_core_left_size=%lld",
                last_core_size, last_core_loop_cnt, last_core_left_size);

        // set tiling data
        run_info.AddTilingData((int64_t)need_core_num);
        run_info.AddTilingData((int64_t)total_element_size);
        run_info.AddTilingData((int64_t)per_core_size);
        run_info.AddTilingData((int64_t)per_core_loop_cnt);
        run_info.AddTilingData((int64_t)per_core_left_size);
        run_info.AddTilingData((int64_t)last_core_size);
        run_info.AddTilingData((int64_t)last_core_loop_cnt);
        run_info.AddTilingData((int64_t)last_core_left_size);

        // block_dim, core num used in tik op
        run_info.SetBlockDim(need_core_num);

        OP_LOGD(op_type.c_str(), "IsFiniteTiling run success.");
        return true;
    }

    // register tiling inferface of the IsFinite op
    REGISTER_OP_TILING_FUNC_BUFFERED_V2(IsFinite, IsFiniteTiling);

}  // namespace optiling
