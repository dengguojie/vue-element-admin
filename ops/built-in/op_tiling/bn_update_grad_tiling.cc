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
#include <nlohmann/json.hpp>
#include <string>
#include <iostream>
#include <cmath>

#include "error_util.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "op_tiling.h"
#include "graph/utils/op_desc_utils.h"
#include "op_tiling_util.h"
#include "vector_tiling.h"
#include "vector_tiling_log.h"

namespace optiling {
struct BNTrainingUpdateGradCompileInfo {
    bool has_epsilon;
    int64_t max_ub_count;
    std::string mode;
    std::vector<int32_t> common_info;
};

struct BNGradTilingInfo {
    int32_t block_dim;
    int32_t block_tiling_axis;
    int64_t block_tiling_factor;
    int32_t ub_tiling_axis;
    int64_t ub_tiling_factor;
};

struct BNGradCompileInfo {
    bool atomic = false;
    bool is_keep_dims = false;
    int32_t core_num;
    int32_t input_block_size;
};

int32_t CalcBNGradTilingKey(BNGradTilingInfo& tilingInfo, int32_t pattern, BNGradCompileInfo& compileInfo) {
    using namespace std;
    int db = 0;
    int shape_type = 0;
    vector<int> pos = {db, shape_type, tilingInfo.block_tiling_axis, tilingInfo.ub_tiling_axis, pattern};
    vector<int> coefficient = {1000000000, 10000000, 1000000, 100000, 100};
    int32_t key = 0;
    for (size_t i = 0; i < coefficient.size(); i++) {
    key += pos[i] * coefficient[i];
    }

    return key;
}

int32_t get_nearest_factor(int32_t dim, int32_t split_size) {
    /*
     * find the exact division factor small than split_size as nearest_factor,
     * if distance of nearest_factor and split_size is small, will use the
     * nearest_factor as factor, otherwise use the split_size
     */
    int32_t nearest_factor = split_size;
    while (dim % nearest_factor != 0) {
        nearest_factor -= 1;
    }
    if (int(split_size / nearest_factor) < 2) {
        split_size = nearest_factor;
    }
    return split_size;
}

bool GetBNGradTilingData(int32_t n, int32_t c1, int32_t h, int32_t w, int32_t c0, vector<int64_t> input_shape,
                         int64_t max_ub_count, int32_t core_num, BNGradTilingInfo& tilingInfo) {
    tilingInfo.block_dim = -1;
    tilingInfo.block_tiling_axis = -1;
    tilingInfo.block_tiling_factor = -1;
    tilingInfo.ub_tiling_axis = -1;
    tilingInfo.ub_tiling_factor = -1;

    int32_t ub_split_axis = 0;
    int32_t ub_split_inner = 0;

    const int DB = 2;
    if (max_ub_count / (h * w * c0) >= DB && ((c1 >= core_num && c1 % core_num == 0) ||
        (n >= core_num && n % core_num == 0))) {
        ub_split_axis = 0;
        ub_split_inner = 1;
        int32_t n_inner = 0;
        if (c1 >= core_num && c1 % core_num == 0) {
            n_inner = n;
        } else {
            n_inner = n / core_num;
        }

        for (int32_t i = n_inner; i > 0; i--) {
            if (n_inner % i != 0) {
                continue;
            }
            if (h * w * c0 * i > max_ub_count) {
                continue;
            }

            ub_split_inner = i;
            break;
        }
        tilingInfo.block_dim = core_num;
        tilingInfo.ub_tiling_axis = ub_split_axis;
        tilingInfo.ub_tiling_factor = ub_split_inner;

        return true;
    }

    int32_t block_tiling_inner_loop = input_shape[2];
    int32_t bound_size = max_ub_count;
    int32_t split_axis = 2;
    int32_t temp_size = 1;
    bool need_split = false;

    for (int32_t i = 4; i > 1; i--) {
        temp_size = temp_size * input_shape[i];
        if (temp_size >= bound_size) {
            split_axis = i;
            temp_size = temp_size / input_shape[i];
            need_split = true;
            break;
        }
    }

    int32_t split_size = 1;
    if (need_split) {
        for (int32_t i = 1; i < input_shape[split_axis] + 1; i++) {
            if (temp_size * i > bound_size) {
                split_size = i - 1;
                split_size = get_nearest_factor(input_shape[split_axis], split_size);
                break;
            }
        }
    } else {
        split_size = block_tiling_inner_loop;
    }

    if (split_axis == 2 && split_size > block_tiling_inner_loop) {
        split_size = block_tiling_inner_loop;
    }

    ub_split_inner = split_size;
    ub_split_axis = split_axis;

    tilingInfo.block_dim = core_num;
    tilingInfo.ub_tiling_axis = ub_split_axis;
    tilingInfo.ub_tiling_factor = ub_split_inner;

    return true;
}

vector<int32_t> get_factors_of_positive_integer(int32_t value) {
    vector<int32_t> factors;

    int32_t sqrt_n = sqrt(value);
    for (int32_t i = 1; i <= sqrt_n; i++) {
        if (value % i == 0) {
            int32_t tmp = value / i;
            factors.push_back(i);
            if (tmp != i) {
                factors.push_back(tmp);
            }
        }
    }
    sort(factors.begin(), factors.end());
    return factors;
}

int32_t find_closest_factor(vector<int32_t> factors, int32_t value) {
    int32_t index = 0;
    bool is_find = false;
    for (uint32_t i = 0; i < factors.size(); i++) {
        if (factors[i] > value) {
            index = i;
            is_find = true;
            break;
        }
    }
    if (is_find) {
        if (index > 0) {
            index = index - 1;
        }
    } else {
        index = factors.size() - 1;
    }
    int32_t closest_factor = factors[index];
    return closest_factor;
}

bool GetBNGradCompileInfo(BNGradCompileInfo& compileInfo, const std::string& op_type, 
                          const BNTrainingUpdateGradCompileInfo& parsed_info) {
    std::vector<int32_t> common_info = parsed_info.common_info;
    try {
        compileInfo.core_num = common_info[0];
        compileInfo.is_keep_dims = (bool)common_info[1];
        compileInfo.input_block_size = common_info[2];
        compileInfo.atomic = (bool)common_info[3];
    } catch (const std::exception &e) {
        GE_LOGE("op [%s]: get compileInfo[common_info] error. Error message: %s", op_type.c_str(), e.what());
        return false;
    }

    return true;
}

bool BNUpdateGradTiling(const std::string& op_type, const ge::Operator& op_paras,
                        const BNTrainingUpdateGradCompileInfo& parsed_info, utils::OpRunInfo& run_info) {
    auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
    OP_TILING_CHECK(operator_info == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetOpDescFromOperator return nullptr"), return false);
    auto input_x_desc = operator_info->MutableInputDesc(1);
    OP_TILING_CHECK(input_x_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get input_x opdesc failed"),
                    return false);
    auto output_y_desc = operator_info->MutableOutputDesc(0);
    OP_TILING_CHECK(output_y_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get output_y opdesc failed"),
                    return false);
    
    std::vector<int64_t> input_shape = input_x_desc->MutableShape().GetDims();
    std::vector<int64_t> output_shape = output_y_desc->MutableShape().GetDims();

    // input format is NC1HWC0
    int32_t n = input_shape[0];
    int32_t c1 = input_shape[1];
    int32_t h = input_shape[2];
    int32_t w = input_shape[3];
    int32_t c0 = input_shape[4];

    int64_t max_ub_count = parsed_info.max_ub_count;
    std::string mode = parsed_info.mode;

    BNGradCompileInfo compileInfo;
    BNGradTilingInfo tilingInfo;
    GetBNGradCompileInfo(compileInfo, op_type, parsed_info);
    int32_t core_num = compileInfo.core_num;
    int32_t half_core_num = core_num / 2;

    GetBNGradTilingData(n, c1, h, w, c0, input_shape, max_ub_count, core_num, tilingInfo);

    int32_t block_tiling_axis = -1;
    int32_t ub_tiling_axis = tilingInfo.ub_tiling_axis;
    int32_t ub_tiling_factor = tilingInfo.ub_tiling_factor;

    int32_t inner_loop = input_shape[ub_tiling_axis];
    int32_t outer_loop = input_shape[ub_tiling_axis] / ub_tiling_factor;

    if (c1 >= core_num) {
        // enter schedule_cut_c1
        block_tiling_axis = 1;
    } else if ((ub_tiling_axis == 2 || ub_tiling_axis == 3) &&
                outer_loop >= core_num &&
                input_shape[ub_tiling_axis] % core_num == 0) {
        inner_loop = input_shape[ub_tiling_axis] / core_num;
        // enter schedule_cut_h_or_w_twice
        block_tiling_axis = 2;
    } else if (ub_tiling_axis == 2 &&
               input_shape[ub_tiling_axis] >= half_core_num &&
               input_shape[ub_tiling_axis] % half_core_num == 0 &&
               input_shape[0] < core_num) {
        inner_loop = input_shape[ub_tiling_axis] / half_core_num;
        // enter schedule_fuse_h_n
        block_tiling_axis = 5;
    } else if (n >= core_num) {
        // enter schedule_cut_batch
        block_tiling_axis = 0;
    } else {
        // enter schedule_cut_general
        block_tiling_axis = 6;
    }

    vector<int32_t> factors = get_factors_of_positive_integer(inner_loop);
    ub_tiling_factor = find_closest_factor(factors, ub_tiling_factor);
    tilingInfo.ub_tiling_factor = ub_tiling_factor;

    tilingInfo.block_tiling_axis = block_tiling_axis;
    tilingInfo.block_tiling_factor = (input_shape[block_tiling_axis] + core_num - 1) / core_num;

    if (mode == "const") {
        run_info.SetBlockDim(tilingInfo.block_dim);
        run_info.SetTilingKey(0);

        run_info.AddTilingData(static_cast<int32_t>(tilingInfo.block_tiling_axis));
        run_info.AddTilingData(static_cast<int32_t>(tilingInfo.ub_tiling_axis));
        run_info.AddTilingData(static_cast<int32_t>(tilingInfo.block_tiling_factor));
        run_info.AddTilingData(static_cast<int32_t>(tilingInfo.ub_tiling_factor));
        return true;
    }

    run_info.SetBlockDim(tilingInfo.block_dim);
    int32_t pattern = 134;
    int32_t tiling_key = CalcBNGradTilingKey(tilingInfo, pattern, compileInfo);

    if (parsed_info.has_epsilon) {
        float epsilon;
        AttrUtils::GetFloat(operator_info, "epsilon", epsilon);
        run_info.AddTilingData(static_cast<float>(epsilon));
    }

    run_info.SetTilingKey(tiling_key);
    run_info.AddTilingData(n);
    run_info.AddTilingData(c1);
    run_info.AddTilingData(h);
    run_info.AddTilingData(w);

    run_info.AddTilingData(static_cast<int32_t>(tilingInfo.block_tiling_factor));
    run_info.AddTilingData(static_cast<int32_t>(tilingInfo.ub_tiling_factor));
    return true;
}

static bool ParseJsonCompileInfo(const std::string& op_type, const nlohmann::json& compile_info,
                                 BNTrainingUpdateGradCompileInfo & parsed_info) {
    bool has_epsilon = false;
    if (GetCompileValue(compile_info, "has_epsilon", has_epsilon)) {
        has_epsilon = true;
    }
    parsed_info.has_epsilon = has_epsilon;

    OP_TILING_CHECK(!GetCompileValue(compile_info, "max_ub_count", parsed_info.max_ub_count),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type,
                                                    "BNTrainingUpdateGradParseFunc, get max_ub_count error"),
                    return false);
    OP_TILING_CHECK(!GetCompileValue(compile_info, "mode", parsed_info.mode),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type,
                                                    "BNTrainingUpdateGradParseFunc, get mode error"),
                    return false);
    OP_TILING_CHECK(!GetCompileValue(compile_info, "common_info", parsed_info.common_info),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type,
                                                    "BNTrainingUpdateGradParseFunc, get common_info error"),
                    return false);
    return true;
}

REGISTER_OP_TILING_V3_CUSTOM(BNTrainingUpdateGrad, BNUpdateGradTiling, ParseJsonCompileInfo,
                             BNTrainingUpdateGradCompileInfo);
}
