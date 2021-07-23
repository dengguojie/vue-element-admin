#include <algorithm>
#include <nlohmann/json.hpp>
#include <string>
#include <iostream>

#include "../op_proto/util/error_util.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "op_tiling.h"

namespace optiling {
struct BNGradTilingInfo {
    int32_t block_dim;
    int32_t block_tiling_axis;
    int64_t block_tiling_factor;
    int32_t ub_tiling_axis;
    int64_t ub_tiling_factor;
};

struct BNGradCompileInfo {
    bool is_const = false;
    bool is_const_post = false;
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
    int32_t nearest_factor = split_size;
    while (dim % nearest_factor != 0) {
        nearest_factor -= 1;
    }
    if (int(split_size / nearest_factor) < 2) {
        split_size = nearest_factor;
    }
    return split_size;
}

bool GetBNGradTilingData(int32_t n, int32_t c1, int32_t h, int32_t w, int32_t c0, vector<int64_t> input_shape, int64_t max_ub_count, int32_t core_num, BNGradTilingInfo& tilingInfo) {
    tilingInfo.block_dim = -1;
    tilingInfo.block_tiling_axis = -1;
    tilingInfo.block_tiling_factor = -1;
    tilingInfo.ub_tiling_axis = -1;
    tilingInfo.ub_tiling_factor = -1;

    int32_t ub_split_axis = 0;
    int32_t ub_split_inner = 0;

    if (max_ub_count / (h*w*c0) >= 2 && ((c1 >= core_num && c1 % core_num == 0) || (n >= core_num && n % core_num == 0))) {
        ub_split_axis = 0;
        ub_split_inner = 1;
        int32_t n_inner = 0;
        if (c1 >= core_num && c1 % core_num == 0) {
            n_inner = n;
        } else {
            n_inner = n / core_num;
        }

        for (int32_t i = n_inner; i > 0; i--) {
            if (n_inner % i != 0){
                continue;
            }
            if (h*w*c0*i > max_ub_count){
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

    for (int32_t i = 4; i > 1; i --) {
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
        for (int32_t i = 1; i < input_shape[split_axis]+1; i++) {
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

bool GetBNGradCompileInfo(BNGradCompileInfo& compileInfo, const std::string& op_type, const nlohmann::json& op_info) {
    std::vector<int32_t> common_info;
    std::vector<int32_t> pattern_info;
    try {
        common_info = op_info.at("common_info").get<std::vector<int32_t>>();
        pattern_info = op_info.at("pattern_info").get<std::vector<int32_t>>();
    } catch (const std::exception &e) {
        GE_LOGE("op [%s]: get common_info, pattern_info. Error message: %s", op_type.c_str(), e.what());
        return false;
    }

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

bool BNUpdateGradTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                        OpRunInfo& run_info) {
    
    std::vector<int64_t> input_shape = op_paras.inputs[0].tensor[0].shape;
    std::vector<int64_t> output_shape = op_paras.outputs[0].tensor[0].shape;

    int32_t n = input_shape[0];
    int32_t c1 = input_shape[1];
    int32_t h = input_shape[2];
    int32_t w = input_shape[3];
    int32_t c0 = input_shape[4];

    int64_t max_ub_count = op_info.at("max_ub_count").get<std::int64_t>();
    string mode = op_info["mode"].get<string>();

    BNGradCompileInfo compileInfo;
    BNGradTilingInfo tilingInfo;
    GetBNGradCompileInfo(compileInfo, op_type, op_info);
    int32_t core_num = compileInfo.core_num;
    int32_t half_core_num = core_num / 2;

    GetBNGradTilingData(n, c1, h, w, c0, input_shape, max_ub_count, core_num, tilingInfo);

    int32_t block_tiling_axis = -1;
    int32_t ub_tiling_axis = tilingInfo.ub_tiling_axis;
    int32_t ub_tiling_factor = tilingInfo.ub_tiling_factor;
    int32_t outer_loop = input_shape[ub_tiling_axis] / ub_tiling_factor;

    if (c1 >= core_num) {
        block_tiling_axis = 1;
    } else if ((ub_tiling_axis == 2 || ub_tiling_axis == 3) && 
                outer_loop >= core_num &&
                input_shape[ub_tiling_axis] % core_num == 0) {
        block_tiling_axis = 2;
    } else if (ub_tiling_axis == 2 && 
               input_shape[ub_tiling_axis] >= half_core_num &&
               input_shape[ub_tiling_axis] % half_core_num == 0 &&
               input_shape[0] < core_num) {
        block_tiling_axis = 5;
    } else if (n >= core_num) {
        block_tiling_axis = 0;
    } else {
        block_tiling_axis = 6;
    }

    tilingInfo.block_tiling_axis = block_tiling_axis;
    tilingInfo.block_tiling_factor = (input_shape[block_tiling_axis] + core_num - 1) / core_num;

    if (mode == "const") {
        run_info.block_dim = tilingInfo.block_dim;
        run_info.tiling_key = 0;
        ByteBufferPut(run_info.tiling_data, (int32_t)tilingInfo.block_tiling_axis);
        ByteBufferPut(run_info.tiling_data, (int32_t)tilingInfo.ub_tiling_axis);
        ByteBufferPut(run_info.tiling_data, (int32_t)tilingInfo.block_tiling_factor);
        ByteBufferPut(run_info.tiling_data, (int32_t)tilingInfo.ub_tiling_factor);
        return true;
    }

    run_info.block_dim = tilingInfo.block_dim;
    int32_t pattern = 134;
    int32_t tiling_key = CalcBNGradTilingKey(tilingInfo, pattern, compileInfo);

    run_info.tiling_key = tiling_key;
    ByteBufferPut(run_info.tiling_data, n);
    ByteBufferPut(run_info.tiling_data, c1);
    ByteBufferPut(run_info.tiling_data, h);
    ByteBufferPut(run_info.tiling_data, w);

    ByteBufferPut(run_info.tiling_data, (int32_t)tilingInfo.block_tiling_factor);
    ByteBufferPut(run_info.tiling_data, (int32_t)tilingInfo.ub_tiling_factor);
    return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(BNTrainingUpdateGrad, BNUpdateGradTiling);

}