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
 * \file resize_bilinear_v2_grad.cc
 * \brief
 */
#include <string>
#include <math.h>
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "error_log.h"

namespace optiling {
struct ResizeBilinearV2GradTilingParams
{
    int32_t tiling_mode;
    int32_t need_core_num;
    int32_t nc1_per_core;
    int32_t nc1_last_core;
    int32_t h_per_core;
    int32_t h_last_core;
    int32_t grads_h;
    int32_t grads_w;
    int32_t images_h;
    int32_t images_w;
    int32_t grad_each_core;
    int32_t output_each_core;
    int32_t grad_move_num;
    int32_t output_move_num;
    int32_t nc1;
    int32_t w_loop;
    int32_t w_tail;
};

void InitTilingParams(ResizeBilinearV2GradTilingParams& params) {
    params.tiling_mode = 0;
    params.need_core_num = 0;
    params.nc1_per_core = 0;
    params.nc1_last_core = 0;
    params.h_per_core = 0;
    params.h_last_core = 0;
    params.grads_h = 0;
    params.grads_w = 0;
    params.images_h = 0;
    params.images_w = 0;
    params.grad_each_core = 0;
    params.output_each_core = 0;
    params.grad_move_num = 0;
    params.output_move_num = 0;
    params.nc1 = 0;
    params.w_loop = 0;
    params.w_tail = 0;
}

bool ShapeOne(int32_t h, int32_t w) {
    if (h == 1 && w == 1) {
        return true;
    } else {
        return false;
    }
}

bool GetCompileInfo2(const std::string& op_type, const nlohmann::json& op_compile_info, int32_t& core_num,
                     int32_t& l1_support, int32_t& tensor_c) {
    using namespace nlohmann;
    auto all_vars = op_compile_info["vars"];
    if (all_vars.count("core_num") == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING("ResizeBilinearV2GradTiling", "GetCompileInfo, get core_num error");
        return false;
    }
    if (all_vars.count("l1_support") == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING("ResizeBilinearV2GradTiling", "GetCompileInfo, get l1_support error");
        return false;
    }
    if (all_vars.count("tensor_c") == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING("ResizeBilinearV2GradTiling", "GetCompileInfo, get tensor_c error");
        return false;
    }
    core_num = all_vars["core_num"].get<std::int32_t>();
    l1_support = all_vars["l1_support"].get<std::int32_t>();
    tensor_c = all_vars["tensor_c"].get<std::int32_t>();
    return true;
}

int32_t CeilDiv(int32_t nc1_value, int32_t core_value) {
    int32_t res = 0;
    if (core_value == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING("ResizeBilinearV2GradTiling", "CeilDiv, core_value cannot be zero");
      return 0;
    }
    res = (nc1_value + core_value - 1) / core_value;
    return res;
}

int32_t CalTail(int32_t grads_w, int32_t loop_num) {
    int32_t res = 0;
    if (loop_num == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING("ResizeBilinearV2GradTiling", "CeilDiv, loop_num cannot be zero");
      return 256;
    }
    res = grads_w % loop_num;
    if (res == 0) {
        res = loop_num;
    }
    return res;
}

int32_t CalTilingMode(std::vector<int64_t> grads_shape, std::vector<int64_t> images_shape, int32_t l1_support,
                      int32_t tensor_c) {
    int32_t tiling_mode = 0;
    int32_t grads_h = grads_shape[2];
    int32_t grads_w = grads_shape[3];
    int32_t images_h = images_shape[2];
    int32_t images_w = images_shape[3];

    if (grads_h == images_h && grads_w == images_w) {
        tiling_mode = 0;
        return tiling_mode;
    }

    if (ShapeOne(images_h, images_w) && !ShapeOne(grads_h, grads_w)) {
        if (grads_h * grads_w > tensor_c) {
            tiling_mode = 2;
        } else {
            tiling_mode = 1;
        }
        return tiling_mode;
    }

    if (ShapeOne(grads_h, grads_w) && !ShapeOne(images_h, images_w)) {
        tiling_mode = 3;
        return tiling_mode;
    }

    if (images_h * images_w < tensor_c && grads_h * grads_w < tensor_c) {
        tiling_mode = 4;
        return tiling_mode;
    }

    if (images_w <= 640 && grads_w <= 4096 && grads_w > images_w && grads_w / images_w < 255 && l1_support==1) {
        tiling_mode = 5;
    } else {
        tiling_mode = 6;
    }

    return tiling_mode;
}

bool CoreType(int32_t tiling_mode) {
    int32_t tiling_array[4] = {1, 2, 3, 4};
    bool res = false;
    res = std::any_of(std::begin(tiling_array), std::end(tiling_array), \
                      [tiling_mode](int32_t mode){return mode == tiling_mode;});
    return res;
}

void CalCoreInfo(ResizeBilinearV2GradTilingParams& tiling_params, int32_t core_num,
                 std::vector<int64_t> grads_shape) {
    int32_t need_core_num = 0;
    int32_t n = grads_shape[0];
    int32_t c1 = grads_shape[1];
    int32_t h = grads_shape[2];
    int32_t nc1 = n * c1;

    if (CoreType(tiling_params.tiling_mode)) {
        int32_t nc1_per_core = 0;
        int32_t nc1_last_core = 0;
        nc1_per_core = CeilDiv(nc1, core_num);
        need_core_num = CeilDiv(nc1, nc1_per_core);
        nc1_last_core = nc1 - (need_core_num - 1) * nc1_per_core;

        tiling_params.need_core_num = need_core_num;
        tiling_params.nc1_per_core = nc1_per_core;
        tiling_params.nc1_last_core = nc1_last_core;
    } else {
        int32_t h_per_core = 0;
        int32_t h_last_core = 0;
        h_per_core = CeilDiv(h, core_num);
        need_core_num = CeilDiv(h, h_per_core);
        h_last_core = h - (need_core_num - 1) * h_per_core;

        tiling_params.need_core_num = need_core_num;
        tiling_params.h_per_core = h_per_core;
        tiling_params.h_last_core = h_last_core;
    }
}

void CalRunningInfo(ResizeBilinearV2GradTilingParams& tiling_params, int32_t core_num, int32_t l1_support,
                    int32_t tensor_c, std::vector<int64_t> grads_shape, std::vector<int64_t> images_shape) {
    int32_t grad_each_core = grads_shape[2] * grads_shape[3] * grads_shape[4];
    int32_t output_each_core = images_shape[2] * images_shape[3] * images_shape[4];
    int32_t grad_move_num = grads_shape[3] * grads_shape[4];
    int32_t output_move_num = images_shape[3] * images_shape[4];
    int32_t nc1 = grads_shape[0] * grads_shape[1];
    int32_t grads_w = grads_shape[3];
    int32_t w_loop = CeilDiv(grads_w, 256);
    int32_t w_tail = CalTail(grads_w, 256);

    tiling_params.grads_h = grads_shape[2];
    tiling_params.grads_w = grads_shape[3];
    tiling_params.images_h = images_shape[2];
    tiling_params.images_w = images_shape[3];
    tiling_params.grad_each_core = grad_each_core;
    tiling_params.output_each_core = output_each_core;
    tiling_params.grad_move_num = grad_move_num;
    tiling_params.output_move_num = output_move_num;
    tiling_params.nc1 = nc1;
    tiling_params.w_loop = w_loop;
    tiling_params.w_tail = w_tail;
    tiling_params.tiling_mode = CalTilingMode(grads_shape, images_shape, l1_support, tensor_c);
    CalCoreInfo(tiling_params, core_num, grads_shape);
}

void SetRunningInfo(const ResizeBilinearV2GradTilingParams& tiling_params, OpRunInfo& run_info) {
    ByteBufferPut(run_info.tiling_data, tiling_params.tiling_mode);
    ByteBufferPut(run_info.tiling_data, tiling_params.need_core_num);
    ByteBufferPut(run_info.tiling_data, tiling_params.nc1_per_core);
    ByteBufferPut(run_info.tiling_data, tiling_params.nc1_last_core);
    ByteBufferPut(run_info.tiling_data, tiling_params.h_per_core);
    ByteBufferPut(run_info.tiling_data, tiling_params.h_last_core);
    ByteBufferPut(run_info.tiling_data, tiling_params.grads_h);
    ByteBufferPut(run_info.tiling_data, tiling_params.grads_w);
    ByteBufferPut(run_info.tiling_data, tiling_params.images_h);
    ByteBufferPut(run_info.tiling_data, tiling_params.images_w);
    ByteBufferPut(run_info.tiling_data, tiling_params.grad_each_core);
    ByteBufferPut(run_info.tiling_data, tiling_params.output_each_core);
    ByteBufferPut(run_info.tiling_data, tiling_params.grad_move_num);
    ByteBufferPut(run_info.tiling_data, tiling_params.output_move_num);
    ByteBufferPut(run_info.tiling_data, tiling_params.nc1);
    ByteBufferPut(run_info.tiling_data, tiling_params.w_loop);
    ByteBufferPut(run_info.tiling_data, tiling_params.w_tail);
}

void PrintTilingParams(const ResizeBilinearV2GradTilingParams& tiling_params) {
    GELOGD("op [ResizeBilinearV2GradTiling] : tiling_mode=%d.", tiling_params.tiling_mode);
    GELOGD("op [ResizeBilinearV2GradTiling] : need_core_num=%d.", tiling_params.need_core_num);
    GELOGD("op [ResizeBilinearV2GradTiling] : nc1_per_core=%d.", tiling_params.nc1_per_core);
    GELOGD("op [ResizeBilinearV2GradTiling] : nc1_last_core=%d.", tiling_params.nc1_last_core);
    GELOGD("op [ResizeBilinearV2GradTiling] : h_per_core=%d.", tiling_params.h_per_core);
    GELOGD("op [ResizeBilinearV2GradTiling] : h_last_core=%d.", tiling_params.h_last_core);
    GELOGD("op [ResizeBilinearV2GradTiling] : grads_h=%d.", tiling_params.grads_h);
    GELOGD("op [ResizeBilinearV2GradTiling] : grads_w=%d.", tiling_params.grads_w);
    GELOGD("op [ResizeBilinearV2GradTiling] : images_h=%d.", tiling_params.images_h);
    GELOGD("op [ResizeBilinearV2GradTiling] : images_w=%d.", tiling_params.images_w);
    GELOGD("op [ResizeBilinearV2GradTiling] : grad_each_core=%d.", tiling_params.grad_each_core);
    GELOGD("op [ResizeBilinearV2GradTiling] : output_each_core=%d.", tiling_params.output_each_core);
    GELOGD("op [ResizeBilinearV2GradTiling] : grad_move_num=%d.", tiling_params.grad_move_num);
    GELOGD("op [ResizeBilinearV2GradTiling] : output_move_num=%d.", tiling_params.output_move_num);
    GELOGD("op [ResizeBilinearV2GradTiling] : nc1=%d.", tiling_params.nc1);
    GELOGD("op [ResizeBilinearV2GradTiling] : w_loop=%d.", tiling_params.w_loop);
    GELOGD("op [ResizeBilinearV2GradTiling] : w_tail=%d.", tiling_params.w_tail);
}

bool ResizeBilinearV2GradTiling(const std::string& op_type, const TeOpParas& op_paras,
                                const nlohmann::json& op_compile_info, OpRunInfo& run_info) {
    using namespace ge;
    int32_t core_num;
    int32_t l1_support;
    int32_t tensor_c;
    bool get_compile_info = GetCompileInfo2(op_type, op_compile_info, core_num, l1_support, tensor_c);
    if (!get_compile_info) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ResizeBilinearV2GradTiling: GetCompileInfo error.");
        return false;
    }

    ResizeBilinearV2GradTilingParams tiling_params;
    InitTilingParams(tiling_params);

    const std::vector<int64_t>& grads_shape = op_paras.inputs[0].tensor[0].shape;
    const std::vector<int64_t>& images_shape = op_paras.inputs[1].tensor[0].shape;
    CalRunningInfo(tiling_params, core_num, l1_support, tensor_c, grads_shape, images_shape);
    SetRunningInfo(tiling_params, run_info);
    PrintTilingParams(tiling_params);

    run_info.block_dim = tiling_params.need_core_num;
    return true;
}
// register tiling interface of the ResizeBilinearV2Grad op.
REGISTER_OP_TILING_FUNC_BUFFERED(ResizeBilinearV2Grad, ResizeBilinearV2GradTiling);
// register tiling interface of the SyncResizeBilinearV2Grad op.
REGISTER_OP_TILING_FUNC_BUFFERED(SyncResizeBilinearV2Grad, ResizeBilinearV2GradTiling);
}  // namespace optiling.
