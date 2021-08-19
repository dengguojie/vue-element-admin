/*
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include "error_log.h"
#include "reduce_tiling.h"
#include "op_tiling.h"
#include<unordered_map>

namespace optiling {
std::string GetShape(std::vector<int64_t> shape)
{
    std::string res;
    for(auto it = shape.begin(); it < shape.end(); it++){
        res += std::to_string(*it);
        res += ", ";
    }
    return res;
}

bool BiasAddGradTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                       OpRunInfo& run_info) {
    OP_TILING_CHECK(op_paras.inputs.empty(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs cannot be empty."),
                    return false);
    OP_TILING_CHECK(op_paras.inputs[0].tensor.empty(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs[0].tensor cannot be empty."), return false);

    std::vector<int64_t> ori_shape = op_paras.inputs[0].tensor[0].ori_shape;
    std::vector<int64_t> shape = op_paras.inputs[0].tensor[0].shape;
    std::string format = op_paras.inputs[0].tensor[0].format;
    std::string ori_format = op_paras.inputs[0].tensor[0].ori_format;
    std::unordered_map<char, int> zip_shape;
    std::vector<int64_t> new_shape;
    uint64_t target_shape = 4;
    TeOpParas op_paras_tmp = op_paras;

    GELOGI("BiasAddGrad input format [%s], ori_format [%s], shape: [%s], ori_shape: [%s], "
          "and ori_format lens not compare ori_shape lens.",
           format.c_str(), ori_format.c_str(), GetShape(shape).c_str(), GetShape(ori_shape).c_str());
    if(format == "FRACTAL_Z" or format == "FRACTAL_Z_3D"){
        if(shape.size() == target_shape){
            if(ori_format.size() != ori_shape.size()){
                GELOGD("BiasAddGrad input format [%s], ori_format [%s], shape: [%s], ori_shape: [%s], "
                       "and ori_format lens not compare ori_shape lens.",
                       format.c_str(), ori_format.c_str(), GetShape(shape).c_str(),
                       GetShape(ori_shape).c_str());
                return false;
            } else{
                for(uint64_t i = 0; i < ori_format.size(); ++i){
                    zip_shape[ori_format[i]] = ori_shape[i];
                }
                int64_t shape_h_dim = zip_shape['H'];
                int64_t shape_w_dim = zip_shape['W'];
                int64_t shape_c1_dim = shape[0] / (shape_h_dim * shape_w_dim);
                new_shape = std::vector<int64_t>{shape_c1_dim, shape_h_dim, shape_w_dim};
                new_shape.insert(new_shape.end(), shape.begin() + 1, shape.end());
                if(format == "FRACTAL_Z_3D"){
                    int64_t shape_d_dim = zip_shape['D'];
                    shape_c1_dim = new_shape[0] / shape_d_dim;
                    new_shape.insert(new_shape.begin(), {shape_d_dim});
                    new_shape[1] = shape_c1_dim;
                }
            }
            op_paras_tmp.inputs[0].tensor[0].shape = new_shape;
        }
    }

    bool ret = ReduceTiling(op_type, op_paras_tmp, op_info, run_info);
    return ret;
}

REGISTER_OP_TILING_FUNC_BUFFERED(BiasAddGrad, BiasAddGradTiling);
}  // namespace optiling
