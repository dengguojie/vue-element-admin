/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include <math.h>

#include "../op_proto/util/error_util.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "op_tiling.h"
#include "../fusion_pass/common/fp16_t.hpp"

namespace optiling{

    int32_t get_nearest_factor(int32_t dim, int32_t split_size){
        int32_t nearest_factor = split_size;
        while (dim % nearest_factor != 0)
        {
            nearest_factor -= 1;
        }
        if (split_size / nearest_factor < 2)
        {
            split_size = nearest_factor;
        }
        return split_size;
    }

    void get_ub_tiling(vector<int64_t> shape_x, int32_t block_split_axis, int32_t block_split_inner_size,
                        int32_t max_ub_count, int32_t &ub_split_axis, int32_t &ub_split_inner){
        int32_t last_axis = shape_x.size() - 1;
        if (block_split_axis < 0 || block_split_axis > last_axis){
            return ;
        }
        int32_t bound_size = max_ub_count;
        int32_t split_axis = block_split_axis;
        int32_t temp_size = 1;
        bool need_split = false;

        for(int32_t i = last_axis; i >= block_split_axis; i--){
            temp_size = temp_size * shape_x[i];
            if (temp_size >= bound_size){
                split_axis = i;
                temp_size = temp_size / shape_x[i];
                need_split = true;
                break;
            }
        }

        int32_t split_size = 1;
        if (need_split){
            for(int32_t i = 1; i <= shape_x[split_axis]; i++){
                if((temp_size*i) == bound_size){
                    split_size = i;
                    break;
                }
                if((temp_size*i) > bound_size){
                    split_size = i - 1;
                    split_size = get_nearest_factor(shape_x[split_axis], split_size);
                    break;
                }
            }
        }
        else{
            split_size = block_split_inner_size;
        }

        if(split_axis==block_split_axis && split_size>block_split_inner_size){
            split_size = block_split_inner_size;
        }

        ub_split_inner = split_size;
        ub_split_axis = split_axis;
    }

    vector<int32_t> GetTilingData(vector<int64_t> shape_x, int32_t core_num, int32_t max_ub_count, int32_t &key){
        vector<int32_t> tiling_data;
        int32_t block_split_axis = 0;
        int32_t block_split_inner_size = 1;
        int32_t ub_split_axis = 0;
        int32_t ub_split_inner = 1;
        int32_t block_factor = core_num;
        if(shape_x[0] >= core_num){
            block_split_inner_size = floor(shape_x[block_split_axis] / core_num);
        }

        get_ub_tiling(shape_x, block_split_axis, block_split_inner_size, max_ub_count, ub_split_axis, ub_split_inner);

        int32_t ub_factor = ub_split_inner;
        if(ub_split_axis == block_split_axis){
            if(block_split_inner_size % ub_factor != 0){
                while(block_split_inner_size % ub_factor != 0){
                    ub_factor--;
                }
            }

            if(ub_split_axis == 1 && ub_factor > 1){
                ub_split_axis = 2;
                ub_factor = shape_x[2];
            }
            else if(ub_split_axis == 0){
                ub_split_axis = 1;
                ub_factor = shape_x[1];
            }
        }

        key = block_split_axis * 4 + ub_split_axis + 1;
        tiling_data.push_back(block_factor);
        tiling_data.push_back(ub_factor);
        return tiling_data;
    }

    bool BnTrainingUpdateTiling(const std::string& op_type, const TeOpParas& op_paras, 
                                const nlohmann::json& op_compile_info_json, OpRunInfo& run_info){
        GELOGI("BnTrainingUpdateTiling running.");
        std::vector<int64_t> shape_x = op_paras.inputs[0].tensor[0].shape;
        std::vector<int64_t> input_mean_shape = op_paras.inputs[1].tensor[0].shape;
        int32_t N = shape_x[0];
        int32_t C1 = shape_x[1];
        int32_t H = shape_x[2];
        int32_t W = shape_x[3];
        int32_t C0 = shape_x[4];
        
        int32_t key{0};
        int32_t core_num = op_compile_info_json["base_info"]["000"][3];
        int32_t max_ub_count = op_compile_info_json["max_ub_count"];
        int32_t block_dim = op_compile_info_json["block_dim"];

        vector<int32_t> tiling_data = GetTilingData(shape_x, core_num, max_ub_count, key);

        run_info.tiling_key = key;
        run_info.block_dim = block_dim;

        float batch_var_scalar = 1.0;
        float num_rec = 1.0;
        int32_t num = N*H*W;
        if(op_compile_info_json.count("bn_update_num_rec_dtype")>0){
            num_rec = 1.0 / (float)num;
        }
        if(op_compile_info_json.count("bn_update_batch_var_scaler_dtype")>0){
            batch_var_scalar = (float)(num) / (float)((num)-1);
        }

        ByteBufferPut(run_info.tiling_data, N);
        ByteBufferPut(run_info.tiling_data, C1);
        ByteBufferPut(run_info.tiling_data, H);
        ByteBufferPut(run_info.tiling_data, W);
        ByteBufferPut(run_info.tiling_data, C0);
        ByteBufferPut(run_info.tiling_data, (float)num_rec);
        ByteBufferPut(run_info.tiling_data, (float)batch_var_scalar);

        for(size_t i = 0; i < tiling_data.size(); i++){
            ByteBufferPut(run_info.tiling_data, tiling_data[i]);
        }
        GELOGI("BnTrainingUpdateTiling end.");
        return true;
    }

    REGISTER_OP_TILING_FUNC_BUFFERED(BnTrainingUpdate, BnTrainingUpdateTiling);
}   //namespace optiling