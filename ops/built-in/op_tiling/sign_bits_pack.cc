/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0\\ m s 
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file sign_bits_pack.cc
 * \brief dynamic shape tiling of sign_bits_pack
 */

#include <map>
#include <nlohmann/json.hpp>

#include "error_util.h"
#include "error_log.h"
#include "op_log.h"
#include "op_tiling.h"

namespace optiling {
using namespace ge;
using namespace std;

struct TilingParam {
  int64_t shape_ceil8 = 0;
  int64_t neg1_to_fill = 0;
  int64_t shape_div8 = 0;
  int64_t shape_div_size = 0;
  int64_t overlap = 0;
  int64_t one_core_ele = 0;
  int64_t act_core_num = 0;
  int64_t last_core_ele = 0;
  int64_t one_core_loop_num = 0;
  int64_t one_core_loop_left = 0;
  int64_t last_core_loop_num = 0;
  int64_t last_core_loop_left = 0;
  int64_t last_core_number = 0;
  int64_t last_core_number_fill8 = 0;
  int64_t last_core_input_move_para = 0;
};

struct CompileInfoParam {
  // get compile info
  int64_t pack_rate = 1;
  int64_t size = 1;
  int64_t core_num = 1;
  int64_t align_unit = 1;
  int64_t max_ele = 1;
  int64_t block = 1;
};

static void PrintTilingParam(const TilingParam& param) {
  OP_LOGD("SignBitsPackTiling", "shape_ceil8=%d.", param.shape_ceil8);
  OP_LOGD("SignBitsPackTiling", "neg1_to_fill=%d.", param.neg1_to_fill);
  OP_LOGD("SignBitsPackTiling", "shape_div8=%d.", param.shape_div8);
  OP_LOGD("SignBitsPackTiling", "shape_div_size=%d.", param.shape_div_size);
  OP_LOGD("SignBitsPackTiling", "overlap=%d.", param.overlap);
  OP_LOGD("SignBitsPackTiling", "one_core_ele=%d.", param.one_core_ele);
  OP_LOGD("SignBitsPackTiling", "act_core_num=%d.", param.act_core_num);
  OP_LOGD("SignBitsPackTiling", "last_core_ele=%d.", param.last_core_ele);
  OP_LOGD("SignBitsPackTiling", "one_core_loop_num=%d.", param.one_core_loop_num);
  OP_LOGD("SignBitsPackTiling", "one_core_loop_left=%d.", param.one_core_loop_left);
  OP_LOGD("SignBitsPackTiling", "last_core_loop_num=%d.", param.last_core_loop_num);
  OP_LOGD("SignBitsPackTiling", "last_core_loop_left=%d.", param.last_core_loop_left);
  OP_LOGD("SignBitsPackTiling", "last_core_number=%d.", param.last_core_number);
  OP_LOGD("SignBitsPackTiling", "last_core_number_fill8=%d.", param.last_core_number_fill8);
  OP_LOGD("SignBitsPackTiling", "last_core_input_move_para=%d.", param.last_core_input_move_para);
}

static void SetTilingParam(const TilingParam& param, OpRunInfo& run_info) {
  ByteBufferPut(run_info.tiling_data, param.shape_ceil8);
  ByteBufferPut(run_info.tiling_data, param.neg1_to_fill);
  ByteBufferPut(run_info.tiling_data, param.shape_div8);
  ByteBufferPut(run_info.tiling_data, param.shape_div_size);
  ByteBufferPut(run_info.tiling_data, param.overlap);
  ByteBufferPut(run_info.tiling_data, param.one_core_ele);
  ByteBufferPut(run_info.tiling_data, param.act_core_num);
  ByteBufferPut(run_info.tiling_data, param.last_core_ele);
  ByteBufferPut(run_info.tiling_data, param.one_core_loop_num);
  ByteBufferPut(run_info.tiling_data, param.one_core_loop_left);
  ByteBufferPut(run_info.tiling_data, param.last_core_loop_num);
  ByteBufferPut(run_info.tiling_data, param.last_core_loop_left);
  ByteBufferPut(run_info.tiling_data, param.last_core_number);
  ByteBufferPut(run_info.tiling_data, param.last_core_number_fill8);
  ByteBufferPut(run_info.tiling_data, param.last_core_input_move_para);
}

static void CalCoreNum(TilingParam& param, int64_t total_ele, int64_t core_num) {
  if (core_num == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("SignBitsPack", "core_num value cannot be zero");
    return;
  }
  param.one_core_ele = (total_ele + core_num - 1) / core_num;
  param.act_core_num = total_ele / param.one_core_ele;
  if (total_ele % param.one_core_ele != 0) {
    ++param.act_core_num;
  }
  param.last_core_ele = total_ele - (param.act_core_num - 1) * param.one_core_ele;
}

static void CalTilingParam(TilingParam& param, int64_t input_shape, const CompileInfoParam& compile_info_param) {
  int64_t pack_rate = compile_info_param.pack_rate;
  int64_t size = compile_info_param.size;
  int64_t core_num = compile_info_param.core_num;
  int64_t align_unit = compile_info_param.align_unit;
  int64_t max_ele = compile_info_param.max_ele;
  int64_t block = compile_info_param.block;
  int64_t total_unit = 0;

  if(input_shape % pack_rate == 0) {
    param.shape_ceil8 = input_shape;
    param.neg1_to_fill = 0;
  } else {
    param.shape_ceil8 = input_shape + pack_rate - input_shape % pack_rate;
    param.neg1_to_fill = pack_rate - input_shape % pack_rate;
  }
  if (param.shape_ceil8 % pack_rate == 0) {
    param.shape_div8 = param.shape_ceil8 / pack_rate; 
  } else {
    param.shape_div8 = param.shape_ceil8 / pack_rate + 1;
  }
  if (param.shape_div8 % size == 0) {
    param.shape_div_size = param.shape_div8 / size;
  } else {
    param.shape_div_size = param.shape_div8 / size + 1;
  }
  if (param.shape_ceil8 % align_unit != 0) {
    total_unit = param.shape_ceil8 / align_unit + 1;
  } else {
    total_unit = param.shape_ceil8 / align_unit;
  }
  if (param.shape_ceil8 > align_unit) {
    param.overlap = total_unit * align_unit - param.shape_ceil8;
  } else {
    param.overlap = 0;
  }

  CalCoreNum(param, total_unit, core_num);
  param.one_core_loop_num = param.one_core_ele / max_ele;
  param.one_core_loop_left = param.one_core_ele % max_ele;
  param.last_core_loop_num = param.last_core_ele / max_ele;
  param.last_core_loop_left = param.last_core_ele % max_ele;
  param.last_core_number = (input_shape - param.one_core_ele * (param.act_core_num - 1) * align_unit - 1)
                           % align_unit + 1;
  param.last_core_number_fill8 = (param.shape_ceil8 - param.one_core_ele * (param.act_core_num - 1) * align_unit - 1) 
                                 % align_unit + 1;
  if (param.last_core_number_fill8 % block == 0) {
    param.last_core_input_move_para = param.last_core_number_fill8 / block;
  } else {
    param.last_core_input_move_para = param.last_core_number_fill8 / block + 1;
  }
}

static bool GetCompileInfo(const nlohmann::json& op_info, const string& name, int64_t& value) {
  const nlohmann::json& all_vars = op_info["vars"];
  if (all_vars.empty()) {
    OP_LOGW("vars can not be empty.");
    return false;
  }
  if (all_vars.count(name) == 0) {
    value = 0;
    OP_LOGW("Get compile info parameter failed, maybe need update om, set %s default value 0", name.c_str());
    return true;
  }
  value = all_vars[name].get<int64_t>();
  return true;
}

/*
 * @brief: tiling function of op
 * @param [in] op_type: type of the op
 * @param [in] op_paras: inputs/outputs/attrs of the op
 * @param [in] op_info: compile time generated info of the op
 * @param [out] run_info: result data
 * @return bool: success or not success
 */
bool SignBitsPackTiling(const string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                        OpRunInfo& run_info) {
  OP_LOGI(op_type.c_str(), "SignBitsPackTiling running.");
  
  CompileInfoParam compile_info_param;
  const map<string, int64_t&> compile_params = {
      {"pack_rate", compile_info_param.pack_rate}, {"size", compile_info_param.size},
      {"core_num", compile_info_param.core_num}, {"align_unit", compile_info_param.align_unit},
      {"max_ele", compile_info_param.max_ele}, {"block", compile_info_param.block}};
  
  for (auto& param : compile_params) {
    const auto& name = param.first;
    OP_LOGD(op_type.c_str(), "GetCompileInfo %s.", name.c_str());
    OP_TILING_CHECK(!GetCompileInfo(op_info, name, param.second),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetCompileInfo %s failed.", name.c_str()), 
                    return false);
    OP_LOGD(op_type.c_str(), "%s=%d.", name.c_str(), param.second);
  }

  int64_t in_shape = op_paras.inputs[0].tensor[0].shape[0];
  int64_t pack_rate = 8;
  int64_t out_shape = 0;
  if (in_shape % pack_rate == 0) {
    out_shape = in_shape / pack_rate;
  } else {
    out_shape = in_shape / pack_rate + 1;
  }
  OP_TILING_CHECK((out_shape % compile_info_param.size != 0),
                  VECTOR_INNER_ERR_REPORT_TILIING(
                  op_type, "out put shape must be divisible by dim size, dim is %lu.", compile_info_param.size),
                  return false); 

  TilingParam param;
  // calc tiling params, set tiling params, print tiling params
  CalTilingParam(param, in_shape, compile_info_param);

  SetTilingParam(param, run_info);
  PrintTilingParam(param);

  // block_dim, use for tik op; workspace, null for tik op
  run_info.block_dim = param.act_core_num;

  OP_LOGI(op_type.c_str(), "SignBitsPackTiling run success.");
  return true;
}

// register tiling interface of SignBitsPack op.
REGISTER_OP_TILING_FUNC_BUFFERED(SignBitsPack, SignBitsPackTiling);
}  // namespace optiling