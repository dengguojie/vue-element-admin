/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file sign_bits_unpack.cc
 * \brief dynamic shape tiling of sign_bits_unpack
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
  int64_t act_core_num = 0;
  int64_t one_core_ele = 0;
  int64_t last_core_ele = 0;
  int64_t one_core_loop_num = 0;
  int64_t one_core_loop_left = 0;
  int64_t last_core_loop_num = 0;
  int64_t last_core_loop_left = 0;
  int64_t overlap = 0;
  int64_t max_ele = 0;
  int64_t align_block = 0;
  int64_t shape = 0;
};

struct CompileInfoParam {
  // get compile info
  int64_t ub_ele = 1;
  int64_t align_unit = 1;
  int64_t ub_spilt = 1;
  int64_t num_each_block = 1;
  int64_t core_num = 1;
  int64_t dim = 1;
};

static void PrintTilingParam(const TilingParam& param) {
  OP_LOGD("SignBitsUnpackTiling", "act_core_num=%d.", param.act_core_num);
  OP_LOGD("SignBitsUnpackTiling", "one_core_ele=%d.", param.one_core_ele);
  OP_LOGD("SignBitsUnpackTiling", "last_core_ele=%d.", param.last_core_ele);
  OP_LOGD("SignBitsUnpackTiling", "one_core_loop_num=%d.", param.one_core_loop_num);
  OP_LOGD("SignBitsUnpackTiling", "one_core_loop_left=%d.", param.one_core_loop_left);
  OP_LOGD("SignBitsUnpackTiling", "last_core_loop_num=%d.", param.last_core_loop_num);
  OP_LOGD("SignBitsUnpackTiling", "last_core_loop_left=%d.", param.last_core_loop_left);
  OP_LOGD("SignBitsUnpackTiling", "overlap=%d.", param.overlap);
  OP_LOGD("SignBitsUnpackTiling", "max_ele=%d.", param.max_ele);
  OP_LOGD("SignBitsUnpackTiling", "align_block=%d.", param.align_block);
  OP_LOGD("SignBitsUnpackTiling", "shape=%d.", param.shape);
}

static void SetTilingParam(const TilingParam& param, OpRunInfo& run_info) {
  ByteBufferPut(run_info.tiling_data, param.act_core_num);
  ByteBufferPut(run_info.tiling_data, param.one_core_ele);
  ByteBufferPut(run_info.tiling_data, param.last_core_ele);
  ByteBufferPut(run_info.tiling_data, param.one_core_loop_num);
  ByteBufferPut(run_info.tiling_data, param.one_core_loop_left);
  ByteBufferPut(run_info.tiling_data, param.last_core_loop_num);
  ByteBufferPut(run_info.tiling_data, param.last_core_loop_left);
  ByteBufferPut(run_info.tiling_data, param.overlap);
  ByteBufferPut(run_info.tiling_data, param.max_ele);
  ByteBufferPut(run_info.tiling_data, param.align_block);
  ByteBufferPut(run_info.tiling_data, param.shape);
}

static void CalCoreNum(TilingParam& param, int64_t total_ele, int64_t core_num) {
  if (core_num == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("SignBitsUnpack", "core_num value cannot be zero");
    return;
  }
  param.one_core_ele = (total_ele + core_num - 1) / core_num;
  param.act_core_num = total_ele / param.one_core_ele;
  if (total_ele % param.one_core_ele != 0) {
    ++param.act_core_num;
  }
  param.last_core_ele = total_ele - (param.act_core_num - 1) * param.one_core_ele;
}

static void CalTilingParam(TilingParam& param, int64_t shape, const CompileInfoParam& compile_info_param) {
  int64_t ub_ele = compile_info_param.ub_ele;
  int64_t align_unit = compile_info_param.align_unit;
  int64_t ub_spilt = compile_info_param.ub_spilt;
  int64_t num_each_block = compile_info_param.num_each_block;
  int64_t core_num = compile_info_param.core_num;
  int64_t pack_rate = 8;

  shape = shape * pack_rate;
  param.shape = shape;
  int64_t total_unint = 0;
  if (shape % align_unit != 0) {
    total_unint = shape / align_unit + 1;
  } else {
    total_unint = shape / align_unit;
  }

  if (shape > align_unit) {
    param.overlap = total_unint * align_unit - shape;
  } else {
    param.overlap = 0;
  }

  param.align_block = shape / num_each_block + 1;
  param.max_ele = ub_ele / ub_spilt;
  CalCoreNum(param, total_unint, core_num);
  param.one_core_loop_num = param.one_core_ele / param.max_ele;
  param.one_core_loop_left = param.one_core_ele % param.max_ele;
  param.last_core_loop_num = param.last_core_ele / param.max_ele;
  param.last_core_loop_left = param.last_core_ele % param.max_ele;
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
bool SignBitsUnpackTiling(const string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                          OpRunInfo& run_info) {
  OP_LOGI(op_type.c_str(), "SignBitsUnpackTiling running.");

  CompileInfoParam compile_info_param;
  const map<string, int64_t&> compile_params = {
      {"ub_ele", compile_info_param.ub_ele}, {"align_unit", compile_info_param.align_unit},
      {"ub_spilt", compile_info_param.ub_spilt}, {"num_each_block", compile_info_param.num_each_block},
      {"core_num", compile_info_param.core_num}, {"dim", compile_info_param.dim}};

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
  OP_TILING_CHECK(((in_shape * pack_rate) % compile_info_param.dim != 0),
                  VECTOR_INNER_ERR_REPORT_TILIING(
                  op_type, "out put shape must be divisible by dim size, dim is %lu.", compile_info_param.dim),
                  return false);

  TilingParam param;
  // calc tiling params, set tiling params, print tiling params
  CalTilingParam(param, in_shape, compile_info_param);

  SetTilingParam(param, run_info);
  PrintTilingParam(param);

  // block_dim, use fot tik op; workspace, null for tik op
  run_info.block_dim = param.act_core_num;
  vector<int64_t> workspace;
  run_info.workspaces = workspace;

  OP_LOGI(op_type.c_str(), "SignBitsUnpackTiling run success.");
  return true;
}

// register tiling interface of SignBitsUnpack op.
REGISTER_OP_TILING_FUNC_BUFFERED(SignBitsUnpack, SignBitsUnpackTiling);
}  // namespace optiling