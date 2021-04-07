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
 * \file space_to_batch_nd.cc
 * \brief dynamic shape tiling of space_to_batch_nd
 */
#include <map>
#include <nlohmann/json.hpp>

#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "op_tiling.h"

namespace optiling {
using namespace ge;
using namespace std;

struct TilingParam {
  int32_t tiling_mode = 0;
  int32_t act_core_num = 0;
  int32_t one_core_ele = 0;
  int32_t last_core_ele = 0;
  int32_t input_b = 0;
  int32_t block_d = 0;  // block at depth
  int32_t block_h = 0;  // block at height
  int32_t block_w = 0;  // block at width
  int32_t pads_f = 0;   // front
  int32_t pads_a = 0;   // after
  int32_t pads_t = 0;   // top
  int32_t pads_b = 0;   // bottom
  int32_t pads_l = 0;   // left
  int32_t pads_r = 0;   // right
  int32_t input_d = 0;
  int32_t channel_one = 0;
  int32_t input_h = 0;
  int32_t input_w = 0;
  int32_t channel_zero = 0;  // equal to 16
  int32_t output_b = 0;
  int32_t output_d = 0;
  int32_t output_h = 0;
  int32_t output_w = 0;
};

static void PrintTilingParam(const TilingParam& param) {
  OP_LOGD("SpaceToBatchNDTiling",
          "(tiling_mode,act_core_num,one_core_ele,last_core_ele,input_b,block_d,block_h,block_w,pads_f,pads_a,pads_t,"
          "pads_b,pads_l,pads_r,input_d,channel_one,input_h,input_w,channel_zero,output_b,output_d,output_h,output_w):"
          "(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d)",
          param.tiling_mode, param.act_core_num, param.one_core_ele, param.last_core_ele, param.input_b, param.block_d,
          param.block_h, param.block_w, param.pads_f, param.pads_a, param.pads_t, param.pads_b, param.pads_l,
          param.pads_r, param.input_d, param.channel_one, param.input_h, param.input_w, param.channel_zero,
          param.output_b, param.output_d, param.output_h, param.output_w);
}

static void SetTilingParam(const TilingParam& param, OpRunInfo& run_info) {
  ByteBufferPut(run_info.tiling_data, param.tiling_mode);
  ByteBufferPut(run_info.tiling_data, param.act_core_num);
  ByteBufferPut(run_info.tiling_data, param.one_core_ele);
  ByteBufferPut(run_info.tiling_data, param.last_core_ele);
  ByteBufferPut(run_info.tiling_data, param.input_b);
  ByteBufferPut(run_info.tiling_data, param.block_d);
  ByteBufferPut(run_info.tiling_data, param.block_h);
  ByteBufferPut(run_info.tiling_data, param.block_w);
  ByteBufferPut(run_info.tiling_data, param.pads_f);
  ByteBufferPut(run_info.tiling_data, param.pads_a);
  ByteBufferPut(run_info.tiling_data, param.pads_t);
  ByteBufferPut(run_info.tiling_data, param.pads_b);
  ByteBufferPut(run_info.tiling_data, param.pads_l);
  ByteBufferPut(run_info.tiling_data, param.pads_r);
  ByteBufferPut(run_info.tiling_data, param.input_d);
  ByteBufferPut(run_info.tiling_data, param.channel_one);
  ByteBufferPut(run_info.tiling_data, param.input_h);
  ByteBufferPut(run_info.tiling_data, param.input_w);
  ByteBufferPut(run_info.tiling_data, param.channel_zero);
  ByteBufferPut(run_info.tiling_data, param.output_b);
  ByteBufferPut(run_info.tiling_data, param.output_d);
  ByteBufferPut(run_info.tiling_data, param.output_h);
  ByteBufferPut(run_info.tiling_data, param.output_w);
}

static void CalCoreNum(TilingParam& param, int32_t total_ele, int32_t core_num) {
  param.one_core_ele = (total_ele + core_num - 1) / core_num;
  param.act_core_num = total_ele / param.one_core_ele;
  if (total_ele % param.one_core_ele != 0) {
    param.act_core_num = param.act_core_num + 1;
  }
  param.last_core_ele = total_ele - (param.act_core_num - 1) * param.one_core_ele;
}

static void CalTilingParam(TilingParam& param, const vector<int64_t>& input_shape, const string& input_format,
                           int32_t ub_ele, int32_t core_num, const vector<int64_t>& block_vec,
                           const vector<int64_t>& pads_vec) {
  // calc input and output dim
  param.input_b = input_shape[0];
  if (input_format == "NC1HWC0") {
    param.block_h = block_vec[0];
    param.block_w = block_vec[1];
    param.pads_t = pads_vec[0];
    param.pads_b = pads_vec[1];
    param.pads_l = pads_vec[2];
    param.pads_r = pads_vec[3];
    param.channel_one = input_shape[1];
    param.input_h = input_shape[2];
    param.input_w = input_shape[3];
    param.channel_zero = input_shape[4];
    param.output_b = param.input_b * param.block_h * param.block_w;
    param.output_h = (param.input_h + param.pads_t + param.pads_b) / param.block_h;
    param.output_w = (param.input_w + param.pads_l + param.pads_r) / param.block_w;
  } else {
    param.block_d = block_vec[0];
    param.block_h = block_vec[1];
    param.block_w = block_vec[2];
    param.pads_f = pads_vec[0];
    param.pads_a = pads_vec[1];
    param.pads_t = pads_vec[2];
    param.pads_b = pads_vec[3];
    param.pads_l = pads_vec[4];
    param.pads_r = pads_vec[5];
    param.input_d = input_shape[1];
    param.channel_one = input_shape[2];
    param.input_h = input_shape[3];
    param.input_w = input_shape[4];
    param.channel_zero = input_shape[5];
    param.output_b = param.input_b * param.block_d * param.block_h * param.block_w;
    param.output_d = (param.input_d + param.pads_f + param.pads_a) / param.block_d;
    param.output_h = (param.input_h + param.pads_t + param.pads_b) / param.block_h;
    param.output_w = (param.input_w + param.pads_l + param.pads_r) / param.block_w;
  }

  // select tiling mode
  if (param.output_h * param.output_w * param.block_w * param.channel_zero <= ub_ele / 4) {
    param.tiling_mode = input_format == "NC1HWC0" ? 0 : 6;
  } else if (param.output_h * param.output_w * param.block_w * param.channel_zero <= ub_ele / 2) {
    param.tiling_mode = input_format == "NC1HWC0" ? 1 : 7;
  } else if (param.output_w * param.block_w * param.channel_zero <= ub_ele / 4) {
    param.tiling_mode = input_format == "NC1HWC0" ? 2 : 8;
  } else if (param.output_w * param.block_w * param.channel_zero <= ub_ele / 2) {
    param.tiling_mode = input_format == "NC1HWC0" ? 3 : 9;
  } else if (param.output_w * param.channel_zero <= ub_ele && param.output_w < 4096) {
    param.tiling_mode = input_format == "NC1HWC0" ? 4 : 10;
  } else {
    param.tiling_mode = input_format == "NC1HWC0" ? 5 : 11;
  }

  // calc act core_num
  if (input_format == "NC1HWC0") {
    CalCoreNum(param, param.input_b * param.channel_one, core_num);
  } else {
    CalCoreNum(param, param.output_d, core_num);
  }
}

static void GetConstDataBs(const uint8_t*& const_data, const string& dtype, size_t size, vector<int64_t>& const_vec) {
  if (dtype == "int32") {
    for (size_t i = 0; i < size / sizeof(int32_t); ++i) {
      const_vec.push_back(*((int32_t*)const_data + i));
    }
  } else {  // other dtype is int64 form ops_info
    for (size_t i = 0; i < size / sizeof(int64_t); ++i) {
      const_vec.push_back(*((int64_t*)const_data + i));
    }
  }
}

template <typename T>
static bool GetCompileInfo(const nlohmann::json& op_info, const string& name, T& value) {
  const nlohmann::json& all_vars = op_info["vars"];
  if (all_vars.empty()) {
    return false;
  }
  if (all_vars.count(name) == 0) {
    return false;
  }
  value = all_vars[name].get<T>();
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
bool SpaceToBatchNDTiling(const string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                          OpRunInfo& run_info) {
  OP_LOGI(op_type.c_str(), "SpaceToBatchNDTiling running.");

  // get and check input shape
  vector<int64_t> input_shape = op_paras.inputs[0].tensor[0].shape;
  string input_format = op_paras.inputs[0].tensor[0].format;
  string ori_format = op_paras.inputs[0].tensor[0].ori_format;
  if ((input_format != "NC1HWC0") && (input_format != "NDC1HWC0")) {
    OP_LOGE(op_type.c_str(), "Get input format failed, only support NC1HWC0 and NDC1HWC0, but got %s.",
            input_format.c_str());
    return false;
  }
  if ((input_format == "NC1HWC0") && (input_shape.size() != 5)) {
    OP_LOGE(op_type.c_str(),
            "Get input shape failed at format NC1HWC0, the length of input shape must be 5, but got %d.",
            input_shape.size());
    return false;
  }
  if ((input_format == "NDC1HWC0") && (input_shape.size() != 6)) {
    OP_LOGE(op_type.c_str(),
            "Get input shape failed at format NDC1HWC0, the length of input shape must be 6, but got %d.",
            input_shape.size());
    return false;
  }

  // get compile info
  int32_t ub_ele = 0;
  int32_t core_num = 1;
  int32_t block_size = 0;
  const map<string, int32_t&> compile_params = {
      {"ub_ele", ub_ele},
      {"core_num", core_num},
      {"block_size", block_size},
  };
  for (auto& param : compile_params) {
    const auto& name = param.first;
    OP_LOGD(op_type.c_str(), "GetCompileInfo %s.", name.c_str());
    if (!GetCompileInfo<int32_t>(op_info, name, param.second)) {
      OP_LOGE(op_type.c_str(), "GetCompileInfo %s failed.", name.c_str());
      return false;
    }
    OP_LOGD(op_type.c_str(), "%s=%d.", name.c_str(), param.second);
  }

  vector<int64_t> block_vec;
  vector<int64_t> pads_vec;
  // calc block_vec and pads_vec and check supported
  // cppcheck-suppress *
  if (block_size != 0) {
    // SpaceToBatch
    block_vec.push_back(block_size);
    block_vec.push_back(block_size);
    const uint8_t* pads_data = get<0>(op_paras.const_inputs.at("paddings"));
    size_t pads_size = get<1>(op_paras.const_inputs.at("paddings"));
    string pads_dtype = op_paras.inputs[1].tensor[0].dtype;
    GetConstDataBs(pads_data, pads_dtype, pads_size, pads_vec);
  } else {
    // SpaceToBatchND
    const uint8_t* block_data = get<0>(op_paras.const_inputs.at("block_shape"));
    size_t block_size = get<1>(op_paras.const_inputs.at("block_shape"));
    string block_dtype = op_paras.inputs[1].tensor[0].dtype;
    GetConstDataBs(block_data, block_dtype, block_size, block_vec);
    const uint8_t* pads_data = get<0>(op_paras.const_inputs.at("paddings"));
    size_t pads_size = get<1>(op_paras.const_inputs.at("paddings"));
    string pads_dtype = op_paras.inputs[2].tensor[0].dtype;
    GetConstDataBs(pads_data, pads_dtype, pads_size, pads_vec);
  }

  // check and resize block_shape and paddings
  if (input_format == "NC1HWC0") {
    if ((ori_format == "NHWC") && (block_vec.size() == 1) && (pads_vec.size() == 2)) {
      block_vec.insert(block_vec.begin(), 1);
      pads_vec.insert(pads_vec.begin(), 2, 0);
    } else if (((ori_format == "NHWC") || (ori_format == "NCHW")) && (block_vec.size() == 2) &&
               (pads_vec.size() == 4)) {
      ;
    } else if ((ori_format == "NCHW") && (block_vec.size() == 3) && (pads_vec.size() == 6) && (block_vec[0] == 1) &&
               (pads_vec[0] == 0) && (pads_vec[1] == 0)) {
      block_vec.erase(block_vec.begin());
      pads_vec.erase(pads_vec.begin(), pads_vec.begin() + 2);
    } else {
      OP_LOGE(op_type.c_str(),
              "Input with format NC1HWC0 which does not meet the rules, ori_format is %s, block size is %d, pads size "
              "is %d",
              ori_format.c_str(), block_vec.size(), pads_vec.size());
      return false;
    }
  } else {
    if (((ori_format == "NDHWC") || (ori_format == "NCDHW")) && (block_vec.size() == 3) && (pads_vec.size() == 6)) {
      ;
    } else if ((ori_format == "NCDHW") && (block_vec.size() == 4) && (pads_vec.size() == 8) && (block_vec[0] == 1) &&
               (pads_vec[0] == 0) && (pads_vec[1] == 0)) {
      block_vec.erase(block_vec.begin());
      pads_vec.erase(pads_vec.begin(), pads_vec.begin() + 2);
    } else {
      OP_LOGE(op_type.c_str(),
              "Input with format NDC1HWC0 which does not meet the rules, ori_format is %s, block size is %d, pads size "
              "is %d",
              ori_format.c_str(), block_vec.size(), pads_vec.size());
      return false;
    }
  }

  // check block_shape and paddings
  if (input_format == "NC1HWC0") {
    if ((block_vec[0] <= 0) || (block_vec[1] <= 0)) {
      OP_LOGE(op_type.c_str(),
              "Get block_shape failed at format NC1HWC0, the value of block_shape must be greater to 0, but "
              "got [%d, %d].",
              block_vec[0], block_vec[1]);
      return false;
    }
    if ((pads_vec[0] < 0) || (pads_vec[1] < 0) || (pads_vec[2] < 0) || (pads_vec[3] < 0)) {
      OP_LOGE(op_type.c_str(),
              "Get pads failed at format NC1HWC0, the value of pads must be greater and equal to 0, but "
              "got [%d, %d, %d, %d].",
              pads_vec[0], pads_vec[1], pads_vec[2], pads_vec[3]);
      return false;
    }
    if ((input_shape[2] + pads_vec[0] + pads_vec[1]) % block_vec[0] != 0 ||
        (input_shape[3] + pads_vec[2] + pads_vec[3]) % block_vec[1] != 0) {
      OP_LOGE(op_type.c_str(),
              "The (input+pads)/(block_shape) should be integer, but got input:[%d, %d], block:[%d, %d], pads:[%d, %d, "
              "%d, %d]",
              input_shape[2], input_shape[3], block_vec[0], block_vec[1], pads_vec[0], pads_vec[1], pads_vec[2],
              pads_vec[3]);
      return false;
    }
  } else {
    if ((block_vec[0] <= 0) || (block_vec[1] <= 0) || (block_vec[2] <= 0)) {
      OP_LOGE(op_type.c_str(),
              "Get block_shape failed at format NDC1HWC0, the value of block_shape must be greater to 0, but "
              "got [%d, %d, %d].",
              block_vec[0], block_vec[1], block_vec[2]);
      return false;
    }
    if ((pads_vec[0] < 0) || (pads_vec[1] < 0) || (pads_vec[2] < 0) || (pads_vec[3] < 0) || (pads_vec[4] < 0) ||
        (pads_vec[5] < 0)) {
      OP_LOGE(op_type.c_str(),
              "Get pads failed at format NDC1HWC0, the value of pads must be greater and equal 0, but "
              "got [%d, %d, %d, %d, %d, %d].",
              pads_vec[0], pads_vec[1], pads_vec[2], pads_vec[3], pads_vec[4], pads_vec[5]);
      return false;
    }
    if ((input_shape[1] + pads_vec[0] + pads_vec[1]) % block_vec[0] != 0 ||
        (input_shape[3] + pads_vec[2] + pads_vec[3]) % block_vec[1] != 0 ||
        (input_shape[4] + pads_vec[4] + pads_vec[5]) % block_vec[2] != 0) {
      OP_LOGE(op_type.c_str(),
              "The (input+pads)/(block_shape) should be integer, but got input:[%d, %d, %d], block:[%d, %d, %d], "
              "pads:[%d, %d, %d, %d, %d, %d]",
              input_shape[1], input_shape[3], input_shape[4], block_vec[0], block_vec[1], block_vec[2], pads_vec[0],
              pads_vec[1], pads_vec[2], pads_vec[3], pads_vec[4], pads_vec[5]);
      return false;
    }
  }

  // calc tiling params, set tiling params, print tiling params
  TilingParam param;
  CalTilingParam(param, input_shape, input_format, ub_ele, core_num, block_vec, pads_vec);
  SetTilingParam(param, run_info);
  PrintTilingParam(param);

  // block_dim, use fot tik op; workspace, null for tik op
  run_info.block_dim = param.act_core_num;
  vector<int64_t> workspace;
  run_info.workspaces = workspace;

  OP_LOGI(op_type.c_str(), "SpaceToBatchNDTiling run success.");
  return true;
}

// register tiling interface of space_to_batch and space_to_batch_nd op.
REGISTER_OP_TILING_FUNC_BUFFERED(SpaceToBatch, SpaceToBatchNDTiling);
REGISTER_OP_TILING_FUNC_BUFFERED(SpaceToBatchND, SpaceToBatchNDTiling);
}  // namespace optiling
