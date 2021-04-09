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

/*!
 * \file softmax_tiling.cpp
 * \brief tiling function of op
 */

#include "softmax_tiling.h"

namespace optiling {

bool Softmax::IsInVector(std::vector<int32_t>& input, int32_t value) {
  for (uint32_t i = 0; i < input.size(); i++) {
    if (input[i] == value) {
      return true;
    }
  }
  return false;
}

bool Softmax::Init() {
  OP_LOGD(op_type.c_str(), "begin softmax_tiling init.");
  // get ori input_shape
  if (op_paras.inputs.size() > 0 && op_paras.inputs[0].tensor.size() > 0) {
    input_shape_ori = op_paras.inputs[0].tensor[0].shape;
  } else {
    OP_LOGE(op_type.c_str(), "input shape error.");
    return false;
  }

  const auto& reduce_axis_tmp = op_info["ori_axis"];
  reduce_axis_ori.resize(reduce_axis_tmp.size());
  size_t i = 0;
  for (const auto& axis : reduce_axis_tmp) {
      reduce_axis_ori[i] = axis;
      i++;
  }

  // convert reduce axis (-1 -> length+1)
  for (size_t i = 0; i < reduce_axis_ori.size(); i++) {
    if (reduce_axis_ori[i] < 0) {
      reduce_axis_ori[i] = input_shape_ori.size() + reduce_axis_ori[i];
    }
  }
  OP_LOGD(op_type.c_str(), "end softmax_tiling init.");

  return true;
}

bool Softmax::FusedReduceAxis() {
  size_t capacity_shape = 0;
  size_t capacity_axis = 1;
  size_t axis_ori = reduce_axis_ori[0];
  if (input_shape_ori.size() == 1) {
    reduce_axis[0] = 1;
    input_shape[0] = 1;
    input_shape[1] = input_shape_ori[0];
    capacity_shape = 2;
    pattern = 400000000;
  } else if (axis_ori == 0) {
    reduce_axis[0] = 0;
    input_shape[0] = input_shape_ori[0];
    input_shape[1] = 1;
    for (size_t i = 1; i < input_shape_ori.size(); i++) {
      input_shape[1] *= input_shape_ori[i];
    }
    capacity_shape = 2;
    pattern = 300000000;
  } else if (axis_ori == input_shape_ori.size() - 1) {
    reduce_axis[0] = 1;
    input_shape[1] = input_shape_ori[input_shape_ori.size() - 1];
    input_shape[0] = 1;
    for (size_t i = 0; i < input_shape_ori.size() - 1; i++) {
      input_shape[0] *= input_shape_ori[i];
    }
    capacity_shape = 2;
    pattern = 400000000;
  } else {
    reduce_axis[0] = 1;
    capacity_shape = 3;
    input_shape[1] = input_shape_ori[axis_ori];
    input_shape[0] = 1;
    input_shape[2] = 1;
    for (size_t i = 0; i < axis_ori - 1; i++) {
      input_shape[0] *= input_shape_ori[i];
    }
    for (size_t i = axis_ori + 1; i < input_shape_ori.size(); i++) {
      input_shape[2] *= input_shape_ori[i];
    }
    pattern = 500000000;
  }

  input_shape.resize(capacity_shape);
  reduce_axis.resize(capacity_axis);
  return true;
}

bool Softmax::GetCompileInfo() {
  std::vector<int32_t> info = op_info["common_info"];
  compileInfo.max_ub_count = info[0];
  compileInfo.core_num = info[1];
  compileInfo.is_keep_dims = (bool)info[2];
  compileInfo.reduce_block_size = info[3];
  output_dtypeUB = op_paras.inputs[0].tensor[0].dtype;
  is_last_axis_reduce = IsInVector(reduce_axis, input_shape.size() - 1);

  return true;
}

bool Softmax::GetBlockTilingInfo() {
  // rewrite block_tiling_axis, block_tiling_factor.
  int32_t core_num = compileInfo.core_num;
  if (pattern == 300000000) {
    tilingInfo.block_tiling_axis = 1;
    tilingInfo.block_tiling_factor = 1;
    if (input_shape[1] > core_num) {
      tilingInfo.block_tiling_factor = (input_shape[1] + core_num - 1) / core_num;
      tilingInfo.block_dim = core_num;
    } else {
      tilingInfo.block_dim = input_shape[1];
    }
  } else if (pattern == 400000000) {
    tilingInfo.block_tiling_axis = 0;
    tilingInfo.block_tiling_factor = 1;
    int32_t output_block_size = GetBlockSize(output_dtypeUB);
    if (input_shape[1] <= output_block_size) {
      tilingInfo.block_tiling_factor = 1;
      tilingInfo.block_dim = 1;
    } else if (input_shape[0] > core_num) {
      tilingInfo.block_tiling_factor = core_num;
      tilingInfo.block_dim = core_num;
    } else {
      tilingInfo.block_dim = input_shape[0];
    }
  } else if (pattern == 500000000) {
    if (input_shape[0] > core_num) {
      tilingInfo.block_tiling_axis = 0;
      tilingInfo.block_tiling_factor = (input_shape[0] + core_num - 1) / core_num;
      tilingInfo.block_dim = core_num;
      pattern = 500000010;
    } else {
      tilingInfo.block_tiling_axis = 2;
      if (input_shape[0] * input_shape[2] >= core_num) {
        int64_t block_tiling_factor_outer = core_num / input_shape[0];
        tilingInfo.block_tiling_factor = (input_shape[0] + block_tiling_factor_outer - 1) / block_tiling_factor_outer;
        tilingInfo.block_dim = core_num;
      } else {
        tilingInfo.block_dim = input_shape[0] * input_shape[2];
      }
      pattern = 500000030;
    }
  }
  return true;
}

int32_t Softmax::GetBlockSize(std::string dtypeUB) {
  int32_t block_size = 0;
  if (dtypeUB == "float32" || dtypeUB == "int32" || dtypeUB == "uint32") {
    block_size = 8;
  } else if (dtypeUB == "float16" || dtypeUB == "int16" || dtypeUB == "uint16") {
    block_size = 16;
  } else if (dtypeUB == "int8" || dtypeUB == "uint8") {
    block_size = 32;
  }

  return block_size;
}

bool Softmax::GetUbTilingInfo() {
  // rewrite ub_tiling_factor, ub_tiling_axis
  int64_t result = 1;
  if (pattern == 300000000) {
    if (tilingInfo.block_tiling_factor * input_shape[0] > compileInfo.max_ub_count) {
      tilingInfo.ub_tiling_axis = 1;
      tilingInfo.ub_tiling_factor = compileInfo.max_ub_count / input_shape[1];
    } else {
      tilingInfo.ub_tiling_axis = 1;
      tilingInfo.ub_tiling_factor = result;
    }
  } else if (pattern == 400000000) {
    tilingInfo.ub_tiling_factor = 1;
    tilingInfo.ub_tiling_axis = 0;
  } else if (pattern == 500000010 || pattern == 500000030) {
    if (pattern == 500000010) {
      result = input_shape[0] / tilingInfo.block_tiling_factor;
      if (result * input_shape[1] * input_shape[2] > compileInfo.max_ub_count) {
        if (input_shape[1] * input_shape[2] < compileInfo.max_ub_count) {
          tilingInfo.ub_tiling_axis = 0;
          tilingInfo.ub_tiling_factor = compileInfo.max_ub_count / input_shape[1] / input_shape[2];
        } else {
          tilingInfo.ub_tiling_axis = 2;
          tilingInfo.ub_tiling_factor = compileInfo.max_ub_count / input_shape[1];
          pattern = 500000020;
        }
      } else {
        tilingInfo.ub_tiling_axis = 2;
        tilingInfo.ub_tiling_factor = compileInfo.max_ub_count / input_shape[1];
      }
    }
  }

  return true;
}

bool Softmax::ProcessTiling() {
  // init
  tilingInfo.block_dim = 0;
  tilingInfo.block_tiling_axis = 0;
  tilingInfo.block_tiling_factor = 0;
  tilingInfo.ub_tiling_axis = 0;
  tilingInfo.ub_tiling_factor = 0;
  // rewrite TilingInfo(block)
  if (!GetBlockTilingInfo()) {
    return false;
  }
  // rewrite TilingInfo(ub)
  return GetUbTilingInfo();
}

bool Softmax::DoTiling() {
  /* Situations of DoTiling include:
     1. input(known):
        status of compile: do others except FusedReduceAxis
        status of runtime: do WriteTilingData
     2. input(unknown):
        do all process
  */
  OP_LOGD(op_type.c_str(), "tiling running...");
  bool ret = true;
  ret = ret && Init();
  ret = ret && FusedReduceAxis();
  // common process
  ret = ret && GetCompileInfo();

  if (pattern == 400000000) {
    ret = ret && ProcessTiling();
  }

  return ret;
}

bool Softmax::WriteTilingData() {
  run_info.block_dim = tilingInfo.block_dim;
  // tiling_key
  run_info.tiling_key = pattern;

  for (size_t i = 0; i < input_shape.size(); i++) {
    ByteBufferPut(run_info.tiling_data, (int32_t)input_shape[i]);
  }

  ByteBufferPut(run_info.tiling_data, (int32_t)tilingInfo.block_tiling_factor);
  ByteBufferPut(run_info.tiling_data, (int32_t)tilingInfo.ub_tiling_factor);
  OP_LOGD(op_type.c_str(), "block tilling axis=%d", tilingInfo.block_tiling_axis);
  OP_LOGD(op_type.c_str(), "block tilling factor=%d", tilingInfo.block_tiling_factor);
  OP_LOGD(op_type.c_str(), "ub tilling axis=%d", tilingInfo.ub_tiling_axis);
  OP_LOGD(op_type.c_str(), "ub tilling factor=%d", tilingInfo.ub_tiling_factor);
  OP_LOGD(op_type.c_str(), "block dim=%d", tilingInfo.block_dim);

  return true;
}

bool SoftmaxTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                  OpRunInfo& run_info) {
  Softmax softmax(op_type, op_paras, op_info, run_info);
  bool ret = true;
  ret = softmax.DoTiling();
  ret = ret && softmax.WriteTilingData();
  OP_LOGD(op_type.c_str(), "SoftmaxTiling end");
  return ret;
}

REGISTER_OP_TILING_FUNC_BUFFERED(SoftmaxV2, SoftmaxTiling);
REGISTER_OP_TILING_FUNC_BUFFERED(SoftmaxGrad, SoftmaxTiling);
REGISTER_OP_TILING_FUNC_BUFFERED(LogSoftmaxV2, SoftmaxTiling);
REGISTER_OP_TILING_FUNC_BUFFERED(LogSoftmaxGrad, SoftmaxTiling);
}  // namespace optiling