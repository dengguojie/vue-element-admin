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
 * \file ascend_anti_quant.cc
 * \brief dynamic shape tiling of ascend_anti_quant
 */
#include <map>
#include <nlohmann/json.hpp>

#include "../op_proto/util/error_util.h"
#include "error_log.h"
#include "op_log.h"
#include "op_tiling.h"
#include "external/graph/operator.h"
#include "error_log.h"

using namespace ge;

namespace {
  constexpr int32_t ALIGN_BLOCK = 32;
}

namespace optiling {
struct TilingParams {
  int32_t block_dim{0};
  int32_t block_tiling_axis{-1};
  int64_t block_factor{1};
  int32_t ub_tiling_axis{-1};
  int64_t ub_factor{1};
  int32_t is_fuse_block{1};
};

struct CompileInfo {
  int64_t max_ub_count{0};
  int32_t core_num{1};
};

static void PrintTilingParams(const TilingParams &param) {
  OP_LOGD("ascend_anti_quant_tiling",
          "(block_dim,block_tiling_axis,block_factor,ub_tiling_axis,ub_factor):(%d,%d,%d,%d,%d)",
          param.block_dim, param.block_tiling_axis, param.block_factor, param.ub_tiling_axis, param.ub_factor);
}

bool GetComInfo(const std::string &op_type,
                const ge::Operator &op_paras,
                const nlohmann::json &op_info,
                utils::OpRunInfo &run_info,
                CompileInfo &compile_info) {
  std::vector<int32_t> info = op_info["common_info"];
  compile_info.max_ub_count = info[0];
  compile_info.core_num = info[1];

  return true;
}

int32_t GetBlockSize(ge::DataType dtype) {
  int32_t block_size = 0;
  int32_t type_size = GetSizeByDataType(dtype);
  if (type_size > 0) {
    block_size = ALIGN_BLOCK / type_size;
  }

  return block_size;
}

int32_t GetBlockNum(std::vector<int64_t> &out_shape,
                    int32_t tiling_axis,
                    int64_t tiling_n_parts) {
  int32_t block_dim = 1;
  for (int32_t i = 0; i <= tiling_axis; i++) {
    if (out_shape[i] != 0) {
      if (i == tiling_axis) {
        block_dim = (int32_t)tiling_n_parts * block_dim;
      } else {
        block_dim = (int32_t)out_shape[i] * block_dim;
      }
    }
  }

  return block_dim;
}

void GetUbTilingParam(TilingParams &param,
                      int32_t block_tiling_axis,
                      int64_t block_factor,
                      std::vector<int64_t> &out_shape,
                      int64_t max_ub_size) {
  int64_t ub_size = 1;
  int32_t ub_tiling_axis = 0;
  int64_t ub_outer = 1;
  int64_t ub_factor = 1;
  bool is_need_split_block_factor = true;

  for (int64_t j = out_shape.size() - 1; j > block_tiling_axis; j--) {
    ub_size *= out_shape[j];
    if (ub_size > max_ub_size) {
      ub_tiling_axis = j;
      ub_outer = ub_size % max_ub_size == 0 ? ub_size / max_ub_size : (ub_size + max_ub_size - 1) / max_ub_size;
      ub_factor = out_shape[j] % ub_outer == 0 ? out_shape[j] / ub_outer : (out_shape[j] + ub_outer - 1) / ub_outer;
      is_need_split_block_factor = false;
      break;
    }
  }

  if (is_need_split_block_factor) {
    ub_size *= block_factor;
    ub_tiling_axis = block_tiling_axis;
    if (ub_size > max_ub_size) {
      ub_outer = ub_size % max_ub_size == 0 ? ub_size / max_ub_size : (ub_size + max_ub_size - 1) / max_ub_size;
    }
    ub_factor = block_factor % ub_outer == 0 ? block_factor / ub_outer : (block_factor + ub_outer - 1) / ub_outer;
  }

  param.ub_tiling_axis = ub_tiling_axis;
  param.ub_factor = ub_factor;
}

void GetTilingData(TilingParams& param,
                   std::vector<int64_t> &out_shape,
                   CompileInfo &compile_info,
                   const ge::DataType input_dtype) {
  int64_t core_limit = out_shape.size() - 3;
  int32_t block_tiling_axis = core_limit - 1;
  int64_t block_factor = 1;
  int64_t n_parts = out_shape[core_limit - 1];
  int64_t core_size = 1;
  int32_t core_num = compile_info.core_num;
  int32_t is_fuse_block = 1;

  for (int64_t i = 0; i < core_limit; i++) {
    core_size *= out_shape[i];
    if (out_shape[i] >= core_num && i == 1) {
      is_fuse_block = 0;
      block_tiling_axis = i;
      n_parts = core_num;
      block_factor = (out_shape[i] + core_num - 1) / core_num;
      break;
    }
    if (core_size >= core_num) {
      int64_t left_block_dim = core_size / out_shape[i];
      int64_t cur_block_dim = core_num / left_block_dim;
      int64_t cur_block_factor = (out_shape[i] + cur_block_dim - 1) / cur_block_dim;
      n_parts = cur_block_dim;
      block_tiling_axis = i;
      block_factor = cur_block_factor;
      break;
    }
  }

  param.block_tiling_axis = block_tiling_axis;
  param.is_fuse_block = is_fuse_block;
  if (is_fuse_block > 0) {
    param.block_factor = n_parts;
    param.block_dim = GetBlockNum(out_shape, block_tiling_axis, n_parts);
  } else {
    param.block_factor = n_parts;
    param.block_dim = n_parts;
  }
  int32_t dtype_size = GetBlockSize(input_dtype);
  OP_TILING_CHECK(dtype_size == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("AscendAntiQuant", "dtype_size cannot be zero."),
                  return);
  int64_t max_ub_size = compile_info.max_ub_count / dtype_size;
  GetUbTilingParam(param, block_tiling_axis, block_factor, out_shape, max_ub_size);
}

int32_t CalcPatternKey(std::vector<int64_t> shape,
                       int32_t block_tiling_axis,
                       int32_t ub_tiling_axis) {
  int32_t pattern = 0;
  for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); i++) {
    pattern += pow(2, (shape.size() - 1 - i));
  }
  pattern += block_tiling_axis * 100 + ub_tiling_axis * 10;

  return pattern;
}

int32_t CalcTilingKey(std::vector<int64_t> shape,
                      TilingParams &param) {
  int32_t key = 0;
  int32_t block_tiling_axis = param.block_tiling_axis;
  int32_t ub_tiling_axis = param.ub_tiling_axis;
  int32_t is_fuse_block = param.is_fuse_block;
  int32_t pattern = CalcPatternKey(shape, block_tiling_axis, ub_tiling_axis);
  std::vector<int32_t> val = {1000000000, 10000000, 1000000, 100000, 10000, 1000};
  std::vector<int32_t> pos = {0, is_fuse_block, 0, block_tiling_axis, ub_tiling_axis, pattern};
  for (int64_t i = 0; i < static_cast<int64_t>(pos.size()); i++) {
    key += pos[i] * val[i];
  }

  return key;
}

bool AscendAntiQuantTiling(const std::string &op_type,
                           const ge::Operator &op_paras,
                           const nlohmann::json &op_info,
                           utils::OpRunInfo &run_info) {
  OP_LOGI(op_type.c_str(), "AscendAntiQuantTiling running.");

  ge::Format input_format = op_paras.GetInputDesc(0).GetFormat();
  ge::Shape input_x = op_paras.GetInputDesc(0).GetShape();
  const ge::DataType output_dtype = op_paras.GetOutputDesc(0).GetDataType();
  OP_TILING_CHECK(input_format != FORMAT_NC1HWC0,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type,
                          "input format only support NC1HWC0, but got %d.", input_format),
                  return false);

  CompileInfo compile_info;
  bool compile_flag = GetComInfo(op_type, op_paras, op_info, run_info, compile_info);
  if (!compile_flag) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetCompileInfo failed.");
  }

  std::vector<int64_t> input_x_new;
  std::vector<int64_t> input_y;
  int64_t c1 = input_x.GetDim(1) * 2;
  int64_t hw = input_x.GetDim(2) * input_x.GetDim(3);

  input_y.push_back(input_x.GetDim(0));
  input_y.push_back(c1);
  input_y.push_back(hw);
  input_y.push_back(16);

  input_x_new.push_back(input_x.GetDim(0));
  input_x_new.push_back(input_x.GetDim(1));
  input_x_new.push_back(hw);
  input_x_new.push_back(input_x.GetDim(4));

  TilingParams tiling_params;
  GetTilingData(tiling_params, input_y, compile_info, output_dtype);

  // tiling_key
  int32_t tiling_key = CalcTilingKey(input_y, tiling_params);
  run_info.SetBlockDim(tiling_params.block_dim);
  run_info.SetTilingKey(tiling_key);

  for (int64_t i = 0; i < static_cast<int64_t>(input_x_new.size()) - 1; i++) {
    run_info.AddTilingData((int32_t)input_x_new[i]);
    OP_LOGD(op_type.c_str(), "input_x_new shape:%d", input_x_new[i]);
  }
  run_info.AddTilingData((int32_t)tiling_params.block_factor);
  run_info.AddTilingData((int32_t)tiling_params.ub_factor);
  OP_LOGD(op_type.c_str(), "block factor=%d", tiling_params.block_factor);
  OP_LOGD(op_type.c_str(), "ub factor=%d", tiling_params.ub_factor);

  PrintTilingParams(tiling_params);

  OP_LOGI(op_type.c_str(), "AscendAntiQuantTiling end.");
  return true;
}

// register tiling interface of AscendAntiQuant op.
REGISTER_OP_TILING_FUNC_BUFFERED_V2(AscendAntiQuant, AscendAntiQuantTiling);
}  // namespace optiling
