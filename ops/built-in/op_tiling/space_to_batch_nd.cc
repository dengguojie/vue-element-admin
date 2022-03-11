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
 * \file space_to_batch_nd.cc
 * \brief dynamic shape tiling of space_to_batch_nd
 */
#include <map>
#include "op_log.h"
#include "op_tiling.h"
#include "error_log.h"
#include "op_tiling_util.h"
#include <nlohmann/json.hpp>
#include "graph/utils/op_desc_utils.h"
#include "../op_proto/util/error_util.h"

namespace {
  constexpr int32_t FACTOR_TWO = 2;
  constexpr int32_t FACTOR_FOUR = 4;
  constexpr int32_t INPUT_W_LIMIT_1 = 65535;
  constexpr int32_t INPUT_W_LIMIT_2 = 4096;
  constexpr int32_t MODE_2 = 2;
  constexpr int32_t MODE_3 = 3;
  constexpr int32_t MODE_4 = 4;
  constexpr int32_t MODE_5 = 5;
  constexpr int32_t MODE_6 = 6;
  constexpr int32_t MODE_7 = 7;
  constexpr int32_t MODE_8 = 8;
  constexpr int32_t MODE_9 = 9;
  constexpr int32_t MODE_10 = 10;
  constexpr int32_t MODE_11 = 11;
  constexpr int32_t MODE_12 = 12;
  constexpr int32_t MODE_13 = 13;
  constexpr uint32_t SHAPE_LIMIT_5D = 5;
  constexpr uint32_t SHAPE_LIMIT_6D = 6;
  constexpr uint32_t SIZE_2 = 2;
  constexpr uint32_t SIZE_3 = 3;
  constexpr uint32_t SIZE_4 = 4;
  constexpr uint32_t SIZE_6 = 6;
  constexpr uint32_t SIZE_8 = 8;
  constexpr uint32_t SPACE_TO_BATCH_PADDING_INDEX = 1;
  constexpr uint32_t SPACE_TO_BATCH_ND_PADDING_INDEX = 2;
}  // namespace

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

static const std::vector<std::string> COMPILE_INFO_KEY = {"ub_ele", "core_num", "block_size"};

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

static void SetTilingParam(const TilingParam& param, utils::OpRunInfo& run_info) {
  run_info.AddTilingData(param.tiling_mode);
  run_info.AddTilingData(param.act_core_num);
  run_info.AddTilingData(param.one_core_ele);
  run_info.AddTilingData(param.last_core_ele);
  run_info.AddTilingData(param.input_b);
  run_info.AddTilingData(param.block_d);
  run_info.AddTilingData(param.block_h);
  run_info.AddTilingData(param.block_w);
  run_info.AddTilingData(param.pads_f);
  run_info.AddTilingData(param.pads_a);
  run_info.AddTilingData(param.pads_t);
  run_info.AddTilingData(param.pads_b);
  run_info.AddTilingData(param.pads_l);
  run_info.AddTilingData(param.pads_r);
  run_info.AddTilingData(param.input_d);
  run_info.AddTilingData(param.channel_one);
  run_info.AddTilingData(param.input_h);
  run_info.AddTilingData(param.input_w);
  run_info.AddTilingData(param.channel_zero);
  run_info.AddTilingData(param.output_b);
  run_info.AddTilingData(param.output_d);
  run_info.AddTilingData(param.output_h);
  run_info.AddTilingData(param.output_w);
}

static void CalCoreNum(TilingParam& param, int32_t total_ele, int32_t core_num) {
  param.one_core_ele = (total_ele + core_num - 1) / core_num;
  param.act_core_num = total_ele / param.one_core_ele;
  if (total_ele % param.one_core_ele != 0) {
    param.act_core_num = param.act_core_num + 1;
  }
  param.last_core_ele = total_ele - (param.act_core_num - 1) * param.one_core_ele;
}

static void CalTilingParam(TilingParam& param, const ge::GeShape& input_shape, const ge::Format& input_format,
                           int32_t ub_ele, int32_t core_num, const vector<int64_t>& block_vec,
                           const vector<int64_t>& pads_vec) {
  // calc input and output dim
  param.input_b = input_shape.GetDim(0);
  if (input_format == ge::FORMAT_NC1HWC0) {
    param.block_h = block_vec[0];
    param.block_w = block_vec[1];
    param.pads_t = pads_vec[0];
    param.pads_b = pads_vec[1];
    param.pads_l = pads_vec[2];
    param.pads_r = pads_vec[3];
    param.channel_one = input_shape.GetDim(1);
    param.input_h = input_shape.GetDim(2);
    param.input_w = input_shape.GetDim(3);
    param.channel_zero = input_shape.GetDim(4);
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
    param.input_d = input_shape.GetDim(1);
    param.channel_one = input_shape.GetDim(2);
    param.input_h = input_shape.GetDim(3);
    param.input_w = input_shape.GetDim(4);
    param.channel_zero = input_shape.GetDim(5);
    param.output_b = param.input_b * param.block_d * param.block_h * param.block_w;
    param.output_d = (param.input_d + param.pads_f + param.pads_a) / param.block_d;
    param.output_h = (param.input_h + param.pads_t + param.pads_b) / param.block_h;
    param.output_w = (param.input_w + param.pads_l + param.pads_r) / param.block_w;
  }

  // select tiling mode
  if (param.output_h * param.output_w * param.block_w * param.channel_zero <= ub_ele / FACTOR_FOUR &&
      (param.block_h - 1) * param.input_w * FACTOR_TWO <= INPUT_W_LIMIT_1) {
    param.tiling_mode = input_format == ge::FORMAT_NC1HWC0 ? 0 : MODE_6;
  } else if (param.output_h * param.output_w * param.block_w * param.channel_zero <= ub_ele / FACTOR_TWO &&
             (param.block_h - 1) * param.input_w * FACTOR_TWO <= INPUT_W_LIMIT_1) {
    param.tiling_mode = input_format == ge::FORMAT_NC1HWC0 ? 1 : MODE_7;
  } else if (param.output_w * param.block_w * param.channel_zero <= ub_ele / FACTOR_FOUR) {
    param.tiling_mode = input_format == ge::FORMAT_NC1HWC0 ? MODE_2 : MODE_8;
  } else if (param.output_w * param.block_w * param.channel_zero <= ub_ele / FACTOR_TWO) {
    param.tiling_mode = input_format == ge::FORMAT_NC1HWC0 ? MODE_3 : MODE_9;
  } else if (param.output_w * param.channel_zero <= ub_ele && param.output_w < INPUT_W_LIMIT_2) {
    param.tiling_mode = input_format == ge::FORMAT_NC1HWC0 ? MODE_4 : MODE_10;
  } else {
    param.tiling_mode = input_format == ge::FORMAT_NC1HWC0 ? MODE_5 : MODE_11;
  }

  // calc act core_num
  if (input_format == ge::FORMAT_NC1HWC0) {
    CalCoreNum(param, param.input_b * param.channel_one, core_num);
  } else {
    CalCoreNum(param, param.output_d, core_num);
  }

  // when slect branch 2 or 3, calc core at output_h
  if ((param.tiling_mode == 2 || param.tiling_mode == MODE_3) && (param.output_h > param.input_b * param.channel_one)) {
    param.tiling_mode = param.tiling_mode == MODE_2 ? MODE_12 : MODE_13;
    CalCoreNum(param, param.output_h, core_num);
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
bool SpaceToBatchNDTiling(const std::string& op_type, const ge::Operator& opParas, const std::vector<int64_t>& op_info,
                          utils::OpRunInfo& run_info) {
  OP_LOGI(op_type.c_str(), "SpaceToBatchNDTiling running.");
  // get and check input shape
  auto operator_info = OpDescUtils::GetOpDescFromOperator(opParas);
  if (operator_info == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get op_info failed.");
    return false;
  }

  auto input_x_desc = operator_info->MutableInputDesc(0);
  if (input_x_desc == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_x_desc failed.");
    return false;
  }

  ge::GeShape& input_shape = input_x_desc->MutableShape();
  ge::Format input_format = input_x_desc->GetFormat();
  ge::Format ori_format = input_x_desc->GetOriginFormat();

  if ((input_format != ge::FORMAT_NC1HWC0) && (input_format != ge::FORMAT_NDC1HWC0)) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get input format failed, only support NC1HWC0 and NDC1HWC0, but got %d.",
                                    input_format);
    return false;
  }
  if ((input_format == ge::FORMAT_NC1HWC0) && (input_shape.GetDimNum() != SHAPE_LIMIT_5D)) {
    VECTOR_INNER_ERR_REPORT_TILIING(
        op_type, "Get input shape failed at format NC1HWC0, the length of input shape must be 5, but got %lu.",
        input_shape.GetDimNum());
    return false;
  }
  if ((input_format == ge::FORMAT_NDC1HWC0) && (input_shape.GetDimNum() != SHAPE_LIMIT_6D)) {
    VECTOR_INNER_ERR_REPORT_TILIING(
        op_type, "Get input shape failed at format NDC1HWC0, the length of input shape must be 6, but got %lu.",
        input_shape.GetDimNum());
    return false;
  }

  // get compile info
  OP_TILING_CHECK(
      op_info.size() != COMPILE_INFO_KEY.size(),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "the compile info num is not equal expect compile_info(%zu), is %zu",
                                      COMPILE_INFO_KEY.size(), op_info.size()),
      return false);
  int32_t ub_ele = static_cast<int32_t>(op_info[0]);
  int32_t core_num = static_cast<int32_t>(op_info[1]);
  int32_t block_size = static_cast<int32_t>(op_info[2]);

  vector<int64_t> block_vec;
  vector<int64_t> pads_vec;
  
  // calc block_vec and pads_vec and check supported
  // cppcheck-suppress *
  // the parameters order in op proto is: x, block_shape, paddings, y
  int64_t paddings_size_index = SPACE_TO_BATCH_ND_PADDING_INDEX;
  if (op_type == "SpaceToBatch") {
    // the parameters order in op proto is: x, paddings, y
    paddings_size_index = SPACE_TO_BATCH_PADDING_INDEX;
  }
  
  if (block_size != 0) {
    block_vec.push_back(block_size);
    block_vec.push_back(block_size);

    if (!ops::GetConstIntData(opParas, paddings_size_index, pads_vec)) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get const size failed!");
      return false;
    }
    if (pads_vec.size() == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "const_inputs not include paddings");
      return false;
    }
  } else {
    static const int64_t block_size_index = 1;
    if (!ops::GetConstIntData(opParas, block_size_index, block_vec)) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get const block_vec size failed!");
      return false;
    }
    if (block_vec.size() == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "const_inputs not include block_vec");
      return false;
    }

    // the parameters order in op proto is: x, block_shape, paddings, y    
    if (!ops::GetConstIntData(opParas, paddings_size_index, pads_vec)) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get const pads_vec size failed!");
      return false;
    }
    if (pads_vec.size() == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "const_inputs not include paddings");
      return false;
    }
  }

  // check and resize block_shape and paddings
  if (input_format == ge::FORMAT_NC1HWC0) {
    if ((ori_format == ge::FORMAT_NHWC) && (block_vec.size() == 1) && (pads_vec.size() == SIZE_2)) {
      block_vec.push_back(1);
      pads_vec.push_back(0);
      pads_vec.push_back(0);
    } else if (((ori_format == ge::FORMAT_NHWC) || (ori_format == ge::FORMAT_NCHW)) && (block_vec.size() == SIZE_2) &&
               (pads_vec.size() == SIZE_4)) {
      ;
    } else if ((ori_format == ge::FORMAT_NCHW) && (block_vec.size() == SIZE_3) && (pads_vec.size() == SIZE_6) &&
               (block_vec[0] == 1) && (pads_vec[0] == 0) && (pads_vec[1] == 0)) {
      block_vec.erase(block_vec.begin());
      pads_vec.erase(pads_vec.begin(), pads_vec.begin() + FACTOR_TWO);
    } else {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type,
                                      "Input with format NC1HWC0 which does not meet the rules, ori_format is %d, "
                                      "block size is %lu, pads size is %lu",
                                      ori_format, block_vec.size(), pads_vec.size());
      return false;
    }
  } else {
    if (((ori_format == ge::FORMAT_NDHWC) || (ori_format == ge::FORMAT_NCDHW)) && (block_vec.size() == SIZE_3) &&
        (pads_vec.size() == SIZE_6)) {
      ;
    } else if ((ori_format == ge::FORMAT_NCDHW) && (block_vec.size() == SIZE_4) && (pads_vec.size() == SIZE_8) &&
               (block_vec[0] == 1) && (pads_vec[0] == 0) && (pads_vec[1] == 0)) {
      block_vec.erase(block_vec.begin());
      pads_vec.erase(pads_vec.begin(), pads_vec.begin() + FACTOR_TWO);
    } else {
      VECTOR_INNER_ERR_REPORT_TILIING(
          op_type,
          "Input with format NDC1HWC0 which does not meet the rules, ori_format is %d, block size is %lu, pads size "
          "is %lu",
          ori_format, block_vec.size(), pads_vec.size());
      return false;
    }
  }

  // check block_shape and paddings
  if (input_format == ge::FORMAT_NC1HWC0) {
    if ((block_vec[0] <= 0) || (block_vec[1] <= 0)) {
      VECTOR_INNER_ERR_REPORT_TILIING(
          op_type,
          "Get block_shape failed at format NC1HWC0, the value of block_shape must be greater to 0, but "
          "got [%ld, %ld].",
          block_vec[0], block_vec[1]);
      return false;
    }
    if ((pads_vec[0] < 0) || (pads_vec[1] < 0) || (pads_vec[2] < 0) || (pads_vec[3] < 0)) {
      VECTOR_INNER_ERR_REPORT_TILIING(
          op_type,
          "Get pads failed at format NC1HWC0, the value of pads must be greater and equal to 0, but "
          "got [%ld, %ld, %ld, %ld].",
          pads_vec[0], pads_vec[1], pads_vec[2], pads_vec[3]);
      return false;
    }
    if ((input_shape.GetDim(2) + pads_vec[0] + pads_vec[1]) % block_vec[0] != 0 ||
        (input_shape.GetDim(3) + pads_vec[2] + pads_vec[3]) % block_vec[1] != 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type,
                                      "The (input+pads)/(block_shape) should be integer, but got input:[%ld, %ld], "
                                      "block:[%ld, %ld], pads:[%ld, %ld, "
                                      "%ld, %ld]",
                                      input_shape.GetDim(2), input_shape.GetDim(3), block_vec[0], block_vec[1],
                                      pads_vec[0], pads_vec[1], pads_vec[2], pads_vec[3]);
      return false;
    }
  } else {
    if ((block_vec[0] <= 0) || (block_vec[1] <= 0) || (block_vec[2] <= 0)) {
      VECTOR_INNER_ERR_REPORT_TILIING(
          op_type,
          "Get block_shape failed at format NDC1HWC0, the value of block_shape must be greater to 0, but "
          "got [%ld, %ld, %ld].",
          block_vec[0], block_vec[1], block_vec[2]);
      return false;
    }
    if ((pads_vec[0] < 0) || (pads_vec[1] < 0) || (pads_vec[2] < 0) || (pads_vec[3] < 0) || (pads_vec[4] < 0) ||
        (pads_vec[5] < 0)) {
      VECTOR_INNER_ERR_REPORT_TILIING(
          op_type,
          "Get pads failed at format NDC1HWC0, the value of pads must be greater and equal 0, but "
          "got [%ld, %ld, %ld, %ld, %ld, %ld].",
          pads_vec[0], pads_vec[1], pads_vec[2], pads_vec[3], pads_vec[4], pads_vec[5]);
      return false;
    }
    if ((input_shape.GetDim(1) + pads_vec[0] + pads_vec[1]) % block_vec[0] != 0 ||
        (input_shape.GetDim(3) + pads_vec[2] + pads_vec[3]) % block_vec[1] != 0 ||
        (input_shape.GetDim(4) + pads_vec[4] + pads_vec[5]) % block_vec[2] != 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(
          op_type,
          "The (input+pads)/(block_shape) should be integer, but got input:[%ld, %ld, %ld], block:[%ld, %ld, %ld], "
          "pads:[%ld, %ld, %ld, %ld, %ld, %ld]",
          input_shape.GetDim(1), input_shape.GetDim(3), input_shape.GetDim(4), block_vec[0], block_vec[1], block_vec[2],
          pads_vec[0], pads_vec[1], pads_vec[2], pads_vec[3], pads_vec[4], pads_vec[5]);
      return false;
    }
  }

  // if input_h and block_h is one, can swap h and w
  if (input_format == ge::FORMAT_NC1HWC0 && input_shape.GetDim(2) == 1 && block_vec[0] == 1 && pads_vec[0] == 0 &&
      pads_vec[1] == 0) {
    int64_t temp_value = input_shape.GetDim(2);
    input_shape.SetDim(2, input_shape.GetDim(3));
    input_shape.SetDim(3, temp_value);
    std::swap(block_vec[0], block_vec[1]);
    std::swap(pads_vec[0], pads_vec[2]);
    std::swap(pads_vec[1], pads_vec[3]);
  }

  // calc tiling params, set tiling params, print tiling params
  TilingParam param;
  CalTilingParam(param, input_shape, input_format, ub_ele, core_num, block_vec, pads_vec);
  SetTilingParam(param, run_info);
  PrintTilingParam(param);

  // block_dim, use fot tik op; workspace, null for tik op
  run_info.SetBlockDim(param.act_core_num);

  OP_LOGI(op_type.c_str(), "SpaceToBatchNDTiling run success.");
  return true;
}

// register tiling interface of space_to_batch and space_to_batch_nd op.
REGISTER_OP_TILING_V3_WITH_VECTOR(SpaceToBatch, SpaceToBatchNDTiling, COMPILE_INFO_KEY, NO_OPTIONAL_VALUE);
REGISTER_OP_TILING_V3_WITH_VECTOR(SpaceToBatchND, SpaceToBatchNDTiling, COMPILE_INFO_KEY, NO_OPTIONAL_VALUE);
}  // namespace optiling
