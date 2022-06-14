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
 * \file batch_to_space_nd.cc
 * \brief dynamic shape tiling of batch_to_space_nd
 */
#include <map>
#include <nlohmann/json.hpp>

#include "error_util.h"
#include "op_log.h"
#include "op_tiling_util.h"
#include "op_const.h"
#include "error_log.h"

#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"
namespace optiling {
using namespace ge;
using namespace std;

// define the compile key of json.vars
static const std::vector<std::string> COMPILE_INFO_KEY = {"ub_ele", "core_num"};
// BatchToSpace attr index
const struct ops::AttrBase BLOCK_SIZE_DIMS_INFO(0, "block_size");
const int64_t INDEX_0 = 0;
const int64_t INDEX_1 = 1;
const int64_t INDEX_2 = 2;
const int64_t INDEX_3 = 3;
const int64_t INDEX_4 = 4;
const int64_t INDEX_5 = 5;
const int64_t SIZE_2 = 2;
const int64_t SIZE_3 = 3;
const int64_t SIZE_4 = 4;
const int64_t SIZE_6 = 6;
const int64_t SIZE_8 = 8;
const int64_t MODE_2 = 2;
const int64_t MODE_3 = 3;
const int64_t MODE_4 = 4;
const int64_t MODE_5 = 5;
const int64_t MODE_6 = 6;
const int64_t MODE_7 = 7;
const int64_t MODE_8 = 8;
const int64_t MODE_9 = 9;
const int64_t MODE_10 = 10;
const int64_t MODE_11 = 11;

struct TilingParam {
  int32_t tiling_mode = 0;
  int32_t act_core_num = 0;
  int32_t one_core_ele = 0;
  int32_t last_core_ele = 0;
  int32_t input_b = 0;
  int32_t block_d = 0;  // block at depth
  int32_t block_h = 0;  // block at height
  int32_t block_w = 0;  // block at width
  int32_t crops_f = 0;  // front
  int32_t crops_a = 0;  // after
  int32_t crops_t = 0;  // top
  int32_t crops_b = 0;  // bottom
  int32_t crops_l = 0;  // left
  int32_t crops_r = 0;  // right
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
  OP_LOGD("BatchToSpaceNDTiling",
          "(tiling_mode,act_core_num,one_core_ele,last_core_ele,input_b,block_d,block_h,block_w,crops_f,crops_a,"
          "crops_t,crops_b,crops_l,crops_r,input_d,channel_one,input_h,input_w,channel_zero,output_b,output_d,output_h,"
          "output_w):(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d)",
          param.tiling_mode, param.act_core_num, param.one_core_ele, param.last_core_ele, param.input_b, param.block_d,
          param.block_h, param.block_w, param.crops_f, param.crops_a, param.crops_t, param.crops_b, param.crops_l,
          param.crops_r, param.input_d, param.channel_one, param.input_h, param.input_w, param.channel_zero,
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
  run_info.AddTilingData(param.crops_f);
  run_info.AddTilingData(param.crops_a);
  run_info.AddTilingData(param.crops_t);
  run_info.AddTilingData(param.crops_b);
  run_info.AddTilingData(param.crops_l);
  run_info.AddTilingData(param.crops_r);
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
  OP_TILING_CHECK(core_num == 0, VECTOR_INNER_ERR_REPORT_TILIING("batch_to_space_nd", "core_num = 0 is not support"),
                  return);
  param.one_core_ele = (total_ele + core_num - 1) / core_num;
  param.act_core_num = total_ele / param.one_core_ele;
  if (total_ele % param.one_core_ele != 0) {
    param.act_core_num = param.act_core_num + 1;
  }
  param.last_core_ele = total_ele - (param.act_core_num - 1) * param.one_core_ele;
}

static void CalTilingParam(TilingParam& param, const GeShape& input_shape, ge::Format input_format, int32_t ub_ele,
                           int32_t core_num, const vector<int64_t>& block_vec, const vector<int64_t>& crops_vec,
                           bool swap_h_w) {
  // calc input and output dim
  param.input_b = input_shape.GetDim(INDEX_0);
  if (input_format == FORMAT_NC1HWC0) {
    param.block_h = block_vec[INDEX_0];
    param.block_w = block_vec[INDEX_1];
    param.crops_t = crops_vec[INDEX_0];
    param.crops_b = crops_vec[INDEX_1];
    param.crops_l = crops_vec[INDEX_2];
    param.crops_r = crops_vec[INDEX_3];
    param.channel_one = input_shape.GetDim(INDEX_1);
    param.input_h = input_shape.GetDim(INDEX_2);
    param.input_w = input_shape.GetDim(INDEX_3);
    if (swap_h_w) {
      param.input_h = input_shape.GetDim(INDEX_3);
      param.input_w = input_shape.GetDim(INDEX_2);
    }
    param.channel_zero = input_shape.GetDim(INDEX_4);
    param.output_b = param.input_b / param.block_h / param.block_w;
    param.output_h = param.input_h * param.block_h - param.crops_t - param.crops_b;
    param.output_w = param.input_w * param.block_w - param.crops_l - param.crops_r;
  } else {
    param.block_d = block_vec[INDEX_0];
    param.block_h = block_vec[INDEX_1];
    param.block_w = block_vec[INDEX_2];
    param.crops_f = crops_vec[INDEX_0];
    param.crops_a = crops_vec[INDEX_1];
    param.crops_t = crops_vec[INDEX_2];
    param.crops_b = crops_vec[INDEX_3];
    param.crops_l = crops_vec[INDEX_4];
    param.crops_r = crops_vec[INDEX_5];
    param.input_d = input_shape.GetDim(INDEX_1);
    param.channel_one = input_shape.GetDim(INDEX_2);
    param.input_h = input_shape.GetDim(INDEX_3);
    if (swap_h_w) {
      param.channel_one = input_shape.GetDim(INDEX_3);
      param.input_h = input_shape.GetDim(INDEX_2);
    }
    param.input_w = input_shape.GetDim(INDEX_4);
    param.channel_zero = input_shape.GetDim(INDEX_5);
    param.output_b = param.input_b / param.block_d / param.block_h / param.block_w;
    param.output_d = param.input_d * param.block_d - param.crops_f - param.crops_a;
    param.output_h = param.input_h * param.block_h - param.crops_t - param.crops_b;
    param.output_w = param.input_w * param.block_w - param.crops_l - param.crops_r;
  }

  // select tiling mode
  if (param.input_h * param.input_w * param.block_w * param.channel_zero <= ub_ele / 4 &&
      (param.block_h - 1) * param.output_w * 2 <= 65535) {
    param.tiling_mode = input_format == FORMAT_NC1HWC0 ? 0 : MODE_6;
  } else if (param.input_h * param.input_w * param.block_w * param.channel_zero <= ub_ele / 2 &&
             (param.block_h - 1) * param.output_w * 2 <= 65535) {
    param.tiling_mode = input_format == FORMAT_NC1HWC0 ? 1 : MODE_7;
  } else if (param.input_w * param.block_w * param.channel_zero <= ub_ele / 4) {
    param.tiling_mode = input_format == FORMAT_NC1HWC0 ? MODE_2 : MODE_8;
  } else if (param.input_w * param.block_w * param.channel_zero <= ub_ele / 2) {
    param.tiling_mode = input_format == FORMAT_NC1HWC0 ? MODE_3 : MODE_9;
  } else if (param.input_w * param.channel_zero <= ub_ele && param.input_w < 4096) {
    param.tiling_mode = input_format == FORMAT_NC1HWC0 ? MODE_4 : MODE_10;
  } else {
    param.tiling_mode = input_format == FORMAT_NC1HWC0 ? MODE_5 : MODE_11;
  }

  // calc act core_num
  if (input_format == FORMAT_NC1HWC0) {
    CalCoreNum(param, param.output_b * param.channel_one, core_num);
  } else {
    CalCoreNum(param, param.input_d, core_num);
  }

  // when slect branch 2 or 3, calc core at input_h
  if ((param.tiling_mode == 2 || param.tiling_mode == 3) && (param.input_h > param.output_b * param.channel_one)) {
    param.tiling_mode = param.tiling_mode == 2 ? 12 : 13;
    CalCoreNum(param, param.input_h, core_num);
  }
}

/*
 * @brief: tiling function of op
 * @param [in] op_type: type of the op
 * @param [in] op_paras: inputs/outputs/attrs of the op
 * @param [in] op_info: compile time generated info of the op
 * @param [out] run_info: result data
 * @return bool: success or not success
 */
bool BatchToSpaceNDTiling(const string& op_type, const ge::Operator& op_paras, const std::vector<int64_t>& op_info,
                          utils::OpRunInfo& run_info) {
  OP_LOGI(op_type.c_str(), "BatchToSpaceNDTiling running.");
  PROFILING_TILING_INIT(op_type.c_str());
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get OpDesc failed."),
                  return false);

  // get and check input shape
  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get InputDesc failed."),
                  return false);

  const GeShape& input_shape = input_desc->MutableShape();
  ge::Format input_format = input_desc->GetFormat();
  ge::Format ori_format = input_desc->GetOriginFormat();
  PROFILING_TILING_AFTER_GET_SHAPE_REG();
  if ((input_format != FORMAT_NC1HWC0) && (input_format != FORMAT_NDC1HWC0)) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get input format failed, only support NC1HWC0 and NDC1HWC0, but got %s.",
                                    to_string(input_format).c_str());
    return false;
  }
  if (input_format == FORMAT_NC1HWC0) {
    OP_TILING_CHECK(
        input_shape.GetDimNum() != 5,
        VECTOR_INNER_ERR_REPORT_TILIING(
            op_type, "Get input shape failed at format NC1HWC0, the length of input shape must be 5, but got %lu.",
            input_shape.GetDimNum()),
        return false);
  }
  if (input_format == FORMAT_NDC1HWC0) {
    OP_TILING_CHECK(
        input_shape.GetDimNum() != 6,
        VECTOR_INNER_ERR_REPORT_TILIING(
            op_type, "Get input shape failed at format NC1HWC0, the length of input shape must be 6, but got %lu.",
            input_shape.GetDimNum()),
        return false);
  }

  // get compile info
  OP_TILING_CHECK(
      op_info.size() != COMPILE_INFO_KEY.size(),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "the compile info num is not equal expect compile_info(%zu), is %zu",
                                      COMPILE_INFO_KEY.size(), op_info.size()),
      return false);

  int32_t ub_ele = static_cast<int32_t>(op_info[INDEX_0]);
  int32_t core_num = static_cast<int32_t>(op_info[INDEX_1]);
  int32_t block_size = 0;
  if (op_type == "BatchToSpace") {
    OP_TILING_CHECK(!ops::GetAttrValue(op_paras, BLOCK_SIZE_DIMS_INFO, block_size),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get block_size values failed."),
                    return false);
  }
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  std::vector<int64_t> block_vec{block_size, block_size};
  std::vector<int64_t> crops_vec;
  // calc block_vec and crops_vec and check supported
  // cppcheck-suppress *
  if (block_size != 0) {
    // BatchToSpace
    // input crops index is 1
    OP_TILING_CHECK(!ops::GetConstIntData(op_paras, 1, crops_vec),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get crops values failed."), return false);
  } else {
    // BatchToSpaceND
    // input crops index is 2
    OP_TILING_CHECK(!ops::GetConstIntData(op_paras, 2, crops_vec),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get crops values failed."), return false);
    // input block_shape index is 1
    OP_TILING_CHECK(!ops::GetConstIntData(op_paras, 1, block_vec),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get block_shape values failed."), return false);
  }

  // check and resize block_shape and crops
  if (input_format == FORMAT_NC1HWC0) {
    if ((ori_format == FORMAT_NHWC) && (block_vec.size() == 1) && (crops_vec.size() == SIZE_2)) {
      block_vec.push_back(1);
      crops_vec.push_back(0);
      crops_vec.push_back(0);
    } else if (((ori_format == FORMAT_NHWC) || (ori_format == FORMAT_NCHW)) && (block_vec.size() == SIZE_2) &&
               (crops_vec.size() == SIZE_4)) {
      ;
    } else if ((ori_format == FORMAT_NCHW) && (block_vec.size() == SIZE_3) && (crops_vec.size() == SIZE_6) &&
               (block_vec[INDEX_0] == 1) && (crops_vec[INDEX_0] == 0) && (crops_vec[INDEX_1] == 0)) {
      block_vec.erase(block_vec.begin());
      crops_vec.erase(crops_vec.begin(), crops_vec.begin() + 2);
    } else {
      VECTOR_INNER_ERR_REPORT_TILIING(
          op_type,
          "Input with format NC1HWC0 which does not meet the rules, ori_format is %s, block size is %lu, crops size "
          "is %lu",
          to_string(ori_format).c_str(), block_vec.size(), crops_vec.size());
      return false;
    }
  } else {
    if (((ori_format == FORMAT_NDHWC) || (ori_format == FORMAT_NCDHW)) && (block_vec.size() == SIZE_3) &&
        (crops_vec.size() == SIZE_6)) {
      ;
    } else if ((ori_format == FORMAT_NCDHW) && (block_vec.size() == SIZE_4) && (crops_vec.size() == SIZE_8) &&
               (block_vec[INDEX_0] == 1) && (crops_vec[INDEX_0] == 0) && (crops_vec[INDEX_1] == 0)) {
      block_vec.erase(block_vec.begin());
      crops_vec.erase(crops_vec.begin(), crops_vec.begin() + 2);
    } else {
      VECTOR_INNER_ERR_REPORT_TILIING(
          op_type,
          "Input with format NDC1HWC0 which does not meet the rules, ori_format is %s, block size is %lu, crops "
          "size is %lu",
          to_string(ori_format).c_str(), block_vec.size(), crops_vec.size());
      return false;
    }
  }

  // check block_shape and crops
  if (input_format == FORMAT_NC1HWC0) {
    if ((block_vec[INDEX_0] <= 0) || (block_vec[INDEX_1] <= 0)) {
      VECTOR_INNER_ERR_REPORT_TILIING(
          op_type,
          "Get block_shape failed at format NC1HWC0, the value of block_shape must be greater to 0, but "
          "got [%ld, %ld].",
          block_vec[INDEX_0], block_vec[INDEX_1]);
      return false;
    }
    if ((crops_vec[INDEX_0] < 0) || (crops_vec[INDEX_1] < 0) || (crops_vec[INDEX_2] < 0) || (crops_vec[INDEX_3] < 0)) {
      VECTOR_INNER_ERR_REPORT_TILIING(
          op_type,
          "Get crops failed at format NC1HWC0, the value of crops must be greater and equal 0, but "
          "got [%ld, %ld, %ld, %ld].",
          crops_vec[INDEX_0], crops_vec[INDEX_1], crops_vec[INDEX_2], crops_vec[INDEX_3]);
      return false;
    }
    if ((crops_vec[INDEX_0] + crops_vec[INDEX_1] >= input_shape.GetDim(INDEX_2) * block_vec[INDEX_0]) ||
        (crops_vec[INDEX_2] + crops_vec[INDEX_3] >= input_shape.GetDim(INDEX_3) * block_vec[INDEX_1])) {
      VECTOR_INNER_ERR_REPORT_TILIING(
          op_type,
          "The crops should less than (input_shape)*(block_shape), but got input:[%ld, %ld], block:[%ld, %ld], "
          "crops:[%ld, %ld, %ld, %ld]",
          input_shape.GetDim(INDEX_2), input_shape.GetDim(INDEX_3), block_vec[INDEX_0], block_vec[INDEX_1],
          crops_vec[INDEX_0], crops_vec[INDEX_1], crops_vec[INDEX_2], crops_vec[INDEX_3]);
      return false;
    }
    if (input_shape.GetDim(INDEX_0) % (block_vec[INDEX_0] * block_vec[INDEX_1]) != 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(
          op_type, "The batch/(block_shape) should be integer, but got input:[%ld], block:[%ld, %ld]",
          input_shape.GetDim(INDEX_0), block_vec[INDEX_0], block_vec[INDEX_1]);
      return false;
    }
  } else {
    if ((block_vec[INDEX_0] <= 0) || (block_vec[INDEX_1] <= 0) || (block_vec[INDEX_2] <= 0)) {
      VECTOR_INNER_ERR_REPORT_TILIING(
          op_type,
          "Get block_shape failed at format NDC1HWC0, the value of block_shape must be greater to 0, but "
          "got [%ld, %ld, %ld].",
          block_vec[INDEX_0], block_vec[INDEX_1], block_vec[INDEX_2]);
      return false;
    }
    if ((crops_vec[INDEX_0] < 0) || (crops_vec[INDEX_1] < 0) || (crops_vec[INDEX_2] < 0) || (crops_vec[INDEX_3] < 0) ||
        (crops_vec[INDEX_4] < 0) || (crops_vec[INDEX_5] < 0)) {
      VECTOR_INNER_ERR_REPORT_TILIING(
          op_type,
          "Get crops failed at format NDC1HWC0, the value of crops must be greater and equal 0, but "
          "got[%ld, %ld, %ld, %ld, %ld, %ld].",
          crops_vec[INDEX_0], crops_vec[INDEX_1], crops_vec[INDEX_2], crops_vec[INDEX_3], crops_vec[INDEX_4],
          crops_vec[INDEX_5]);
      return false;
    }
    if ((crops_vec[INDEX_0] + crops_vec[INDEX_1] >= input_shape.GetDim(INDEX_1) * block_vec[INDEX_0]) ||
        (crops_vec[INDEX_2] + crops_vec[INDEX_3] >= input_shape.GetDim(INDEX_3) * block_vec[INDEX_1]) ||
        (crops_vec[INDEX_4] + crops_vec[INDEX_5] >= input_shape.GetDim(INDEX_4) * block_vec[INDEX_2])) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type,
                                      "The crops should less than (input_shape)*(block_shape), but got input:[%ld, "
                                      "%ld, %ld], block:[%ld, %ld, %ld], "
                                      "pads:[%ld, %ld, %ld, %ld, %ld, %ld]",
                                      input_shape.GetDim(INDEX_1), input_shape.GetDim(INDEX_3),
                                      input_shape.GetDim(INDEX_4), block_vec[INDEX_0], block_vec[INDEX_1],
                                      block_vec[INDEX_2], crops_vec[INDEX_0], crops_vec[INDEX_1], crops_vec[INDEX_2],
                                      crops_vec[INDEX_3], crops_vec[INDEX_4], crops_vec[INDEX_5]);
      return false;
    }
    if (input_shape.GetDim(INDEX_0) % (block_vec[INDEX_0] * block_vec[INDEX_1] * block_vec[INDEX_2]) != 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(
          op_type, "The batch/(block_shape) should be integer, but got input:[%ld], block:[%ld, %ld, %ld]",
          input_shape.GetDim(INDEX_0), block_vec[INDEX_0], block_vec[INDEX_1], block_vec[INDEX_2]);
      return false;
    }
  }

  // if input_h and block_h is one, can swap h and w
  bool swap_h_w = false;
  if (input_format == FORMAT_NC1HWC0 && input_shape.GetDim(INDEX_2) == 1 && block_vec[INDEX_0] == 1 &&
      crops_vec[INDEX_0] == 0 && crops_vec[INDEX_1] == 0) {
    swap_h_w = true;
    std::swap(block_vec[INDEX_0], block_vec[INDEX_1]);
    std::swap(crops_vec[INDEX_0], crops_vec[INDEX_2]);
    std::swap(crops_vec[INDEX_1], crops_vec[INDEX_3]);
  }
  PROFILING_TILING_AFTER_CALCU_TILING_REG();
  // calc tiling params, set tiling params, print tiling params
  TilingParam param;
  CalTilingParam(param, input_shape, input_format, ub_ele, core_num, block_vec, crops_vec, swap_h_w);
  SetTilingParam(param, run_info);
  PrintTilingParam(param);

  // block_dim, use fot tik op; workspace, null for tik op
  run_info.SetBlockDim(param.act_core_num);
  PROFILING_TILING_END();
  OP_LOGI(op_type.c_str(), "BatchToSpaceNDTiling run success.");
  return true;
}

// register tiling interface of batch_to_space and batch_to_space_nd op.
REGISTER_OP_TILING_V3_WITH_VECTOR(BatchToSpace, BatchToSpaceNDTiling, COMPILE_INFO_KEY, NO_OPTIONAL_VALUE);
REGISTER_OP_TILING_V3_WITH_VECTOR(BatchToSpaceND, BatchToSpaceNDTiling, COMPILE_INFO_KEY, NO_OPTIONAL_VALUE);
}  // namespace optiling
