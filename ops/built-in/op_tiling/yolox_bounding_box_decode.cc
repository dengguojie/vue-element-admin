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
 * \file yolox_bounding_box_decode.cc
 * \brief dynamic shape tiling of yolox_bounding_box_decode
 */
#include<string>
#include <cmath>
#include <nlohmann/json.hpp>
#include "op_log.h"
#include "op_tiling.h"
#include "error_log.h"

namespace optiling {
using namespace ge;
using namespace std;


struct TilingInfo {
  int32_t core_num;
  int32_t bboxes_data_each_block;
  int32_t batch_number;
  int32_t box_number;
};

static void InitTilingParam(TilingInfo& param) {
  param.core_num = 0;
  param.bboxes_data_each_block = 0;
  param.batch_number = 0;
  param.box_number = 0;
}

static void SetTilingParam(const TilingInfo& param, OpRunInfo& run_info) {
  ByteBufferPut(run_info.tiling_data, param.batch_number);
  ByteBufferPut(run_info.tiling_data, param.box_number);
}

static void PrintParam(const TilingInfo& param) {
  // output param
  OP_LOGD("YoloxBoundingBoxDecodeTiling ", "core_num:%d", param.core_num);
  OP_LOGD("YoloxBoundingBoxDecodeTiling ", "bboxes_data_each_block:%d", param.bboxes_data_each_block);
  OP_LOGD("YoloxBoundingBoxDecodeTiling ", "batch_number:%d", param.batch_number);
  OP_LOGD("YoloxBoundingBoxDecodeTiling ", "box_number:%d", param.box_number);
}


static bool GetParam(const TeOpParas& op_paras, TilingInfo& param) {
  std::vector<int64_t> input_shape = op_paras.inputs[1].tensor[0].shape;
 
  // calcu element_number
  int64_t element_total = 1;
  for (size_t i = 0; i < input_shape.size(); i++) {
      element_total *= input_shape[i];
  }
  int32_t element_number = element_total;
  if (0 == element_number) {
      VECTOR_INNER_ERR_REPORT_TILIING("YoloxBoundingBoxDecode", 
          "op[YoloxBoundingBoxDecodeTiling]:GetParam fail.element_number is zero");
      return false;
  }

  // fill calcu param
  int32_t batch_size = input_shape[0];
  int32_t box_num = input_shape[1];
  param.batch_number = batch_size;
  param.box_number = box_num;
   
  return true;
}

static bool GetCompileParam(const std::string& opType,
                            const nlohmann::json& opCompileInfoJson,
                            TilingInfo& param) {
  using namespace nlohmann;

  const auto& allVars = opCompileInfoJson["vars"];
  if(allVars.count("core_num") == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, 
          "op[YoloxBoundingBoxDecodeTiling]:GetCompileParam, get core_num error");
      return false;
  }
  param.core_num = allVars["core_num"].get<std::int32_t>();
  if (allVars.count("bboxes_data_each_block") == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, 
          "op[YoloxBoundingBoxDecodeTiling]:GetCompileParam, get bboxes_data_each_block error");
      return false;
  }
  param.bboxes_data_each_block = allVars["bboxes_data_each_block"].get<std::int32_t>();

  return true;
}

static bool CheckParam(const std::string& opType, TilingInfo& param) {
  if (0 == param.core_num) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, 
          "op[YoloxBoundingBoxDecodeTiling]:check fail.coreNum is zero");
      return false;
  }

  if (0 == param.bboxes_data_each_block) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, 
          "op[YoloxBoundingBoxDecodeTiling]:check fail.bboxes_data_each_block is zero");
      return false;
  }

  return true;
}

/*
 * @brief: tiling function of op
 * @param: [in] opType: opType of op
 * @param: [in] opParas: inputs/outputs/attrs of op
 * @param: [in] op_info: compile time generated info of op
 * @param: [out] runInfo: result data
 * @return bool: success or not
*/
bool YoloxBoundingBoxDecodeTiling(const string& op_type, const TeOpParas& op_paras,
                                  const nlohmann::json& op_info, OpRunInfo& run_info) {
  OP_LOGD(op_type.c_str(), "YoloxBoundingBoxDecodeTiling running.");

  TilingInfo param;
  InitTilingParam(param);
  if (!GetCompileParam(op_type, op_info, param)) {
      return false;
  }
  
  if (!CheckParam(op_type, param)) {
      return false;
  }
  
  if (!GetParam(op_paras, param)) {
      return false;
  }

  SetTilingParam(param, run_info);
  PrintParam(param);
  run_info.block_dim = param.core_num;

  OP_LOGD(op_type.c_str(), "YoloxBoundingBoxDecodeTiling end.");
  return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(YoloxBoundingBoxDecode, YoloxBoundingBoxDecodeTiling);
}  // namespace optiling
