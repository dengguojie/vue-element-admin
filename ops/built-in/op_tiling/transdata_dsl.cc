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
 * \file trans_data_dsl.cc
 * \brief
 */
#include "transdata_dsl_general.h"

namespace optiling {
using namespace transdata_dsl;
int64_t transdata_dsl::CeilDiv(int64_t value, int64_t factor) {
  return factor == 0 ? DEFAULT : (value + factor - 1) / factor;
}

int64_t transdata_dsl::Prod(const int64_t* input, size_t ptr, size_t length) {
  int64_t base = 1;
  if (ptr >= length) {
    return base;
  }
  for (size_t idx = ptr; idx < length; idx++) {
    base *= input[idx];
  }
  return base;
}

int64_t transdata_dsl::SetAlign(int64_t value, int64_t align_factor) {
  return align_factor == 0 ? DEFAULT : (value + align_factor - 1) / align_factor * align_factor;
}

void TransdataClassify::GetInputOutput(Shape& input, Shape& output, Shape& reshape) {
  // if compileInfo.is_forward is 1, infer reshape\output by input.
  // if compileInfo.is_forward is 0, infer reshape\input by output
  if (compileInfo.is_forward) {
    ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
    ge::ConstGeTensorDescPtr input_desc = op_desc->GetInputDescPtr(0);
    auto input_ge_shape = input_desc->GetShape();
    input.SetSize(input_ge_shape.GetDimNum());
    for (size_t i = 0; i < input.size; i++) {
      input.shape[i] = input_ge_shape.GetDim(i);
    }
    DoFusing(input, output, reshape);
  } else {
    ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
    ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc(0);
    auto output_ge_shape = output_desc->MutableShape();
    output.SetSize(output_ge_shape.GetDimNum());
    for (size_t i = 0; i < output.size; i++) {
      output.shape[i] = output_ge_shape.GetDim(i);
    }
    DoFusing(output, input, reshape);
  }
}

void TransdataClassify::DoFusing(Shape& input, Shape& output, Shape& reshape) {
  /*
   * If backward, input is output_shape, output is input_shape.
   * If forward, input is input_shape, output is output_shape.
   * ori_length is valid size of input.
   * Infer Regulation (Forward):
   *    input_shape __fuse__ input_shape (const had been fused in classifier)
   *    input_shape __pad__ pad_shape (if pad one, need change while const)
   *    pad_shape __reshape__ reshape
   *    reshape __transpose__ output_shape
   */
  size_t ori_length = input.size;
  input.size = compileInfo.src_pad.size();
  output.size = compileInfo.permute.size();
  reshape.size = output.size;
  size_t root_ptr = 0;

  // __fuse__
  if (not compileInfo.is_const) {
    while (root_ptr + 1 <= input.size) {
      if (root_ptr == input.size - 1 and compileInfo.src_fuse[root_ptr] == ori_length - 1) {
        input.shape[root_ptr] = input.shape[compileInfo.src_fuse[root_ptr]];
        break;
      }
      size_t begin = compileInfo.src_fuse[root_ptr];
      size_t end = root_ptr + OFFSET_2 <= input.size ? compileInfo.src_fuse[root_ptr + 1] : ori_length;
      int64_t base = 1;
      for (size_t i = begin; i < end; i++) {
        base *= input.shape[i];
      }
      input.shape[root_ptr] = base;
      root_ptr++;
    }
  }

  // __pad__ and __reshape__
  // pad_mark is 0 that do nothing
  // pad_mark is 1 that do padding
  // pad_mark is 2 that split dim to C1 and C0
  root_ptr = 0;
  for (size_t i = 0; i < input.size; i++) {
    if (compileInfo.src_pad[i] == 0) {
      reshape.shape[root_ptr] = input.shape[i];
      root_ptr++;
    } else if (compileInfo.src_pad[i] == 1) {
      reshape.shape[root_ptr] = SetAlign(input.shape[i], compileInfo.pad_align_size);
      root_ptr++;
    } else {
      reshape.shape[root_ptr] = SetAlign(input.shape[i], compileInfo.pad_align_size) / compileInfo.pad_align_size;
      reshape.shape[root_ptr + 1] = compileInfo.pad_align_size;
      root_ptr += OFFSET_2;
    }
  }

  // __transpose__
  if (compileInfo.is_forward) {
    // reshape + perm -> output
    for (size_t i = 0; i < output.size; i++) {
      output.shape[i] = reshape.shape[compileInfo.permute[i]];
    }
  } else {
    // output + perm -> reshape
    for (size_t i = 0; i < output.size; i++) {
      output.shape[compileInfo.permute[i]] = reshape.shape[i];
    }
  }
}

int64_t TransdataClassify::ChooseStrategy(const Shape& input, const Shape& output) const {
  // only choose computeType
  return BaseSch;
}

bool TransdataTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const {
  OP_LOGD(op_type.c_str(), "Transdata DSL tiling running");
  // 1. Get input_shape and output_shape after dimension collapse
  // 2. Choose different SCH
  Shape input;
  Shape output;
  Shape reshape;
  TransdataClassify classify(op_paras, compileInfo);
  classify.GetInputOutput(input, output, reshape);
  int64_t sch_branch = classify.ChooseStrategy(input, output);
  if (sch_branch == 0) {
    TransdataGeneral transdata(op_type, compileInfo, run_info, input, output, reshape);
    return transdata.DoTiling();
  } else if (sch_branch == BorrowNSch) {
    return false;
  } else if (sch_branch == BorrowHSch) {
    return false;
  } else {
    return false;
  }
}

bool TransdataTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info,
                                      const OpInfo& op_info) const {
  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Transdata DSL custom tiling is not supported yet");
  return false;
}

bool CompileInfoTransdataDSL::Check() const {
  V_OP_TILING_CHECK((core_num > 0),
                    VECTOR_INNER_ERR_REPORT_TILIING("TransdataDSL", "core_num is %ld that is illegal", core_num),
                    return false);
  V_OP_TILING_CHECK(
      (pad_align_size > 0),
      VECTOR_INNER_ERR_REPORT_TILIING("TransdataDSL", "pad_align_size is %ld that is illegal", pad_align_size),
      return false);
  V_OP_TILING_CHECK((align_size > 0),
                    VECTOR_INNER_ERR_REPORT_TILIING("TransdataDSL", "align_size is %ld that is illegal", align_size),
                    return false);
  return true;
}

CompileInfoTransdataDSL::CompileInfoTransdataDSL(const std::string& op_type, const nlohmann::json& parsed_json_obj) {
  OP_LOGD(op_type.c_str(), "transdata compile info construct func running");

  const auto& common_info = parsed_json_obj.at("_common_info").get<std::vector<int64_t>>();
  std::size_t expect_common_info_len = 6;
  if (common_info.size() == expect_common_info_len) {
    // index
    std::size_t forward_idx = 0;
    std::size_t align_size_idx = 1;
    std::size_t pad_align_size_idx = 2;
    std::size_t core_num_idx = 3;
    std::size_t is_const_idx = 4;
    std::size_t is_const_compile_idx = 5;
    // value
    is_forward = common_info[forward_idx];
    align_size = common_info[align_size_idx];
    pad_align_size = common_info[pad_align_size_idx];
    core_num = common_info[core_num_idx];
    is_const = common_info[is_const_idx];
    is_const_compile = common_info[is_const_compile_idx];
  }

  // src_pad: backward based on output, forward based on input
  // src_fuse: backward based on output, forward based on input
  // permute: based on output
  src_pad = parsed_json_obj.at("_src_pad").get<std::vector<size_t>>();
  src_fuse = parsed_json_obj.at("_src_fuse").get<std::vector<size_t>>();
  permute = parsed_json_obj.at("_permute").get<std::vector<size_t>>();
  unknown_dims = parsed_json_obj.at("_unknown_dims").get<std::vector<size_t>>();
  ub_info = parsed_json_obj.at("_ub_info").get<std::vector<std::vector<int64_t>>>();

  // collect vars
  if (is_const and not is_const_compile) {
    // const runtime
    const_block_dims = parsed_json_obj.at("_const_block_dims").get<std::unordered_map<std::string, int32_t>>();
  }

  check_success = Check();
}

std::shared_ptr<AutoTilingHandler> CreateTransdataTilingHandler(const std::string& op_type, const std::string& pattern,
                                                                const nlohmann::json& parsed_compile_info) {
  auto compile_info_ptr = std::make_shared<TransdataTilingHandler>(op_type, pattern, parsed_compile_info);
  if (!compile_info_ptr->ParsedSuccess()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Transdata parse compile info failed");
    return std::shared_ptr<AutoTilingHandler>(nullptr);
  }

  return compile_info_ptr;
}
}  // namespace optiling