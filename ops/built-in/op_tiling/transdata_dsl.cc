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
#include "transdata_dsl_borrow.h"
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

int64_t transdata_dsl::SetAlign(int64_t value, int64_t factor) {
  return factor == 0 ? DEFAULT : (value + factor - 1) / factor * factor;
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
  input.size = compileInfo.src_pad_mode.size();
  output.size = compileInfo.permute.size();
  reshape.size = output.size;
  size_t root_ptr = 0;

  // __fuse__
  if (!compileInfo.is_const) {
    while (root_ptr + 1 <= input.size) {
      if (root_ptr == input.size - 1 && compileInfo.src_fuse[root_ptr] == ori_length - 1) {
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
    if (compileInfo.src_pad_mode[i] == 0) {
      reshape.shape[root_ptr] = input.shape[i];
      root_ptr++;
    } else if (compileInfo.src_pad_mode[i] == 1) {
      reshape.shape[root_ptr] = SetAlign(input.shape[i], compileInfo.src_pad_var[i]);
      root_ptr++;
    } else {
      is_data_move = false;
      reshape.shape[root_ptr] = SetAlign(input.shape[i], compileInfo.src_pad_var[i]) / compileInfo.src_pad_var[i];
      reshape.shape[root_ptr + 1] = compileInfo.src_pad_var[i];
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

size_t TransdataClassify::BHBNStrategy(const Shape& input) {
  // Choose BH|BN
  int64_t num = 1;
  int64_t used_buffer = 1;
  int64_t buffer_size = 0;
  int64_t target = 0;
  for (size_t k = 0; k < input.size; k++) {
    // update
    size_t i = input.size - 1 - k;
    int64_t v = input.shape[i];
    if (i == index_c || i == input.size - 1) {
      num = num * v;
      continue;
    }

    // infer
    if (v >= borrow_factor) {
      used_buffer = v / borrow_factor * borrow_factor * num;
      used_buffer = used_buffer >= UBSize ? UBSize : used_buffer;
    } else {
      used_buffer = v * num;
      if (used_buffer >= UBSize / borrow_factor * v) {
        used_buffer = UBSize / borrow_factor * v;
      }
    }

    // update
    if (used_buffer > buffer_size) {
      target = i;
      buffer_size = used_buffer;
    }
    num = num * v;
  }
  return target == 0 ? BORROW_N_SCH : BORROW_H_SCH;
}

size_t TransdataClassify::BorrowStrategy(const Shape& input) {
  // BH and BN used same space.
  // Const had choose (BH|BN|BASE) in Python: ub_info only had one useful value.
  bool base_space_is_ok = compileInfo.ub_info[BASE_SCH][0] != -1;
  bool bn_space_is_ok = compileInfo.ub_info[BORROW_N_SCH][0] != -1;
  bool bh_space_is_ok = compileInfo.ub_info[BORROW_H_SCH][0] != -1;
  if (base_space_is_ok && !bn_space_is_ok && !bh_space_is_ok) {
    return BASE_SCH;
  }
  if (!base_space_is_ok && bn_space_is_ok && !bh_space_is_ok) {
    return BORROW_N_SCH;
  }
  if (!base_space_is_ok && !bn_space_is_ok && bh_space_is_ok) {
    return BORROW_H_SCH;
  }

  // Dynamic choose (BH|BN|BASE) in C++
  int64_t base = Prod(input.shape, index_c + 1, input.size);
  base *= SetAlign(input.shape[index_c], compileInfo.src_pad_var[index_c]);

  UBSize = compileInfo.ub_info[BORROW_N_SCH][0];
  if (UBSize == -1) UBSize = compileInfo.ub_info[BORROW_H_SCH][0];

  // FP32 would be reinterpret as FP16:
  // borrow_factor is 16 in [FP16,FP32], but is 32 in [int8,].
  // UBSize from compileInfo is based on FP16, but shape from ge is based on FP32, it means UBSize need to divided by 2.
  borrow_factor = compileInfo.align_size;
  if (compileInfo.align_size < BLOCK / FP16_BYTE) {
    borrow_factor = BLOCK / FP16_BYTE;
    UBSize = UBSize / (borrow_factor / compileInfo.align_size);
  }

  bool first_dim_is_n = compileInfo.permute[0] == 0;
  bool bn_is_ok = UBSize >= borrow_factor * base && first_dim_is_n;
  bool bh_is_ok = UBSize >= borrow_factor * base;
  if (!compileInfo.is_forward) {
    bh_is_ok = UBSize >= borrow_factor * borrow_factor * base;
  }

  if (bn_is_ok && bh_is_ok) {
    return BHBNStrategy(input);
  } else if (bn_is_ok) {
    return BORROW_N_SCH;
  } else if (bh_is_ok) {
    return BORROW_H_SCH;
  } else {
    return BASE_SCH;
  }
}

bool TransdataClassify::IsLegalBurstLen(const Shape& input) {
  // Can borrow any axis except C
  int idx[input.size] = {0};
  for (size_t i = 0; i < input.size; i++) {
    idx[i] = i > index_c ? i + 1 : i;
  }
  int64_t ub = 1;
  bool burst_len_is_ok = false;
  for (size_t k = 0; k < input.size; k++) {
    size_t i = input.size - 1 - k;
    size_t v = idx[i];
    bool is_last_axis = i == input.size - 1;
    bool is_c_axis = v == index_c;
    bool is_transpose_axis = static_cast<size_t>(VectorIndex(compileInfo.permute, v)) != v;

    int64_t used_buffer = ub;
    if (is_last_axis || is_c_axis || is_transpose_axis) {
      used_buffer *= input.shape[i];
    }
    if (used_buffer >= compileInfo.align_size) {
      burst_len_is_ok = true;
      break;
    }
    ub *= input.shape[i];
  }
  return burst_len_is_ok;
}

size_t TransdataClassify::ChooseStrategy(const Shape& input, const Shape& output) {
  // only choose computeType
  index_c = static_cast<size_t>(VectorIndex(compileInfo.src_pad_mode, OFFSET_2));
  is_last_transpose = compileInfo.is_forward ? compileInfo.permute[output.size - 1] != output.size - 1 :
                      compileInfo.permute[input.size - 1] != input.size - 1;
  // choose branch
  if (is_last_transpose || is_data_move) {
    // last-transpose and data-move branch
    return BASE_SCH;
  } else {
    // n-last-transpose-branch(NHC: C is last-dim, H is second-dim)
    // distinguish between BASE_SCH and Other SCH.
    const Shape &target = compileInfo.is_forward ? input : output;
    int64_t last_dim = target.shape[target.size - 1];
    if (last_dim % compileInfo.align_size == 0 || !IsLegalBurstLen(target)) {
      return BASE_SCH;
    }
    // Empirical Method
    int64_t last_dim_limit = compileInfo.is_forward ? 1024 : 128;
    if (last_dim >= last_dim_limit) {
      return BASE_SCH;
    }
    return BorrowStrategy(target);
  }
}

size_t TransdataClassify::TransposeWork(const Shape& input, const Shape& output) const {
  if (is_last_transpose) {
    return 1;
  }
  Shape after_transpose;
  const Shape *base = compileInfo.is_forward ? &output : &input;
  after_transpose.SetSize(base->size);
  for (size_t i = 0; i < base->size; i++) {
    if (compileInfo.is_forward) {
      after_transpose.shape[i] = base->shape[i];
    } else {
      after_transpose.shape[static_cast<size_t>(VectorIndex(compileInfo.permute, i))] = base->shape[i];
    }
  }
  // Eliminate one
  std::vector<size_t> perm;
  perm.reserve(MAX_DIM);
  for (size_t i = 0; i < after_transpose.size; i++) {
    if (after_transpose.shape[i] != 1) {
      perm.emplace_back(compileInfo.permute[i]);
    }
  }
  // Sorted: if transpose work: perm != sorted(perm)
  size_t i = 0;
  while(i < perm.size() - 1) {
    if (perm[i] > perm[i+1]) {
      return 1;
    }
    i++;
  }
  return 0;
}

bool TransdataTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const {
  // 1. Get input_shape and output_shape after dimension collapse
  // 2. Choose different SCH
  OP_LOGD(op_type.c_str(), "Transdata DSL tiling running");
  Shape input;
  Shape output;
  Shape reshape;
  if (compileInfo.is_const && !compileInfo.is_const_compile) {
    // const runtime
    std::string pattern_str = std::to_string(CONST_KEY);
    run_info.SetBlockDim(compileInfo.const_block_dims.at(pattern_str));
    run_info.SetTilingKey(CONST_KEY);
    return true;
  }
  TransdataClassify classify(op_paras, compileInfo);
  classify.GetInputOutput(input, output, reshape);
  size_t sch_branch = classify.ChooseStrategy(input, output);
  size_t transpose_work = classify.TransposeWork(input, output);
  if (sch_branch == BASE_SCH) {
    TransdataGeneral transdata(op_type, compileInfo, run_info, input, output, reshape, transpose_work);
    return transdata.DoTiling();
  } else {
    TransdataBorrow transdata(op_type, compileInfo, run_info, input, output);
    transdata.SetAttr(sch_branch, transpose_work);
    return transdata.DoTiling();
  }
}

bool TransdataTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info,
                                      const OpInfo& op_info) const {
  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Transdata DSL custom tiling is not supported yet");
  return false;
}

bool CompileInfoTransdataDSL::Check() const {
  V_OP_TILING_CHECK(
      (core_num > 0),
      VECTOR_INNER_ERR_REPORT_TILIING("TransdataDSL", "core_num is %ld that is illegal", core_num),
      return false);
  V_OP_TILING_CHECK(
      (src_pad_mode.size() == src_pad_var.size()),
      VECTOR_INNER_ERR_REPORT_TILIING("TransdataDSL", "sizes of src_pad_mode and src_pad_var are not equal."),
      return false);
  V_OP_TILING_CHECK(
      (align_size > 0),
      VECTOR_INNER_ERR_REPORT_TILIING("TransdataDSL", "align_size is %ld that is illegal", align_size),
      return false);
  return true;
}

void CompileInfoTransdataDSL::ParseCommonInfo(const nlohmann::json& parsed_json_obj) {
  const auto& common_info = parsed_json_obj.at("_common_info").get<std::vector<int64_t>>();
  std::size_t expect_common_info_len = 5;
  if (common_info.size() == expect_common_info_len) {
    // index
    std::size_t forward_idx = 0;
    std::size_t align_size_idx = 1;
    std::size_t core_num_idx = 2;
    std::size_t is_const_idx = 3;
    std::size_t is_const_compile_idx = 4;
    // value
    is_forward = common_info[forward_idx];
    align_size = common_info[align_size_idx];
    core_num = common_info[core_num_idx];
    is_const = common_info[is_const_idx];
    is_const_compile = common_info[is_const_compile_idx];
  }
}

void CompileInfoTransdataDSL::ParseBaseGraphInfo(const nlohmann::json& parsed_json_obj) {
  // src_pad: backward based on output, forward based on input
  // src_fuse: backward based on output, forward based on input
  // permute: based on output
  src_fuse = parsed_json_obj.at("_src_fuse").get<std::vector<size_t>>();
  src_pad_mode = parsed_json_obj.at("_src_pad_mode").get<std::vector<size_t>>();
  src_pad_var = parsed_json_obj.at("_src_pad_var").get<std::vector<size_t>>();
  permute = parsed_json_obj.at("_permute").get<std::vector<size_t>>();
  ub_info = parsed_json_obj.at("_ub_info").get<std::vector<std::vector<int64_t>>>();
  // collect vars
  if (is_const && !is_const_compile) {
    // const runtime
    const_block_dims = parsed_json_obj.at("_const_block_dims").get<std::unordered_map<std::string, int32_t>>();
  }
}

void CompileInfoTransdataDSL::ParseBNGraphInfo(const nlohmann::json& parsed_json_obj) {
  // borrow-n
  if (parsed_json_obj.contains("_bn_x1x0")) {
    bn_x1x0 = parsed_json_obj.at("_bn_x1x0").get<std::vector<size_t>>();
  }
  if (parsed_json_obj.contains("_bn_c1c0")) {
    bn_c1c0 = parsed_json_obj.at("_bn_c1c0").get<std::vector<size_t>>();
  }
  if (parsed_json_obj.contains("_bn_permute")) {
    bn_permute = parsed_json_obj.at("_bn_permute").get<std::vector<size_t>>();
  }
}

void CompileInfoTransdataDSL::ParseBHGraphInfo(const nlohmann::json& parsed_json_obj) {
  if (parsed_json_obj.contains("_bh_x1x0")) {
    bh_x1x0 = parsed_json_obj.at("_bh_x1x0").get<std::vector<size_t>>();
  }
  if (parsed_json_obj.contains("_bh_c1c0")) {
    bh_c1c0 = parsed_json_obj.at("_bh_c1c0").get<std::vector<size_t>>();
  }
  if (parsed_json_obj.contains("_bh_permute")) {
    bh_permute = parsed_json_obj.at("_bh_permute").get<std::vector<size_t>>();
  }
}

CompileInfoTransdataDSL::CompileInfoTransdataDSL(const std::string& op_type, const nlohmann::json& parsed_json_obj) {
  OP_LOGD(op_type.c_str(), "transdata dsl compile info construct func running");
  ParseCommonInfo(parsed_json_obj);
  ParseBaseGraphInfo(parsed_json_obj);
  ParseBNGraphInfo(parsed_json_obj);
  ParseBHGraphInfo(parsed_json_obj);
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