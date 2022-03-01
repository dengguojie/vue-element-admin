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
 * \file transdata_dsl_borrow_n.cpp
 * \brief
 */
#include "transdata_dsl_borrow_n.h"

namespace optiling {
using namespace transdata_dsl;

int64_t TransdataBN::CalcTilingKey() {
  using namespace std;
  int64_t db = 0;
  int64_t is_forward = compileInfo.is_forward ? FORWARD_KEY : BACKWARD_KEY;
  size_t key_num = 7;
  int64_t pos[key_num] = {db,
                          is_forward,
                          static_cast<int64_t>(computeType),
                          static_cast<int64_t>(shapeType),
                          static_cast<int64_t>(tilingInfo.blk_idx),
                          VectorIndex(permute, tilingInfo.ub_0_idx),
                          static_cast<int64_t>(tilingInfo.ub_1_idx)};
  int64_t val[key_num] = {1000000000, 100000000, 10000000, 100000, 10000, 1000, 100};
  int64_t key = 0;
  for (size_t i = 0; i < key_num; i++) {
    key += pos[i] * val[i];
  }
  return key;
}

bool TransdataBN::InferTilingInput() {
  // Reinterpret tensor(32bit, 64bit) as tensor(16bit)
  align_size = compileInfo.align_size;
  if (compileInfo.align_size < BLOCK / FP16_BYTE) {
    is_reinterpret = true;
    int64_t pad_var = BLOCK / FP16_BYTE / compileInfo.align_size;
    tiling_output.shape[tiling_output.size] = pad_var;
    tiling_output.SetSize(tiling_output.size + 1);
    input.Insert(input.size, pad_var);
    output.Insert(output.size, pad_var);
    align_size = BLOCK / FP16_BYTE;
  }
  // Create tiling_input [N.o, x0,...,xn, 16]
  tiling_input.SetSize(tiling_output.size);
  permute.resize(tiling_input.size);
  for (size_t i = 0; i < permute.size(); i++) {
    permute[i] = i == 0 ? 0 : (i == 1 ? permute.size() - 1 : i - 1);
  }
  for (size_t i = 0; i < tiling_output.size; i++) {
    tiling_input.shape[permute[i]] = tiling_output.shape[i];
  }

  // Update input and output
  if (not has_dim_n) {
    input.Insert(0, 1);
    output.Insert(0, 1);
  }

  // Other params
  ele_byte = BLOCK / align_size;
  mte_rate = PACKET_SENDING_RATE / ele_byte;
  return true;
}

bool TransdataBN::OnceTiling() {
  // ptrI and ptrO split same axis: [N,H,C,16] -> [N,16,H,C]
  // handle_c1: make c1\c0 in ub-internal while c1 belong to ub-external and don't split c1\c0.
  bool init_success = false;
  size_t length = tiling_input.size;
  int64_t n_inner = tiling_input.shape[length - 1];
  bool handle_c1 = compileInfo.is_forward and permute[c1_index] < ptrI + 1;
  // Calc Bound
  num_in_ub = Prod(tiling_input.shape, ptrI + 1, tiling_input.size - 1);
  num_in_ub = handle_c1 ? num_in_ub * tiling_input.shape[permute[c1_index]] : num_in_ub;
  if (num_in_ub == 0) {
    // static check
    return false;
  }
  int64_t bound = ptrI == 0 ?
          UBSize / n_inner / SetAlign(num_in_ub, STRIDE_3 * align_size) :
          (UBSize / n_inner - STRIDE_3 * align_size + 1) / num_in_ub;

  for (int64_t factor = 1; factor <= bound; factor++) {
    // Check tail
    GetOutputRealTail(ptrO, factor, mte3);
    bool tail_is_legal = mte3.tailLen == 0 || mte3.tailLen >= align_size;
    // Check main
    core = Prod(tiling_input.shape, 0, ptrI) * CeilDiv(tiling_input.shape[ptrI], factor);
    core = handle_c1 ? core / tiling_input.shape[permute[c1_index]] : core;
    bool main_is_legal = mte3.mainLen >= align_size || core == 1;
    if (tail_is_legal and main_is_legal) {
      factorI = factor;
      init_success = true;
      if (mte3.mainLen >= mte_rate) {
        break;
      }
    }
  }

  // Check init
  if (not init_success) {
    return false;
  }

  // Adjust
  if (core > compileInfo.core_num) {
    // extra core can improve UB usage
    for (int64_t factor = factorI; factor <= bound; factor++) {
      // Check tail
      GetOutputRealTail(ptrO, factor, mte3);
      bool tail_is_legal = mte3.tailLen == 0 || mte3.tailLen >= align_size;
      if (tail_is_legal) {
        core = Prod(tiling_input.shape, 0, ptrI) * CeilDiv(tiling_input.shape[ptrI], factor);
        core = handle_c1 ? core / tiling_input.shape[permute[c1_index]] : core;
        factorI = factor;
        if (core < compileInfo.core_num) {
          break;
        }
      }
    }
  }
  factorO = factorI;
  return true;
}

void TransdataBN::CompareTiling() {
  if (tilingInfo.core == 0) {
    // First Init
    tilingInfo.UBInfoSet(ptrI, ptrO, factorI, factorO, core);
  } else {
    int64_t new_core_distance = abs(core - compileInfo.core_num);
    int64_t old_core_distance = abs(tilingInfo.core - compileInfo.core_num);
    if (new_core_distance < old_core_distance) {
      tilingInfo.UBInfoSet(ptrI, ptrO, factorI, factorO, core);
    }
  }
}

void TransdataBN::AdjustUBFactorForward() {
  // Forward's out is align
  int64_t mainFactor = tilingInfo.ub_1_factor;
  int64_t dimBound = tiling_output.shape[tilingInfo.ub_1_idx];
  int64_t tailFactor = dimBound % mainFactor;
  float tail_percent = static_cast<float>(tailFactor) / static_cast<float>(mainFactor);
  float fine_tuning_threshold = 0.8;
  if (tail_percent != 0 and tail_percent < fine_tuning_threshold) {
    int loop = dimBound / mainFactor + 1;
    mainFactor = (dimBound % loop) ? (dimBound / loop + 1) : (dimBound / loop);
    tilingInfo.ub_0_factor = mainFactor;
    tilingInfo.ub_1_factor = mainFactor;
  }
}

void TransdataBN::AdjustUBFactorBackward() {
  // Backward: try to make MTE3 align
  int64_t factor = tilingInfo.ub_1_factor;
  for (; factor >= 1; factor--) {
    GetOutputRealTail(tilingInfo.ub_1_idx, factor, mte3);
    bool main_align = mte3.mainLen % align_size == 0;
    bool tail_align = mte3.tailLen % align_size == 0;
    // best
    if (main_align and tail_align) {
      tilingInfo.ub_1_factor = factor;
      break;
    }
    // better (choose bigger main)
    if (main_align and mte3.tailLen >= align_size) {
      GetOutputRealTail(tilingInfo.ub_1_idx, tilingInfo.ub_1_factor, mte3);
      bool had_aligned = mte3.mainLen % align_size == 0;
      tilingInfo.ub_1_factor = had_aligned ? tilingInfo.ub_1_factor : factor;
    }
  }
  tilingInfo.ub_0_factor = tilingInfo.ub_1_factor;
}

void TransdataBN::GetOutputRealTail(int64_t ptr, int64_t factor, MTEInfo& mte) {
  /* Backward:
   * 1. output is [N,H,C], tiling_output is [N.o,16,H,C].
   * 2. Mapping between output and tiling_output is [0:(1,2),1:2,2:3].
   * 3. [H,C] is serial that be burst_len.
   * Forward:
   * 1. output is [N, C1, H, C0], tiling_output is [N.o, 16, C1, H, C0]
   * 2. [H, C0] is serial that be burst_len, C1 would be n_burst
   * */
  size_t out_ptr = ptr > 1 ? ptr - 1 : 0;
  factor = out_ptr != 0 ? factor : factor * compileInfo.pad_align_size;
  factor = output.shape[out_ptr] > factor ? factor : output.shape[out_ptr];
  int64_t baseLen = Prod(output.shape, out_ptr + 1, output.size);
  if (out_ptr != 0) {
    mte.mainLen = factor * baseLen;
    // static check
    factor = factor == 0 ? 1 : factor;
    mte.tailLen = (output.shape[out_ptr] % factor) * baseLen;
  } else {
    mte.mainLen = baseLen;
    mte.tailLen = 0;
  }
}

void TransdataBN::DiscriminationAxisType(AxisType* type_array, size_t length) {
  // Assure type of axis is belong to internal of UB or not.
  // type_array based on output
  for (size_t i = tilingInfo.ub_0_idx + 1; i < length; i++) {
    // deal input
    type_array[VectorIndex(permute, i)] = UB_INTERNAL;
  }
  for (size_t i = tilingInfo.ub_1_idx + 1; i < length; i++) {
    // deal output
    type_array[i] = UB_INTERNAL;
  }
  // make c1\c0 in ub-internal while forward
  if (compileInfo.is_forward and type_array[c1_index] == UB_EXTERNAL) {
    type_array[c1_index] = UB_INTERNAL;
  }
  // part of split-axis belong to ub_external, part belong to ub_internal.
  type_array[VectorIndex(permute, tilingInfo.ub_0_idx)] = UB_FIRST_SPLIT;
  type_array[tilingInfo.ub_1_idx] = UB_SECOND_SPLIT;
}

bool TransdataBN::InitBackward() {
  /* Support Backward (eg:NC1HC0 -> NHC1C0 -> NHC)
   * (N.o,16,C1,H,C0) -> (N.o,C1,H,C0,16) ->(N.o,H,C1,C0,16) -> (N.o,H,C,16) -> (N.o,16,H,C)
   * Const: const-shape would change permute\src-pad in python while pad dim N.
   * */
  int64_t o_pad[output.size];
  for (size_t i = 0; i < output.size; i++) {
    if (compileInfo.src_pad[i] == 0) {
      o_pad[i] = output.shape[i];
    } else if (compileInfo.src_pad[i] == 1) {
      o_pad[i] = SetAlign(output.shape[i], compileInfo.pad_align_size);
    } else {
      o_pad[i] = SetAlign(output.shape[i], compileInfo.pad_align_size);
      c_index = i;
    }
  }

  // Create tiling_out [N.o, 16, HX, CX]
  size_t ptr = 0;
  for (size_t i = 0; i < output.size; i++) {
    if (i == 0 and compileInfo.permute[0] != 0) {
      // Don't have dim N
      has_dim_n = false;
      tiling_output.shape[ptr] = 1;
      tiling_output.shape[ptr + 1] = compileInfo.pad_align_size;
      tiling_output.shape[ptr + 2] = o_pad[i];
      ptr += OFFSET_2;
      c_index += OFFSET_2;
    } else if (i == 0) {
      // Have dim N
      tiling_output.shape[ptr] = CeilDiv(o_pad[i], compileInfo.pad_align_size);
      tiling_output.shape[ptr + 1] = compileInfo.pad_align_size;
      ptr++;
      c_index++;
    } else {
      // Normal dim
      tiling_output.shape[ptr] = o_pad[i];
    }
    ptr++;
  }
  tiling_output.SetSize(ptr);
  return InferTilingInput();
}

bool TransdataBN::InitForward() {
  /* Support Forward (eg:NHC -> NHC1C0 -> NC1HC0)
   * (N.o,16,H,C) -> (N.o,H,C,16) -> (N.o,H,C1,C0,16) -> (N.o,C1,H,C0,16) -> (N.o,16,C1,H,C0)
   * Const: const-shape would change permute\src-pad in python while pad dim N.
   * */
  for (size_t i = 0; i < input.size; i++) {
    if (compileInfo.src_pad[i] == OFFSET_2) {
      c1_index = i;
      c0_index = i + 1;
    }
  }
  // index base on output(NC1HC0)
  c1_index = VectorIndex(compileInfo.permute, c1_index);
  c0_index = VectorIndex(compileInfo.permute, c0_index);

  // Create tiling_out [N.o,16,C1,H,C0]
  size_t ptr = 0;
  for (size_t i = 0; i < output.size; i++) {
    if (i == 0 and compileInfo.permute[0] != 0) {
      // Don't have dim N
      has_dim_n = false;
      tiling_output.shape[ptr] = 1;
      tiling_output.shape[ptr + 1] = compileInfo.pad_align_size;
      tiling_output.shape[ptr + 2] = output.shape[i];
      ptr += OFFSET_2;
      c1_index += OFFSET_2;
      c0_index += OFFSET_2;
    } else if (i == 0) {
      // Have dim N
      tiling_output.shape[ptr] = CeilDiv(output.shape[i], compileInfo.pad_align_size);
      tiling_output.shape[ptr + 1] = compileInfo.pad_align_size;
      ptr++;
      c1_index++;
      c0_index++;
    } else {
      // Normal dim
      tiling_output.shape[ptr] = output.shape[i];
    }
    ptr++;
  }
  tiling_output.SetSize(ptr);
  return InferTilingInput();
}

bool TransdataBN::IsConstRuntime() {
  if (compileInfo.is_const && (not compileInfo.is_const_compile)) {
    std::string pattern_str = std::to_string(CONST_KEY);
    tilingInfo.blk_dim = compileInfo.const_block_dims.at(pattern_str);
    return true;
  }
  return false;
}

bool TransdataBN::Strategy() {
  // Choose ShapeType and UBInfo
  if (compileInfo.ub_info.size() < computeType + 1) {
    return false;
  }
  if (compileInfo.ub_info[computeType].size() < shapeType + 1) {
    return false;
  }
  UBSize = compileInfo.ub_info[computeType][shapeType];
  return true;
}

bool TransdataBN::Filter() {
  size_t length = tiling_input.size;
  for (size_t i = 0; i < length; i++) {
    // not split 16,C (backward), not split 16,c1,c0 (forward)
    bool split_c = compileInfo.is_forward ? i == permute[c0_index] || i == permute[c1_index] : i == permute[c_index];
    if (i == length - 1 || split_c) {
      continue;
    }

    // not exceed UBSize [n, h, c, 16] -> [n, 16, h, c] (backward)
    // not exceed UBSize [n, c1, h, c0, 16] -> [n, 16, c1, h, c0] (forward)
    // forward would make c1\c0 in ub_inner
    int64_t base = Prod(tiling_input.shape, i + 1, length - 1);
    if (compileInfo.is_forward and permute[c1_index] < i + 1) {
      base *= tiling_input.shape[permute[c1_index]];
    }
    base = SetAlign(base, align_size * STRIDE_3);
    base *= tiling_input.shape[length - 1];
    if (base > UBSize) {
      continue;
    }
    // error index
    int64_t ptrO = VectorIndex(permute, i);
    if (ptrO < 0) {
      return false;
    }
    split_array[array_size].Set(i, static_cast<size_t>(ptrO));
    array_size++;
  }
  return array_size > 0;
}

bool TransdataBN::UBTilingProcess() {
  /* In Borrow-N-SCH, shape of tiling_input must be (N.o,h0,h1,h2,...,C,16),
   * shape of tiling_output must be (N.o,16,h0,h1,h2,...,C).The feature decides
   * split of Borrow-N-SCH must be once-tiling.
   * */
  bool tiling_success = false;
  while (array_size >= 1) {
    ptrI = split_array[array_size - 1].ptrA;
    ptrO = split_array[array_size - 1].ptrB;
    if (not OnceTiling()) {
      array_size--;
      continue;
    }
    CompareTiling();
    tiling_success = true;
    array_size--;
  }
  if (compileInfo.is_forward) {
    AdjustUBFactorForward();
  } else {
    AdjustUBFactorBackward();
  }
  return tiling_success;
}

bool TransdataBN::UBTiling() {
  /* In Borrow-N-SCH, shape of tiling_input must be (N.o,h0,h1,h2,...,C,16),
   * shape of tiling_output must be (N.o,16,h0,h1,h2,...,C).The feature decides
   * split of Borrow-N-SCH must be once-tiling.
   * */
  V_OP_TILING_CHECK(Filter(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Filter Failed"), return false);
  V_OP_TILING_CHECK(UBTilingProcess(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "UB-Tiling Failed"), return false);
  return true;
}

bool TransdataBN::BlockTiling() {
  AxisType type_array[tiling_output.size] = {UB_EXTERNAL};
  DiscriminationAxisType(type_array, tiling_output.size);
  BlkTilingProcess(type_array, tiling_output);
  return true;
}

void TransdataBN::BlkTilingProcess(const AxisType* type_array, const Shape& res) {
  // Forward: res is output, Backward: res is reshape
  core = 1;
  factorI = tilingInfo.ub_0_factor;
  factorO = tilingInfo.ub_1_factor;
  bool exceed_limit = false;
  int64_t dim_bound = 1;
  size_t block_idx = 0;
  size_t slide_idx = 0;

  // find split idx
  while (slide_idx < res.size) {
    if (core >= compileInfo.core_num) {
      exceed_limit = true;
      break;
    }
    if (type_array[slide_idx] == UB_EXTERNAL) {
      dim_bound = res.shape[slide_idx];
    } else if (type_array[slide_idx] == UB_FIRST_SPLIT) {
      dim_bound = CeilDiv(res.shape[slide_idx], factorI);
    } else if (type_array[slide_idx] == UB_SECOND_SPLIT) {
      dim_bound = CeilDiv(res.shape[slide_idx], factorO);
    } else {
      slide_idx++;
      continue;
    }
    core *= dim_bound;
    slide_idx++;
    block_idx = slide_idx;
  }

  tilingInfo.blk_idx = block_idx == 0 ? block_idx : block_idx - 1;
  if (not exceed_limit) {
    tilingInfo.blk_dim = core;
    tilingInfo.blk_factor = 1;
  } else {
    // static check
    dim_bound = dim_bound == 0 ? 1 : dim_bound;
    core = core / dim_bound;
    for (int64_t i = 1; i <= dim_bound; i++) {
      int64_t outer = core * CeilDiv(dim_bound, i);
      if (outer <= compileInfo.core_num) {
        tilingInfo.blk_factor = i;
        tilingInfo.blk_dim = outer;
        break;
      }
    }
  }
}

bool TransdataBN::WriteTilingData() {
  if (compileInfo.is_const && (not compileInfo.is_const_compile)) {
    // const runtime
    run_info.SetBlockDim(tilingInfo.blk_dim);
    run_info.SetTilingKey(CONST_KEY);
    return true;
  }

  if (compileInfo.is_const && compileInfo.is_const_compile) {
    // const compile
    run_info.AddTilingData(static_cast<int32_t>(computeType));
    run_info.AddTilingData(static_cast<int32_t>(shapeType));
    run_info.AddTilingData(static_cast<int32_t>(tilingInfo.blk_idx));
    run_info.AddTilingData(static_cast<int32_t>(VectorIndex(permute, tilingInfo.ub_0_idx)));
    run_info.AddTilingData(static_cast<int32_t>(tilingInfo.ub_1_idx));
    run_info.AddTilingData(static_cast<int32_t>(tilingInfo.blk_factor));
    run_info.AddTilingData(static_cast<int32_t>(tilingInfo.ub_0_factor));
    run_info.AddTilingData(static_cast<int32_t>(tilingInfo.ub_1_factor));
    run_info.AddTilingData(static_cast<uint32_t>(tilingInfo.blk_dim));
    return true;
  }

  // dynamic
  run_info.SetBlockDim(static_cast<uint32_t>(tilingInfo.blk_dim));
  run_info.SetTilingKey(static_cast<uint32_t>(CalcTilingKey()));
  // convert dim which is input after fused
  const Shape* res_shape = compileInfo.is_forward ? &input : &output;
  for (size_t i = 0; i < res_shape->size; i++) {
    bool skip_dim = is_reinterpret and i == res_shape->size - 1;
    if (not skip_dim) {
      run_info.AddTilingData(static_cast<int32_t>(res_shape->shape[i]));
      OP_LOGD(op_type.c_str(), "input shape : %d", res_shape->shape[i]);
    }
  }
  // convert factor
  run_info.AddTilingData(static_cast<int32_t>(tilingInfo.blk_factor));
  run_info.AddTilingData(static_cast<int32_t>(tilingInfo.ub_0_factor));

  return true;
}

bool TransdataBN::CalcTiling() {
  V_OP_TILING_CHECK((compileInfo.is_forward ? InitForward() : InitBackward()),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "UpdateValue Failed"), return false);
  if (not IsConstRuntime()) {
    V_OP_TILING_CHECK(Strategy(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ChooseStrategy Failed"), return false);
    V_OP_TILING_CHECK(UBTiling(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "UBTiling Failed"), return false);
    V_OP_TILING_CHECK(BlockTiling(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "BlockTiling Failed"), return false);
  }
  return true;
}

bool TransdataBN::DoTiling() {
  // main process
  bool ret = CalcTiling();
  return ret && WriteTilingData();
}
}  // namespace optiling
