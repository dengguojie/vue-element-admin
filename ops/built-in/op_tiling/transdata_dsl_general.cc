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
 * \file transdata_dsl_general.cpp
 * \brief
 */
#include "transdata_dsl_general.h"

namespace optiling {
using namespace transdata_dsl;

int64_t TransdataGeneral::CommonAlignLimit(int64_t factor) const {
  int64_t fp32_limit = 64;
  int64_t fp16_limit = 256;
  int64_t int8_limit = 1024;
  factor = factor == 0 ? DEFAULT : factor;
  switch (BLOCK / factor) {
    case FP32_BYTE:
      return fp32_limit;
    case FP16_BYTE:
      return fp16_limit;
    case INT8_BYTE:
      return int8_limit;
    default:
      return DEFAULT;
  }
}

int64_t TransdataGeneral::CalcTilingKey() {
  using namespace std;
  int64_t db = 0;
  int64_t is_forward = compileInfo.is_forward ? FORWARD_KEY : BACKWARD_KEY;
  size_t key_num = 7;
  int64_t pos[key_num] = {db,
                          is_forward,
                          static_cast<int64_t>(computeType),
                          static_cast<int64_t>(shapeType),
                          static_cast<int64_t>(tilingInfo.blk_idx),
                          VectorIndex(compileInfo.permute, tilingInfo.ub_0_idx),
                          static_cast<int64_t>(tilingInfo.ub_1_idx)};
  int64_t val[key_num] = {1000000000, 100000000, 10000000, 100000, 10000, 1000, 100};
  int64_t key = 0;
  for (size_t i = 0; i < key_num; i++) {
    key += pos[i] * val[i];
  }
  return key;
}

bool TransdataGeneral::ChooseType(int64_t dim_len, int64_t ub_size) const {
  int64_t factor = compileInfo.align_size;
  int64_t num_bit = BLOCK / factor;
  bool is_not_align = dim_len % factor != 0;
  bool is_legal_len = dim_len <= PACKET_SENDING_RATE / num_bit;
  is_legal_len = is_legal_len and (dim_len <= ub_size / num_bit / CommonAlignLimit(factor) / factor * factor);
  return is_legal_len and is_not_align;
}

bool TransdataGeneral::CheckValidSplit(size_t ptrA, size_t ptrB) const {
  /* Regulation:
   * 1. Forward: reshape + perm -> output
   * 2. Backward: input + perm -> reshape
   * 3. split as same as sch
   * */
  // VALID SPLIT
  int64_t shadow_ptrA = VectorIndex(compileInfo.permute, ptrA);
  if (shadow_ptrA < 0) {
    return false;
  }
  bool valid_split = compileInfo.permute[ptrB] <= ptrA and shadow_ptrA <= static_cast<int64_t>(ptrB);

  // NOT_EXCEED_UB
  int64_t base = Prod(tiling_input.shape, ptrA + 1, tiling_input.size);
  for (size_t idx = ptrB + 1; idx < tiling_output.size; idx++) {
    base = compileInfo.permute[idx] < ptrA ? base * tiling_output.shape[idx] : base;
  }
  bool not_exceed_ub = base <= UBSize;

  // DATA_MOVE
  if (is_data_move) {
    return valid_split && not_exceed_ub;
  }

  // SPLIT C0 (new_c0 base on input)
  size_t new_c0 = compileInfo.is_forward ? c0_index : compileInfo.permute[c0_index];
  bool not_split_c0 = new_c0 != ptrA and VectorIndex(compileInfo.permute, new_c0) != static_cast<int64_t>(ptrB);

  // Common align need tiling not split last dim
  bool valid_common_align = shapeType != CommonAlign;
  if (shapeType == CommonAlign) {
    if (compileInfo.is_forward) {
      // new_c0 base on input
      if (new_c0 != tiling_input.size - 1) {
        // input is (n,c,h), tiling_input is (n,c1,c0,h), common_align not split h
        size_t limit = tiling_input.size - 1;
        valid_common_align = ptrA != limit and compileInfo.permute[ptrB] != limit;
      } else {
        // input is (n,h,c), tiling_input is (n,h,c1,c0), common_align not split c1
        size_t limit = tiling_input.size - OFFSET_2;
        valid_common_align = ptrA < limit and compileInfo.permute[ptrB] < limit;
      }
    } else {
      // new_c0 base on input, need to adjust that make it base on output
      new_c0 = static_cast<size_t>(VectorIndex(compileInfo.permute, new_c0));
      if (new_c0 != tiling_output.size - 1) {
        // output is (n,c,h), tiling_output is (n,c1,c0,h), common_align not split h
        size_t limit = tiling_output.size - 1;
        valid_common_align = static_cast<size_t>(VectorIndex(compileInfo.permute, ptrA)) != limit and ptrB != limit;
      } else {
        // output is (n,h,c), tiling_output is (n,h,c1,c0), common_align not split c1
        size_t limit = tiling_output.size - OFFSET_2;
        valid_common_align = static_cast<size_t>(VectorIndex(compileInfo.permute, ptrA)) < limit and ptrB < limit;
      }
    }
  }

  // Template avoid emit_insn of pass
  bool isFp32 = (BLOCK / compileInfo.align_size) == FP32_BYTE;
  bool transpose_limit = tiling_input.shape[tiling_input.size - OFFSET_2] > FP32_TRANSPOSE_LIMIT;
  if (is_last_transpose and isFp32 and transpose_limit) {
    if (ptrA < tiling_input.size - OFFSET_2) {
      return false;
    }
  }
  return valid_split and not_split_c0 and not_exceed_ub and valid_common_align;
}

bool TransdataGeneral::InitUBFactorMTE3(int64_t lower, int64_t higher) {
  // Update factorO.
  // [lower, higher] is boundary of factorO.
  // MTE3 care about effect of tail factor.
  // Attention: the func assume mte2's init had been done.
  bool init_success = false;
  int64_t base_core = total_num / num_in_ub / tiling_output.shape[ptrO];
  base_core = compileInfo.permute[ptrO] == ptrI
                  ? base_core
                  : base_core / tiling_input.shape[ptrI] * CeilDiv(tiling_input.shape[ptrI], factorI);

  for (int64_t factor = lower; factor <= higher; factor++) {
    // check tail
    GetOutputRealTail(ptrO, factor, mte3);
    bool tail_is_legal = mte3.tailLen == 0 or mte3.tailLen >= compileInfo.align_size;
    // check main
    int64_t core_num = base_core * CeilDiv(tiling_output.shape[ptrO], factor);
    bool main_is_legal = mte3.mainLen >= compileInfo.align_size or core_num == 1;
    if (tail_is_legal and main_is_legal) {
      factorO = factor;
      init_success = true;
      if (factor * mte3.virLen >= mte_rate) {
        break;
      }
    }
  }
  return init_success;
}

bool TransdataGeneral::InitUBFactorMTE2(int64_t lower, int64_t higher) {
  // Update factorI
  // [lower, higher] is boundary of factorI
  // MTE2 don't consider effect of tail factor.
  for (int64_t i = lower; i <= higher; i++) {
    if (i * mte2.virLen >= mte_rate) {
      factorI = i;
      return true;
    }
    factorI = i;
  }
  return true;
}

bool TransdataGeneral::OnceTiling() {
  // ptrI and ptrO split same axis, use ptrO and tiling_output
  bool run_out_ub = tiling_output.shape[ptrO] * num_in_ub >= UBSize;
  int64_t bound = run_out_ub ? UBSize / num_in_ub : tiling_output.shape[ptrO];
  // Axes which do last-transpose need special align, eg:(m,n)->(n,m):
  // if ptrO point m, m need align for (32,16,128) -> (int8,fp16,fp32),
  // if ptrO point n, n need align for (32,16,8) -> (int8,fp16,fp32).
  // Due to insn of pass, 128 is fixed.
  if (is_last_transpose and ptrO == tiling_output.size - OFFSET_2) {
    bound = bound / compileInfo.align_size * compileInfo.align_size;
  } else if (is_last_transpose and ptrO == tiling_output.size - 1) {
    bound = ele_byte == FP16_BYTE or ele_byte == INT8_BYTE ? bound / compileInfo.align_size * compileInfo.align_size
                                                           : STRIDE_16 * compileInfo.align_size;
  }

  // Init Tiling
  bool init_success = false;
  for (int64_t factor = 1; factor <= bound; factor++) {
    // check tail
    GetOutputRealTail(ptrO, factor, mte3);
    bool tail_is_legal = mte3.tailLen == 0 or mte3.tailLen >= compileInfo.align_size;
    // check main
    core = total_num / num_in_ub / tiling_output.shape[ptrO] * CeilDiv(tiling_output.shape[ptrO], factor);
    bool main_is_legal = mte3.mainLen >= compileInfo.align_size or core == 1;
    if (tail_is_legal and main_is_legal) {
      factorO = factor;
      init_success = true;
      if (factor * mte3.virLen >= mte_rate and factor * mte2.virLen >= mte_rate) {
        break;
      }
    }
  }

  // Check init
  if (not init_success) {
    return false;
  }

  // Adjust MTE3
  core = total_num / num_in_ub / tiling_output.shape[ptrO] * CeilDiv(tiling_output.shape[ptrO], factorO);
  if (core > compileInfo.core_num) {
    // extra core can improve UB usage
    for (int64_t factor = factorO; factor <= bound; factor++) {
      // Check tail
      GetOutputRealTail(ptrO, factor, mte3);
      bool tail_is_legal = mte3.tailLen == 0 or mte3.tailLen >= compileInfo.align_size;
      if (tail_is_legal) {
        core = total_num / num_in_ub / tiling_output.shape[ptrO] * CeilDiv(tiling_output.shape[ptrO], factor);
        factorO = factor;
        if (core <= compileInfo.core_num) {
          break;
        }
      }
    }
  }
  percent = static_cast<float>(factorO * num_in_ub) / static_cast<float>(UBSize);
  factorI = factorO;
  mte2.virLen = factorI * mte2.virLen;
  mte3.virLen = factorO * mte3.virLen;
  return true;
}

bool TransdataGeneral::TwiceTiling() {
  /* Forward:
   * step_0 : Init MTE2/MTE3, let MTE2/MTE3 arrive mte2_rate
   * step_1 : Fix MTE3, adjust MTE2
   * step_2 : Fix MTE2, adjust MTE3
   *
   * Backward:
   * step_0 : Init MTE2/MTE3, let MTE2/MTE3 arrive mte2_rate
   * step_1 : Fix MTE2, adjust MTE3
   * step_2 : Fix MTE3, adjust MTE2
   * */
  // step_0: Init MTE2
  int64_t bound = UBSize / num_in_ub;
  int64_t boundI = bound > tiling_input.shape[ptrI] ? tiling_input.shape[ptrI] : bound;
  bool is_success = InitUBFactorMTE2(1, boundI);
  // step_0: Init MTE3
  int64_t boundO = bound / factorI > tiling_output.shape[ptrO] ? tiling_output.shape[ptrO] : bound / factorI;
  is_success = is_success and InitUBFactorMTE3(1, boundO);

  // Check init
  if (not is_success) {
    return false;
  }

  if (compileInfo.is_forward) {
    // step_1: Fix MTE3, adjust MTE2
    boundI = bound / factorO > tiling_input.shape[ptrI] ? tiling_input.shape[ptrI] : bound / factorO;
    AdjustUBFactorMTE2(factorI, boundI);
    // step_2: Fix MTE2, adjust MTE3
    boundO = bound / factorI > tiling_output.shape[ptrO] ? tiling_output.shape[ptrO] : bound / factorI;
    AdjustUBFactorMTE3(factorO, boundO);
  } else {
    // step_1 : Fix MTE2, adjust MTE3
    boundO = bound / factorI > tiling_output.shape[ptrO] ? tiling_output.shape[ptrO] : bound / factorI;
    AdjustUBFactorMTE3(factorO, boundO);
    // step_2 : Fix MTE3, adjust MTE2
    boundI = bound / factorO > tiling_input.shape[ptrI] ? tiling_input.shape[ptrI] : bound / factorO;
    AdjustUBFactorMTE2(factorI, boundI);
  }
  // Result
  mte2.virLen = factorI * mte2.virLen;
  mte3.virLen = factorO * mte3.virLen;
  percent = static_cast<float>(factorO * factorI * num_in_ub) / static_cast<float>(UBSize);
  return true;
}

void TransdataGeneral::SetStorageAlign(Shape& input_shape, Shape& output_shape, const Shape& ori_input) const {
  size_t length = input_shape.size;
  std::copy(ori_input.shape, ori_input.shape + length, input_shape.shape);

  // align for mte2(pad had been done)
  input_shape.shape[length - 1] = SetAlign(input_shape.shape[length - 1], compileInfo.align_size);
  // align for different instruction(5HD,NZ)
  if (is_last_transpose) {
    bool isFp16 = BLOCK / compileInfo.align_size == FP16_BYTE;
    bool isInt8 = BLOCK / compileInfo.align_size == INT8_BYTE;
    if (isFp16 or isInt8) {
      input_shape.shape[length - OFFSET_2] = SetAlign(input_shape.shape[length - OFFSET_2], compileInfo.align_size);
    } else {
      // FP32(128),INT32(128),INT64(64)
      input_shape.shape[length - OFFSET_2] =
          SetAlign(input_shape.shape[length - OFFSET_2], STRIDE_16 * compileInfo.align_size);
    }
  }

  // input_shape + perm -> output_shape
  for (size_t i = 0; i < length; i++) {
    output_shape.shape[i] = input_shape.shape[compileInfo.permute[i]];
  }
}

void TransdataGeneral::AdjustUBFactorMTE2(int64_t lower, int64_t higher) {
  // Update factorI
  // [lower, higher] is boundary of factorI
  // Attention: the func only work in twice split
  int64_t ub_i_outer = CeilDiv(tiling_input.shape[ptrI], factorI);
  int64_t ub_o_outer = CeilDiv(tiling_output.shape[ptrO], factorO);
  int64_t base_core = total_num / num_in_ub / tiling_output.shape[ptrO] / tiling_input.shape[ptrI];
  core = base_core * ub_i_outer * ub_o_outer;
  if (core > compileInfo.core_num) {
    // extra core can improve UB usage
    for (int64_t i = lower; i <= higher; i++) {
      factorI = i;
      ub_i_outer = CeilDiv(tiling_input.shape[ptrI], i);
      core = base_core * ub_i_outer * ub_o_outer;
      if (core <= compileInfo.core_num) {
        break;
      }
    }
  }
}

void TransdataGeneral::AdjustUBFactorMTE3(int64_t lower, int64_t higher) {
  // Update factorO
  // [lower, higher] is boundary of factorI
  // Attention: the func only work in twice split
  int64_t ub_o_outer = CeilDiv(tiling_output.shape[ptrO], factorO);
  int64_t ub_i_outer = CeilDiv(tiling_input.shape[ptrI], factorI);
  int64_t base_core = total_num / num_in_ub / tiling_output.shape[ptrO] / tiling_input.shape[ptrI];
  core = base_core * ub_i_outer * ub_o_outer;
  if (core > compileInfo.core_num) {
    // extra core can improve UB usage
    for (int64_t i = lower; i <= higher; i++) {
      // check tail
      GetOutputRealTail(ptrO, i, mte3);
      bool tail_is_legal = mte3.tailLen == 0 or mte3.tailLen >= compileInfo.align_size;
      if (tail_is_legal) {
        factorO = i;
        ub_o_outer = CeilDiv(tiling_output.shape[ptrO], i);
        core = base_core * ub_o_outer * ub_i_outer;
        if (core <= compileInfo.core_num) {
          break;
        }
      }
    }
  }
}

void TransdataGeneral::CompareTiling() {
  if (tilingInfo.percent == 0) {
    // First Init
    tilingInfo.UBInfoSet(ptrI, ptrO, factorI, factorO, mte2.virLen, mte3.virLen, percent, core, split_once);
  } else {
    // Compare
    int64_t new_core = percent >= GENERAL_THRESHOLD and core > compileInfo.core_num ? compileInfo.core_num : core;
    int64_t old_core = tilingInfo.percent >= GENERAL_THRESHOLD and tilingInfo.core > compileInfo.core_num
                           ? compileInfo.core_num
                           : tilingInfo.core;
    int64_t new_core_distance = abs(new_core - compileInfo.core_num);
    int64_t old_core_distance = abs(old_core - compileInfo.core_num);
    if (new_core_distance < old_core_distance) {
      tilingInfo.UBInfoSet(ptrI, ptrO, factorI, factorO, mte2.virLen, mte3.virLen, percent, core, split_once);
    } else if (new_core_distance == old_core_distance) {
      if (split_once and not tilingInfo.split_once) {
        tilingInfo.UBInfoSet(ptrI, ptrO, factorI, factorO, mte2.virLen, mte3.virLen, percent, core, split_once);
      } else if (split_once and tilingInfo.split_once) {
        if (mte3.virLen > tilingInfo.mte3_burst_len) {
          tilingInfo.UBInfoSet(ptrI, ptrO, factorI, factorO, mte2.virLen, mte3.virLen, percent, core, split_once);
        }
      }
    }
  }
}

void TransdataGeneral::AdjustUBFactor() {
  if (tilingInfo.split_once) {
    // deal factor + deal pure data move
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
}

void TransdataGeneral::GetOutputRealTail(int64_t ptr, int64_t factor, MTEInfo& mte) {
  if (compileInfo.is_forward) {
    // In forward: don't mapping out
    int64_t baseLen = Prod(output.shape, ptr + 1, output.size);
    factor = output.shape[ptr] > factor ? factor : output.shape[ptr];
    mte.mainLen = factor * baseLen;
    mte.tailLen = (output.shape[ptr] % factor) * baseLen;
  } else {
    // In backward: r_mapping_o: out mapping reshape, v of r_mapping_o is out
    size_t out_ptr = r_mapping_o[ptr];
    factor = is_data_move ? factor : out_ptr == c1_index ? factor * compileInfo.pad_align_size : factor;
    factor = output.shape[out_ptr] > factor ? factor : output.shape[out_ptr];
    int64_t baseLen = Prod(output.shape, out_ptr + 1, output.size);
    mte.mainLen = factor * baseLen;
    mte.tailLen = (output.shape[out_ptr] % factor) * baseLen;
  }
}

void TransdataGeneral::DiscriminationAxisType(AxisType* type_array, size_t length) {
  // Assure type of axis is belong to internal of UB or not.
  // type_array based on output
  for (size_t i = tilingInfo.ub_0_idx + 1; i < length; i++) {
    // deal input
    type_array[VectorIndex(compileInfo.permute, i)] = UB_INTERNAL;
  }
  for (size_t i = tilingInfo.ub_1_idx + 1; i < length; i++) {
    // deal output
    type_array[i] = UB_INTERNAL;
  }

  // part of split-axis belong to ub_external, part belong to ub_internal.
  type_array[VectorIndex(compileInfo.permute, tilingInfo.ub_0_idx)] = UB_FIRST_SPLIT;
  type_array[tilingInfo.ub_1_idx] = UB_SECOND_SPLIT;
}

bool TransdataGeneral::Init() {
  // 1. Match length and array
  // 2. Last-Transpose or not
  if (input.size < 1 or output.size < 1 or reshape.size < 1) {
    return false;
  }

  // Check is pure data move branch or not
  // if forward, index of c0 based on reshape(input -> reshape ~transpose~> output)
  // if backward, index of c0 based on reshape(output <- reshape <~transpose~ input)
  size_t root_ptr = 0;
  for (size_t i = 0; i < compileInfo.src_pad.size(); i++) {
    if (compileInfo.src_pad[i] == EXISTED_C1C0) {
      is_data_move = false;
      c1_index = root_ptr;
      c0_index = root_ptr + 1;
      r_mapping_o[root_ptr] = i;
      r_mapping_o[root_ptr + 1] = i;
      root_ptr += OFFSET_2;
    } else {
      r_mapping_o[root_ptr] = i;
      root_ptr++;
    }
  }

  if (compileInfo.is_forward) {
    is_last_transpose = compileInfo.permute[output.size - 1] != output.size - 1;
  } else {
    is_last_transpose = compileInfo.permute[input.size - 1] != input.size - 1;
  }
  ele_byte = BLOCK / compileInfo.align_size;
  mte_rate = PACKET_SENDING_RATE / ele_byte;
  return true;
}

bool TransdataGeneral::IsConstRuntime() {
  if (compileInfo.is_const && (not compileInfo.is_const_compile)) {
    std::string pattern_str = std::to_string(CONST_KEY);
    tilingInfo.blk_dim = compileInfo.const_block_dims.at(pattern_str);
    return true;
  }
  return false;
}

bool TransdataGeneral::Strategy() {
  // Choose ShapeType and UBInfo
  int64_t last_dim = compileInfo.is_forward ? input.shape[input.size - 1] : output.shape[output.size - 1];
  if (compileInfo.ub_info.size() < computeType + 1) {
    return false;
  }
  if (compileInfo.ub_info[computeType].size() < shapeType + 1) {
    return false;
  }
  shapeType = ChooseType(last_dim, compileInfo.ub_info[computeType][shapeType]) ? CommonAlign : StorageAlign;
  UBSize = compileInfo.ub_info[computeType][shapeType];
  return true;
}

bool TransdataGeneral::Filter() {
  // ptrA: index of input
  // ptrB: index of output
  for (size_t ptrA = 0; ptrA < tiling_input.size; ptrA++) {
    for (size_t ptrB = 0; ptrB < tiling_output.size; ptrB++) {
      if (CheckValidSplit(ptrA, ptrB)) {
        split_array[array_size].Set(ptrA, ptrB);
        array_size++;
      }
    }
  }
  return array_size > 0;
}

bool TransdataGeneral::UBTilingProcess() {
  int64_t loop = array_size;
  total_num = Prod(tiling_input.shape, 0, tiling_input.size);
  bool tiling_success = false;
  while (loop >= 1) {
    ptrI = split_array[loop - 1].ptrA;
    ptrO = split_array[loop - 1].ptrB;
    split_once = compileInfo.permute[ptrO] == ptrI;
    mte2.virLen = Prod(tiling_input.shape, ptrI + 1, tiling_input.size);
    mte3.virLen = Prod(tiling_output.shape, ptrO + 1, tiling_output.size);
    num_in_ub = mte2.virLen;
    for (size_t i = ptrO + 1; i < tiling_output.size; i++) {
      num_in_ub = compileInfo.permute[i] < ptrI ? num_in_ub * tiling_output.shape[i] : num_in_ub;
    }

    // Tiling
    if (not(split_once ? OnceTiling() : TwiceTiling())) {
      loop--;
      continue;
    }

    // Compare
    CompareTiling();
    tiling_success = true;
    loop--;
  }
  AdjustUBFactor();
  return tiling_success;
}

bool TransdataGeneral::UBTiling() {
  /*
   * <Forward> ------------------<Backward>
   * input-<PAD>-pad_shape-<RESHAPE>-<TRANSPOSE>-out
   * 1. In forward, output is mte3 that is align, input is mte2 that maybe not align.
   * 2. In backward, input is mte2 that is align, output is mte3 that may be not align.
   * 3. In pure data move, mte3.mainLen and mte3.tailLen need bigger than align size.
   * 4. In backward, MTE3 need protect mte3.mainLen Core and mte3.tailLen bigger than align_size.
   */
  if (compileInfo.is_forward) {
    // reshape -> out
    tiling_input.SetSize(output.size);
    tiling_output.SetSize(output.size);
    SetStorageAlign(tiling_input, tiling_output, reshape);
  } else {
    // input -> reshape -> ac_out
    tiling_input.SetSize(input.size);
    tiling_output.SetSize(input.size);
    SetStorageAlign(tiling_input, tiling_output, input);
  }
  V_OP_TILING_CHECK(Filter(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Filter Failed"), return false);
  V_OP_TILING_CHECK(UBTilingProcess(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "F-Tiling Failed"), return false);
  return true;
}

bool TransdataGeneral::BlockTiling() {
  if (compileInfo.is_forward) {
    AxisType type_array[output.size] = {UB_EXTERNAL};
    DiscriminationAxisType(type_array, output.size);
    BlkTilingProcess(type_array, output);
  } else {
    AxisType type_array[reshape.size] = {UB_EXTERNAL};
    DiscriminationAxisType(type_array, reshape.size);
    BlkTilingProcess(type_array, reshape);
  }
  return true;
}

void TransdataGeneral::BlkTilingProcess(const AxisType* type_array, const Shape& res) {
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

bool TransdataGeneral::WriteTilingData() {
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
    run_info.AddTilingData(static_cast<int32_t>(VectorIndex(compileInfo.permute, tilingInfo.ub_0_idx)));
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
  Shape* res_shape = compileInfo.is_forward ? &input : &output;
  for (size_t i = 0; i < compileInfo.unknown_dims.size(); i++) {
    run_info.AddTilingData(static_cast<int32_t>(res_shape->shape[compileInfo.unknown_dims[i]]));
    OP_LOGD(op_type.c_str(), "input shape : %d", res_shape->shape[i]);
  }
  // convert factor
  run_info.AddTilingData(static_cast<int32_t>(tilingInfo.blk_factor));
  run_info.AddTilingData(static_cast<int32_t>(tilingInfo.ub_0_factor));
  if (not tilingInfo.split_once) {
    run_info.AddTilingData(static_cast<int32_t>(tilingInfo.ub_1_factor));
  }

  return true;
}

bool TransdataGeneral::CalcTiling() {
  V_OP_TILING_CHECK(Init(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "UpdateValue Failed"), return false);
  if (not IsConstRuntime()) {
    V_OP_TILING_CHECK(Strategy(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ChooseStrategy Failed"), return false);
    V_OP_TILING_CHECK(UBTiling(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "UBTiling Failed"), return false);
    V_OP_TILING_CHECK(BlockTiling(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "BlockTiling Failed"), return false);
  }
  return true;
}

bool TransdataGeneral::DoTiling() {
  // main process
  bool ret = CalcTiling();
  return ret && WriteTilingData();
}
}  // namespace optiling
