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
 * \file transdata_dsl_general.cc
 * \brief
 */
#include "transdata_dsl_general.h"

namespace optiling {
using namespace transdata_dsl;

bool TransdataGeneral::CommonAlignLimit(int64_t ub_size, int64_t last_dim_align_value, int64_t num_bit) const {
  int64_t last_dim_max = 1;
  int64_t co2co = 1;
  switch (num_bit) {
    case FP32_BYTE:
      last_dim_max = STRIDE_168;
      co2co = STRIDE_128;
      break;
    case FP16_BYTE:
      last_dim_max = STRIDE_160;
      co2co = STRIDE_256;
      break;
    case INT8_BYTE:
      last_dim_max = STRIDE_64;
      co2co = STRIDE_1024;
      break;
    default:
      return false;
  }
  return last_dim_align_value <= last_dim_max && co2co * last_dim_align_value <= ub_size;
}

int64_t TransdataGeneral::CalcTilingKey() {
  using namespace std;
  int64_t db = 0;
  int64_t is_forward = compileInfo.is_forward ? FORWARD_KEY : BACKWARD_KEY;
  size_t key_num = 9;
  int64_t pos[key_num] = {db,
                          is_forward,
                          static_cast<int64_t>(computeType),
                          static_cast<int64_t>(shapeType),
                          static_cast<int64_t>(tilingInfo.blk_idx),
                          VectorIndex(compileInfo.permute, tilingInfo.ub_0_idx),
                          static_cast<int64_t>(tilingInfo.ub_1_idx),
                          static_cast<int64_t>(transposeWork),
                          static_cast<int64_t>(avoidBCWork)};
  int64_t val[key_num] = {1000000000, 100000000, 10000000, 1000000, 100000, 10000, 1000, 100, 10};
  int64_t key = 0;
  for (size_t i = 0; i < key_num; i++) {
    key += pos[i] * val[i];
  }
  return key;
}

int64_t TransdataGeneral::CommonRefineFactor(int64_t ori_factor, size_t ptr) {
  // factor: factor
  // ptr: based on tiling-output
  int64_t dimBound = tiling_output.shape[ptr];
  size_t length = tiling_output.size - 1;
  int64_t factor = ori_factor;
  bool split_last_dim = ptr == length || compileInfo.permute[ptr] == length;
  if (split_last_dim && factor > compileInfo.align_size && dimBound > compileInfo.align_size) {
    dimBound = dimBound / compileInfo.align_size;
    factor = factor / compileInfo.align_size;
    factor = REFINE(dimBound, factor);
    factor *= compileInfo.align_size;
  } else {
    factor = REFINE(dimBound, factor);
  }
  GetOutputRealTail(ptr, factor, mte3);
  bool tail_is_legal = mte3.tailLen == 0 || mte3.tailLen >= compileInfo.align_size;
  return tail_is_legal ? factor : ori_factor;
}

int64_t TransdataGeneral::AxisValueInUB(size_t ptr) const {
  // ptr based on tiling-input
  // eg: input is [16,32000,16], split 32000 as 100*32, 32000 in UB is 32.
  int64_t result = tilingInfo.ub_0_idx == ptr ?
                   tilingInfo.ub_0_factor : compileInfo.permute[tilingInfo.ub_1_idx] == ptr ?
                   tilingInfo.ub_1_factor : tiling_input.shape[ptr];
  return result;
}

bool TransdataGeneral::ChooseType(int64_t dim_len, int64_t ub_size) const {
  int64_t num_bit = BLOCK / compileInfo.align_size;
  bool is_not_align = dim_len % compileInfo.align_size != 0;
  bool is_legal_len = dim_len <= PACKET_SENDING_RATE / num_bit;
  bool is_common_align_ok = CommonAlignLimit(ub_size, SetAlign(dim_len, compileInfo.align_size), num_bit);
  return is_legal_len && is_not_align && is_common_align_ok;
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
  bool valid_split = compileInfo.permute[ptrB] <= ptrA && shadow_ptrA <= static_cast<int64_t>(ptrB);

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
  bool not_split_c0 = new_c0 != ptrA && VectorIndex(compileInfo.permute, new_c0) != static_cast<int64_t>(ptrB);

  // Best VOR in n-last-transpose(5HD\NZ)
  if (avoid_bc && !is_last_transpose) {
    size_t c1_idx = compileInfo.is_forward ? c1_index : compileInfo.permute[c1_index];
    size_t h_idx = compileInfo.is_forward ? c1_idx - 1 : c1_idx + 1;
    int64_t c1 = tiling_input.shape[c1_idx];
    int64_t h = tiling_input.shape[h_idx];
    int64_t c0 = tiling_input.shape[tiling_input.size - 1];
    int64_t stride = 255 / (c0 / compileInfo.align_size);
    if (c1 > stride && (ptrA != c1_idx && compileInfo.permute[ptrB] != c1_idx)) return false;
    if (h > stride && (ptrA != h_idx && compileInfo.permute[ptrB] != h_idx)) return false;
  }

  // Common align need tiling not split last dim
  bool valid_common_align = shapeType != COMMON_ALIGN;
  if (shapeType == COMMON_ALIGN) {
    if (compileInfo.is_forward) {
      // new_c0 base on input
      if (new_c0 != tiling_input.size - 1) {
        // input is (n,c,h), tiling_input is (n,c1,c0,h), common_align not split h
        size_t limit = tiling_input.size - 1;
        valid_common_align = ptrA != limit && compileInfo.permute[ptrB] != limit;
      } else {
        // input is (n,h,c), tiling_input is (n,h,c1,c0), common_align not split c1
        size_t limit = tiling_input.size - OFFSET_2;
        valid_common_align = ptrA < limit && compileInfo.permute[ptrB] < limit;
      }
    } else {
      // new_c0 base on input, need to adjust that make it base on output
      new_c0 = static_cast<size_t>(VectorIndex(compileInfo.permute, new_c0));
      if (new_c0 != tiling_output.size - 1) {
        // output is (n,c,h), tiling_output is (n,c1,c0,h), common_align not split h
        size_t limit = tiling_output.size - 1;
        valid_common_align = static_cast<size_t>(VectorIndex(compileInfo.permute, ptrA)) != limit && ptrB != limit;
      } else {
        // output is (n,h,c), tiling_output is (n,h,c1,c0), common_align not split c1
        size_t limit = tiling_output.size - OFFSET_2;
        valid_common_align = static_cast<size_t>(VectorIndex(compileInfo.permute, ptrA)) < limit && ptrB < limit;
      }
    }
  }

  // Template avoid emit_insn of pass
  bool isFp32 = (BLOCK / compileInfo.align_size) == FP32_BYTE;
  bool transpose_limit = tiling_input.shape[tiling_input.size - OFFSET_2] > FP32_TRANSPOSE_LIMIT;
  if (is_last_transpose && isFp32 && transpose_limit) {
    if (ptrA < tiling_input.size - OFFSET_2) {
      return false;
    }
  }
  return valid_split && not_split_c0 && not_exceed_ub && valid_common_align;
}

bool TransdataGeneral::InitUBFactorMTE3(int64_t lower, int64_t higher) {
  // Update factorO.
  // [lower, higher] is boundary of factorO.
  // MTE3 care about effect of tail factor.
  // Attention: the func assume mte2's init had been done.
  bool init_success = false;
  int64_t base_core = total_num / tiling_in_ub / tiling_output.shape[ptrO];
  base_core = compileInfo.permute[ptrO] == ptrI
                  ? base_core
                  : base_core / tiling_input.shape[ptrI] * CeilDiv(tiling_input.shape[ptrI], factorI);

  for (int64_t factor = lower; factor <= higher; factor++) {
    // check tail
    GetOutputRealTail(ptrO, factor, mte3);
    bool tail_is_legal = mte3.tailLen == 0 || mte3.tailLen >= compileInfo.align_size;
    // check main
    int64_t core_num = base_core * CeilDiv(tiling_output.shape[ptrO], factor);
    bool main_is_legal = mte3.mainLen >= compileInfo.align_size || core_num == 1;
    if (tail_is_legal && main_is_legal) {
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
  bool run_out_ub = tiling_output.shape[ptrO] * tiling_in_ub >= UBSize;
  int64_t bound = run_out_ub ? UBSize / tiling_in_ub : tiling_output.shape[ptrO];
  // Axes which do last-transpose need special align, eg:(m,n)->(n,m):
  // if ptrO point m, m need align for (32,16,128) -> (int8,fp16,fp32),
  // if ptrO point n, n need align for (32,16,8) -> (int8,fp16,fp32).
  // Due to insn of pass, 128 is fixed.
  if (is_last_transpose && ptrO == tiling_output.size - OFFSET_2) {
    bound = bound / compileInfo.align_size * compileInfo.align_size;
  } else if (is_last_transpose && ptrO == tiling_output.size - 1) {
    bound = bound / compileInfo.align_size * compileInfo.align_size;
    if (ele_byte != FP16_BYTE && ele_byte != INT8_BYTE) {
      bound = STRIDE_16 * compileInfo.align_size;
    }
  }

  // Init Tiling
  bool init_success = false;
  for (int64_t factor = 1; factor <= bound; factor++) {
    factorO = factor;
    factorI = factor;
    // check tail and main
    GetOutputRealTail(ptrO, factor, mte3);
    bool check_tail = mte3.tailLen == 0 || mte3.tailLen >= compileInfo.align_size;
    bool check_main = mte3.mainLen >= compileInfo.align_size;
    core = total_num / tiling_in_ub / tiling_output.shape[ptrO] * CeilDiv(tiling_output.shape[ptrO], factor);
    check_main = check_main || core == 1;
    if (check_main && check_tail) {
      init_success = true;
      bool mte2_arrive_package_rate = factor * mte2.virLen >= mte_rate;
      bool mte3_arrive_package_rate = factor * mte3.virLen >= mte_rate;
      if (mte2_arrive_package_rate && mte3_arrive_package_rate) {
        break;
      }
    }
  }

  // Choose SINGLE_CORE_MODE
  if (!init_success) {
    core = 1;
    mte2.virLen = factorI * mte2.virLen;
    mte3.virLen = factorO * mte3.virLen;
    percent = static_cast<float>(factorO * tiling_in_ub) / static_cast<float>(UBSize);
    return false;
  }

  // Adjust MTE3
  core = total_num / tiling_in_ub / tiling_output.shape[ptrO] * CeilDiv(tiling_output.shape[ptrO], factorO);
  if (core > compileInfo.core_num) {
    // extra core can improve UB usage
    for (int64_t factor = factorO; factor <= bound; factor++) {
      // Check tail
      GetOutputRealTail(ptrO, factor, mte3);
      bool tail_is_legal = mte3.tailLen == 0 || mte3.tailLen >= compileInfo.align_size;
      if (tail_is_legal) {
        core = total_num / tiling_in_ub / tiling_output.shape[ptrO] * CeilDiv(tiling_output.shape[ptrO], factor);
        factorO = factor;
        if (core <= compileInfo.core_num) {
          break;
        }
      }
    }
  }

  // Avoid VOR Conflict
  factorI = factorO;
  VectorOptimization();
  if (!AvoidBankConflict()) {
    return false;
  }
  UpdateCore();
  percent = static_cast<float>(factorO * tiling_in_ub) / static_cast<float>(UBSize);
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
  int64_t bound = UBSize / tiling_in_ub;
  int64_t boundI = bound > tiling_input.shape[ptrI] ? tiling_input.shape[ptrI] : bound;
  bool is_success = InitUBFactorMTE2(1, boundI);
  // step_0: Init MTE3
  int64_t boundO = bound / factorI;
  boundO = boundO > tiling_output.shape[ptrO] ? tiling_output.shape[ptrO] : boundO;
  is_success = is_success && InitUBFactorMTE3(1, boundO);

  // Check init
  if (!is_success) {
    core = 1;
    mte2.virLen = factorI * mte2.virLen;
    mte3.virLen = factorO * mte3.virLen;
    percent = static_cast<float>(factorO * factorI * tiling_in_ub) / static_cast<float>(UBSize);
    return false;
  }

  if (compileInfo.is_forward) {
    // step_1: Fix MTE3, adjust MTE2
    boundI = bound / factorO;
    boundI = boundI > tiling_input.shape[ptrI] ? tiling_input.shape[ptrI] : boundI;
    AdjustUBFactorMTE2(factorI, boundI);
    // step_2: Fix MTE2, adjust MTE3
    boundO = bound / factorI;
    boundO = boundO > tiling_output.shape[ptrO] ? tiling_output.shape[ptrO] : boundO;
    AdjustUBFactorMTE3(factorO, boundO);
  } else {
    // step_1 : Fix MTE2, adjust MTE3
    boundO = bound / factorI;
    boundO = boundO > tiling_output.shape[ptrO] ? tiling_output.shape[ptrO] : boundO;
    AdjustUBFactorMTE3(factorO, boundO);
    // step_2 : Fix MTE3, adjust MTE2
    boundI = bound / factorO;
    boundI = boundI > tiling_input.shape[ptrI] ? tiling_input.shape[ptrI] : boundI;
    AdjustUBFactorMTE2(factorI, boundI);
  }
  // Result
  // Avoid VOR Conflict
  VectorOptimization();
  if (!AvoidBankConflict()) {
    return false;
  }
  UpdateCore();
  mte2.virLen = factorI * mte2.virLen;
  mte3.virLen = factorO * mte3.virLen;
  percent = static_cast<float>(factorO * factorI * tiling_in_ub) / static_cast<float>(UBSize);
  return true;
}

void TransdataGeneral::VOROptimize(int64_t& a, int64_t& b, int64_t extent, int64_t stride) const {
  // a,b:current value and a >= b.
  // extent: maximum value of b.
  int64_t multi = CeilDiv(a, stride);
  // static-check
  b = b == 0 ? 1 : b;
  extent = extent == 0 ? 1 : extent;
  multi = multi == 0 ? 1 : multi;
  if (extent / b >= multi) {
    a = a / multi;
    b = b * multi;
  } else {
    int64_t temp = a * b / extent;
    int64_t temp_b = extent;
    int64_t temp_a = temp < stride ? temp : stride;
    float rate = float(temp_a * temp_b) / float(a * b);
    float threshold = 0.8;
    if (rate >= threshold) {
      a = temp_a;
      b = temp_b;
    }
  }
}

void TransdataGeneral::UpdateCore() {
  core = total_num / tiling_in_ub / tiling_output.shape[ptrO];
  if (!split_once) {
    core = core / tiling_input.shape[ptrI];
    core = core * CeilDiv(tiling_input.shape[ptrI], factorI);
  }
  core = core * CeilDiv(tiling_output.shape[ptrO], factorO);
}

void TransdataGeneral::VectorOptimization() {
  // Insn of vor Optimization: make [a,b,c0] -> [b,a,c0], block-stride not exceed HardwareLimitation.
  // Couldn't do adapt.
  if (!avoid_bc || (!ptrISplitT && !ptrOSplitT) || is_last_transpose) {
    return;
  }

  int64_t dimA = tiling_input.shape[ptrI];
  int64_t dimB = tiling_input.shape[compileInfo.permute[ptrO]];
  int64_t c0 = tiling_input.shape[tiling_input.size - 1];
  int64_t stride = 255 / (c0 / compileInfo.align_size);
  // Don't need adapt.
  if (factorI <= stride && factorO <= stride) {
    return;
  }

  // SolutionA: Split-Once
  if (split_once) {
    float threshold = 0.8;
    float rate = static_cast<float>(stride) / static_cast<float>(factorI);
    if (rate >= threshold) {
      factorI = stride;
      factorO = stride;
    }
    return;
  }

  // SolutionB: Split-Twice
  if (factorI >= factorO) {
    VOROptimize(factorI, factorO, dimB, stride);
  } else {
    VOROptimize(factorO, factorI, dimA, stride);
  }
}

void TransdataGeneral::AlignVNC(Shape& input_shape) const {
  bool isFp16 = BLOCK / compileInfo.align_size == FP16_BYTE;
  bool isInt8 = BLOCK / compileInfo.align_size == INT8_BYTE;
  size_t length = input_shape.size;
  if (isFp16 || isInt8) {
    input_shape.shape[length - OFFSET_2] = SetAlign(input_shape.shape[length - OFFSET_2], compileInfo.align_size);
  } else {
    // FP32(128),INT32(128),INT64(64)
    input_shape.shape[length - OFFSET_2] =
        SetAlign(input_shape.shape[length - OFFSET_2], STRIDE_16 * compileInfo.align_size);
  }
}

bool TransdataGeneral::AvoidBCOfVOR(int64_t a, int64_t b, int64_t c0, size_t idxA, size_t idxB) {
  // Avoid Bank Conflict of VOR: [a,b,c0]->[b,a,c0]
  // Step0: replace input-shape by a\b while shunning bank conflict.
  bound_input.shape[idxB] = b >= a && b % STRIDE_2 == 0 ? b + 1 : b;
  bound_input.shape[idxA] = a > b && a % STRIDE_2 == 0 ? a + 1 : a;
  // Step1: replace input-shape by factor.
  if (!ptrISplitT) {
    bound_input.shape[ptrI] = factorI;
  }
  if (!ptrOSplitT) {
    bound_input.shape[compileInfo.permute[ptrO]] = factorO;
  }
  // StepHandle: if fp32 and c0 is 16, need adjust.
  if (ele_byte == FP32_BYTE && c0 == STRIDE_16) {
    bound_input.shape[bound_input.size - 1] = c0 / STRIDE_2;
    bound_input.shape[idxB] = bound_input.shape[idxB] * STRIDE_2;
  }
  // Step2: create out-shape by input-shape
  for (size_t i = 0; i < bound_input.size; i++) {
    bound_output.shape[i] = bound_input.shape[compileInfo.permute[i]];
  }
  // Step3: calc ub-size
  int64_t num = Prod(bound_input.shape, ptrI + 1, bound_input.size);
  for (size_t i = ptrO + 1; i < bound_output.size; i++) {
    num = compileInfo.permute[i] < ptrI ? num * bound_output.shape[i] : num;
  }
  num *= split_once ? bound_input.shape[ptrI] : bound_input.shape[ptrI] * bound_input.shape[compileInfo.permute[ptrO]];
  // Step4: Check
  if (num <= UBSize) {
    return true;
  }
  // Step5: Try adapt
  if (split_once) {
    factorI--;
    factorO = factorI;
  } else {
    if (factorI >= factorO) {
      factorI--;
    } else {
      factorO--;
    }
  }

  if (factorI <= 0 || factorO <= 0) {
    return false;
  }
  a = ptrI == idxA ? factorI : compileInfo.permute[ptrO] == idxA ? factorO : a;
  b = ptrI == idxB ? factorI : compileInfo.permute[ptrO] == idxB ? factorO : b;
  return AvoidBCOfVOR(a, b, c0, idxA, idxB);
}

bool TransdataGeneral::AvoidBCOfVNCHWCONV(int64_t a, int64_t b, int64_t c0, size_t idxA, size_t idxB) {
  // Avoid Bank Conflict of VNC: [a,b] -> [b,a], c0 is align_value.
  // Step0: replace input-shape by a\b while shunning bank conflict.
  int64_t b_align = CeilDiv(b, c0);
  int64_t a_align = CeilDiv(a, c0);
  bound_input.shape[idxB] = b >= a && b_align % STRIDE_2 == 0 ? (b_align + 1) * c0 : b_align * c0;
  bound_input.shape[idxA] = a > b && a_align % STRIDE_2 == 0 ? (a_align + 1) * c0 : a_align * c0;
  // Step1: replace input-shape by factor.
  if (!ptrISplitT) {
    bound_input.shape[ptrI] = factorI;
  }
  if (!ptrOSplitT) {
    bound_input.shape[compileInfo.permute[ptrO]] = factorO;
  }
  // Step2: create out-shape by input-shape
  for (size_t i = 0; i < bound_input.size; i++) {
    bound_output.shape[i] = bound_input.shape[compileInfo.permute[i]];
  }
  // Step3: calc ub-size
  int64_t num = Prod(bound_input.shape, ptrI + 1, bound_input.size);
  for (size_t i = ptrO + 1; i < bound_output.size; i++) {
    num = compileInfo.permute[i] < ptrI ? num * bound_output.shape[i] : num;
  }
  num *= split_once ? bound_input.shape[ptrI] : bound_input.shape[ptrI] * bound_input.shape[compileInfo.permute[ptrO]];
  // Step4: Check
  if (num <= UBSize) {
    return true;
  }
  // Step5: Try adapt
  if (split_once) {
    factorI--;
    factorO = factorI;
  } else {
    if (factorI >= factorO) {
      factorI--;
    } else {
      factorO--;
    }
  }

  if (factorI <= 0 || factorO <= 0) {
    return false;
  }
  a = ptrI == idxA ? factorI : compileInfo.permute[ptrO] == idxA ? factorO : a;
  b = ptrI == idxB ? factorI : compileInfo.permute[ptrO] == idxB ? factorO : b;
  return AvoidBCOfVNCHWCONV(a, b, c0, idxA, idxB);
}

bool TransdataGeneral::AvoidVOR() {
  // init transpose axes index
  size_t c1_idx = compileInfo.is_forward ? c1_index : compileInfo.permute[c1_index];
  size_t h_idx = compileInfo.is_forward ? c1_idx - 1 : c1_idx + 1;
  // init transpose axes value
  int64_t c1 = ptrI == c1_idx ? factorI : compileInfo.permute[ptrO] == c1_idx ? factorO : tiling_input.shape[c1_idx];
  int64_t h = ptrI == h_idx ? factorI : compileInfo.permute[ptrO] == h_idx ? factorO : tiling_input.shape[h_idx];
  int64_t c0 = bound_input.shape[bound_input.size - 1];
  return c1_idx > h_idx ? AvoidBCOfVOR(h, c1, c0, h_idx, c1_idx) : AvoidBCOfVOR(c1, h, c0, c1_idx, h_idx);
}

bool TransdataGeneral::AvoidVNCHWCONV() {
  // init transpose axes index
  size_t c0_idx = compileInfo.is_forward ? c0_index : compileInfo.permute[c0_index];
  size_t h_idx = compileInfo.is_forward ? c0_idx + 1 : c0_idx - 1;
  // init transpose axes value
  int64_t c0 = ptrI == c0_idx ? factorI : compileInfo.permute[ptrO] == c0_idx ? factorO : tiling_input.shape[c0_idx];
  int64_t h = ptrI == h_idx ? factorI : compileInfo.permute[ptrO] == h_idx ? factorO : tiling_input.shape[h_idx];
  return c0_idx > h_idx ? AvoidBCOfVNCHWCONV(h, c0, c0, h_idx, c0_idx) : AvoidBCOfVNCHWCONV(c0, h, c0, c0_idx, h_idx);
}

bool TransdataGeneral::AvoidBankConflict() {
  if (!avoid_bc) {
    return true;
  }
  bound_input.SetSize(tiling_input.size);
  bound_output.SetSize(tiling_input.size);
  std::copy(tiling_input.shape, tiling_input.shape + tiling_input.size, bound_input.shape);
  return is_last_transpose ? AvoidVNCHWCONV() : AvoidVOR();
}

void TransdataGeneral::UBPreProcess(int64_t loop, size_t a, size_t b) {
  // PreProcess for calculating ub-tiling.
  // loop: index of possible chooses.
  // a: index of transpose axes based on input.
  // b: index of transpose axes based on input.
  ptrI = split_array[loop - 1].ptrA;
  ptrO = split_array[loop - 1].ptrB;
  ptrISplitT = ptrI == a || ptrI == b;
  ptrOSplitT = compileInfo.permute[ptrO] == a || compileInfo.permute[ptrO] == b;
  split_once = compileInfo.permute[ptrO] == ptrI;
  mte2.virLen = Prod(tiling_input.shape, ptrI + 1, tiling_input.size);
  mte3.virLen = Prod(tiling_output.shape, ptrO + 1, tiling_output.size);

  tiling_in_ub = mte2.virLen;
  for (size_t i = ptrO + 1; i < tiling_output.size; i++) {
    tiling_in_ub = compileInfo.permute[i] < ptrI ? tiling_in_ub * tiling_output.shape[i] : tiling_in_ub;
  }
}

void TransdataGeneral::UBInfoSet(size_t core_mode) {
  tilingInfo.core_mode = core_mode;
  tilingInfo.ub_0_idx = ptrI;
  tilingInfo.ub_1_idx = ptrO;
  tilingInfo.ub_0_factor = factorI;
  tilingInfo.ub_1_factor = factorO;
  tilingInfo.mte2_burst_len = mte2.virLen;
  tilingInfo.mte3_burst_len = mte3.virLen;
  tilingInfo.blk_dim = core;
  tilingInfo.percent = percent;
  tilingInfo.split_once = split_once;
}

void TransdataGeneral::SetStorageAlign(const Shape& input_shape, size_t length) {
  tiling_input.SetSize(length);
  tiling_output.SetSize(length);
  std::copy(input_shape.shape, input_shape.shape + length, tiling_input.shape);
  // align for mte2(pad had been done)
  tiling_input.shape[length - 1] = SetAlign(tiling_input.shape[length - 1], compileInfo.align_size);
  // align for vnchwconv(support 5HD, NZ)
  if (is_last_transpose) {
    AlignVNC(tiling_input);
  }
  // input + perm -> output
  for (size_t i = 0; i < length; i++) {
    tiling_output.shape[i] = tiling_input.shape[compileInfo.permute[i]];
  }
}

void TransdataGeneral::AdjustUBFactorMTE2(int64_t lower, int64_t higher) {
  // Update factorI
  // [lower, higher] is boundary of factorI
  // Attention: the func only work in twice split
  int64_t ub_i_outer = CeilDiv(tiling_input.shape[ptrI], factorI);
  int64_t ub_o_outer = CeilDiv(tiling_output.shape[ptrO], factorO);
  int64_t base_core = total_num / tiling_in_ub / tiling_output.shape[ptrO] / tiling_input.shape[ptrI];
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
  int64_t base_core = total_num / tiling_in_ub / tiling_output.shape[ptrO] / tiling_input.shape[ptrI];
  core = base_core * ub_i_outer * ub_o_outer;
  if (core > compileInfo.core_num) {
    // extra core can improve UB usage
    for (int64_t i = lower; i <= higher; i++) {
      // check tail
      GetOutputRealTail(ptrO, i, mte3);
      bool tail_is_legal = mte3.tailLen == 0 || mte3.tailLen >= compileInfo.align_size;
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

void TransdataGeneral::CompareTiling(bool is_multi_core_mode) {
  // is_multi_core_mode: True that could support multi-core, otherwise only support single-core
  // multi vs single
  size_t mode = is_multi_core_mode ? MULTI_CORE_MODE : SINGLE_CORE_MODE;
  if (tilingInfo.core_mode == MULTI_CORE_MODE && mode == SINGLE_CORE_MODE) {
    return;
  }
  // single vs (multi, single)
  if (tilingInfo.core_mode == SINGLE_CORE_MODE) {
    if (mode == MULTI_CORE_MODE || tilingInfo.percent < percent) {
      UBInfoSet(mode);
    }
    return;
  }

  // multi vs multi
  if (tilingInfo.percent == 0) {
    UBInfoSet(mode);
  } else {
    int64_t new_core_distance = abs(core - compileInfo.core_num);
    int64_t old_core_distance = abs(tilingInfo.blk_dim - compileInfo.core_num);
    if (new_core_distance < old_core_distance) {
      UBInfoSet(mode);
    } else if (new_core_distance == old_core_distance) {
      if (split_once && !tilingInfo.split_once) {
        UBInfoSet(mode);
      }
    }
  }
}

void TransdataGeneral::AvoidBankConflictIsWork() {
  // Sometimes sch(python) doesn't need avoid bank-conflict while avoidBCWork is 1.
  if (transposeWork == 0 || !avoid_bc) {
    return;
  }
  if (compileInfo.is_forward && shapeType == COMMON_ALIGN) {
    return;
  }

  if (is_last_transpose) {
    // NC1C0H <-> NC1HC0
    size_t c0_idx = compileInfo.is_forward ? c0_index : compileInfo.permute[c0_index];
    size_t h_idx = compileInfo.is_forward ? c0_idx + 1 : c0_idx - 1;
    int64_t c0 = AxisValueInUB(c0_idx);
    int64_t h = AxisValueInUB(h_idx);
    if ((c0 >= h && CeilDiv(c0, c0) % STRIDE_2 == 0) || (h >= c0 && CeilDiv(h, c0) % STRIDE_2 == 0)) {
      avoidBCWork = 1;
    }
  } else {
    // NHC1C0 <-> NC1HC0
    size_t c1_idx = compileInfo.is_forward ? c1_index : compileInfo.permute[c1_index];
    size_t h_idx = compileInfo.is_forward ? c1_idx - 1 : c1_idx + 1;
    int64_t c1 = AxisValueInUB(c1_idx);
    int64_t h = AxisValueInUB(h_idx);
    if ((c1 >= h && c1 % STRIDE_2 == 0) || (h >= c1 && h % STRIDE_2 == 0)) {
      avoidBCWork = 1;
    }
  }
}

void TransdataGeneral::AdjustUBFactor() {
  if (tilingInfo.core_mode == SINGLE_CORE_MODE) {
    return;
  }
  size_t ptr = static_cast<size_t>(VectorIndex(compileInfo.permute, tilingInfo.ub_0_idx));
  tilingInfo.ub_0_factor = CommonRefineFactor(tilingInfo.ub_0_factor, ptr);
  tilingInfo.ub_1_factor = CommonRefineFactor(tilingInfo.ub_1_factor, tilingInfo.ub_1_idx);
  AvoidBankConflictIsWork();
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
    factor = is_data_move ? factor : out_ptr == c1_index ? factor * reshape.shape[c0_index] : factor;
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
  if (input.size < 1 || output.size < 1 || reshape.size < 1) {
    return false;
  }

  // Check is pure data move branch or not
  // if forward, index of c0 based on reshape(input -> reshape ~transpose~> output)
  // if backward, index of c0 based on reshape(output <- reshape <~transpose~ input)
  size_t root_ptr = 0;
  for (size_t i = 0; i < compileInfo.src_pad_mode.size(); i++) {
    if (compileInfo.src_pad_mode[i] == EXISTED_C1C0) {
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
  if (compileInfo.is_const && (!compileInfo.is_const_compile)) {
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
  shapeType = ChooseType(last_dim, compileInfo.ub_info[computeType][shapeType]) ? COMMON_ALIGN : STORAGE_ALIGN;
  UBSize = compileInfo.ub_info[computeType][shapeType];

  // avoid bank-conflict of VOR in n-last-transpose (Support 5HD|NZ)
  // avoid bank-conflict of vnchwconv in last-transpose (Support 5HD|NZ)
  if (transposeWork == 0 || (compileInfo.is_forward && shapeType == COMMON_ALIGN)) {
    return true;
  }
  if ((is_last_transpose && ele_byte == FP32_BYTE) || (!is_last_transpose && ele_byte == INT8_BYTE)) {
    return true;
  }
  avoid_bc = true;
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
  // Avoid Conflict to assure axes that do transpose.
  // N_Last_Transpose: VOR, (h,c1) <-> (c1,h).
  // Last_Transpose: VNC, (c0,h) <-> (h,c0).
  size_t a = compileInfo.is_forward ? c1_index : compileInfo.permute[c1_index];
  size_t b = compileInfo.is_forward ? a - 1 : a + 1;
  if (is_last_transpose) {
    a = compileInfo.is_forward ? c0_index : compileInfo.permute[c0_index];
    b = compileInfo.is_forward ? a + 1 : a - 1;
  }

  // Main Process from different chooses.
  // Result contains SINGLE_CORE_MODE and MULTI_CORE_MODE.
  // SINGLE_CORE_MODE: ub-tiling only support single core.
  // MULTI_CORE_MODE: ub-tiling could support multi core.
  int64_t loop = array_size;
  total_num = Prod(tiling_input.shape, 0, tiling_input.size);
  while (loop >= 1) {
    UBPreProcess(loop, a, b);
    CompareTiling(split_once ? OnceTiling() : TwiceTiling());
    loop--;
  }
  AdjustUBFactor();
  return true;
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
    SetStorageAlign(reshape, output.size);
  } else {
    // input -> reshape -> ac_out
    SetStorageAlign(input, input.size);
  }
  V_OP_TILING_CHECK(Filter(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Filter Failed"), return false);
  return UBTilingProcess();
}

bool TransdataGeneral::BlockTiling() {
  const Shape &res = compileInfo.is_forward ? output : reshape;
  AxisType type_array[res.size] = {UB_EXTERNAL};
  DiscriminationAxisType(type_array, res.size);

  bool is_single = tilingInfo.core_mode == SINGLE_CORE_MODE;
  return is_single ? SingleBlkTiling(type_array, res) : MultiBlkTiling(type_array, res);
}

bool TransdataGeneral::SingleBlkTiling(const AxisType* type_array, const Shape& res) {
  // update blk_dim, blk_factor, blk_idx.
  tilingInfo.blk_dim = 1;
  for (size_t i = 0; i < res.size; i++) {
    if (type_array[i] == UB_EXTERNAL) {
      tilingInfo.blk_idx = i;
      tilingInfo.blk_factor = res.shape[i];
      return true;
    } else if (type_array[i] == UB_FIRST_SPLIT) {
      tilingInfo.blk_idx = i;
      tilingInfo.blk_factor = CeilDiv(res.shape[i], tilingInfo.ub_0_factor);
      return true;
    } else if (type_array[i] == UB_SECOND_SPLIT) {
      tilingInfo.blk_idx = i;
      tilingInfo.blk_factor = CeilDiv(res.shape[i], tilingInfo.ub_1_factor);
      return true;
    }
  }
  return false;
}

bool TransdataGeneral::MultiBlkTiling(const AxisType* type_array, const Shape& res) {
  // Forward: res is output, Backward: res is reshape
  core = 1;
  factorI = tilingInfo.ub_0_factor;
  factorO = tilingInfo.ub_1_factor;
  int64_t dim_bound = 1;
  size_t block_idx = 0;
  size_t i = 0;
  bool exceed_limit = false;

  // Find split idx
  while (i < res.size) {
    if (core >= compileInfo.core_num) {
      exceed_limit = true;
      break;
    }

    if (type_array[i] == UB_EXTERNAL) {
      dim_bound = res.shape[i];
    } else if (type_array[i] == UB_FIRST_SPLIT) {
      dim_bound = CeilDiv(res.shape[i], factorI);
    } else if (type_array[i] == UB_SECOND_SPLIT) {
      dim_bound = CeilDiv(res.shape[i], factorO);
    } else {
      i++;
      continue;
    }
    i++;
    core *= dim_bound;
    block_idx = i;
  }

  // Assure blk_idx and factor
  tilingInfo.blk_dim = core;
  tilingInfo.blk_factor = 1;
  tilingInfo.blk_idx = block_idx == 0 ? block_idx : block_idx - 1;
  if (!exceed_limit) {
    return true;
  }

  core = core / dim_bound;
  for (int64_t j = 1; j <= dim_bound; j++) {
    int64_t outer = core * CeilDiv(dim_bound, j);
    if (outer <= compileInfo.core_num) {
      tilingInfo.blk_factor = j;
      tilingInfo.blk_dim = outer;
      break;
    }
  }
  return true;
}

bool TransdataGeneral::WriteTilingData() {
  if (compileInfo.is_const && !compileInfo.is_const_compile) {
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
    run_info.AddTilingData(static_cast<int32_t>(transposeWork));
    run_info.AddTilingData(static_cast<int32_t>(avoidBCWork));
    run_info.AddTilingData(static_cast<uint32_t>(tilingInfo.blk_dim));
    return true;
  }

  // dynamic
  run_info.SetBlockDim(static_cast<uint32_t>(tilingInfo.blk_dim));
  run_info.SetTilingKey(static_cast<uint32_t>(CalcTilingKey()));
  // convert dim which is input after fused
  const Shape* res_shape = compileInfo.is_forward ? &input : &output;
  for (size_t i = 0; i < res_shape->size; i++) {
    run_info.AddTilingData(static_cast<int32_t>(res_shape->shape[i]));
    OP_LOGD(op_type.c_str(), "input shape : %d", res_shape->shape[i]);
  }
  // convert factor
  run_info.AddTilingData(static_cast<int32_t>(tilingInfo.ub_0_factor));
  if (!tilingInfo.split_once) {
    run_info.AddTilingData(static_cast<int32_t>(tilingInfo.ub_1_factor));
  }
  run_info.AddTilingData(static_cast<int32_t>(tilingInfo.blk_factor));

  return true;
}

bool TransdataGeneral::CalcTiling() {
  V_OP_TILING_CHECK(Init(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "UpdateValue Failed"), return false);
  if (!IsConstRuntime()) {
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
