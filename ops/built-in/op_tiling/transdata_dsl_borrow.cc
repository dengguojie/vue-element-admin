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
 * \file transdata_dsl_borrow.cpp
 * \brief
 */
#include "transdata_dsl_borrow.h"

namespace optiling {
using namespace transdata_dsl;

int64_t TransdataBorrow::CalcTilingKey() {
  using namespace std;
  int64_t db = 0;
  int64_t is_forward = compileInfo.is_forward ? FORWARD_KEY : BACKWARD_KEY;
  size_t key_num = 9;
  int64_t pos[key_num] = {db,
                          is_forward,
                          static_cast<int64_t>(computeType),
                          static_cast<int64_t>(shapeType),
                          static_cast<int64_t>(tilingInfo.blk_idx),
                          VectorIndex(*permute, tilingInfo.ub_0_idx),
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

bool TransdataBorrow::InferTilingInput() {
  if (compileInfo.align_size < BLOCK / FP16_BYTE) {
    // Reinterpret tensor(32bit, 64bit) as tensor(16bit).
    // permute had been updated in compilation.
    is_reinterpret = true;
    int64_t pad_var = BLOCK / FP16_BYTE / compileInfo.align_size;
    tiling_output.shape[tiling_output.size] = pad_var;
    tiling_output.SetSize(tiling_output.size + 1);
    input.Insert(input.size, pad_var);
    output.Insert(output.size, pad_var);
  }
  // Create tiling_input just like (N.o,C1,H,C0,16) or (N,C1,H0,C0,H1) while forward
  // Create tiling_input just like (N.o,H,C,16) or (N,h0,C,h1) while backward
  tiling_input.SetSize(tiling_output.size);
  for (size_t i = 0; i < permute->size(); i++) {
    tiling_input.shape[permute->at(i)] = tiling_output.shape[i];
  }
  // Other params
  ele_byte = BLOCK / align_size;
  mte_rate = PACKET_SENDING_RATE / ele_byte;
  coef = computeType == BORROW_H_SCH ? align_size : 1;
  return true;
}

int64_t TransdataBorrow::CalcBNBound() {
  /* BN: [n.o, C1, H, C0, 16] <-> [n.o, 16, C1, H, C0]
   * if ub split H = outer * var, formula is:
   *    SetAlign(C1*var*C0, align) * 16 <= UB
   * if ub split n.o = outer * var, formula is:
   *    SetAlign(C1*H*C0, align) * var * 16 <= UB
   * num_in_ub consider 16.
   * */
  if (ptrO == 0) {
    return UBSize / align_size / SetAlign(num_in_ub / align_size, STRIDE_3 * align_size);
  } else {
    return (UBSize / align_size - STRIDE_3 * align_size + 1) / (num_in_ub / align_size);
  }
}

int64_t TransdataBorrow::CalcBHBound() {
  /* In BH(Forward), vnc0 is [ho,hi,C] -> [hi,C,ho], vnc1 is [C1,hi,C0,ho] -> [C1,ho,hi,C0].
   * In BH(Backward), vnc0 is [C1,ho,hi,C0] -> [C1,hi,C0,ho], vnc1 is [hi,C,ho] -> [ho,hi,C].
   * For calc-bound, assume Cx = C1*C0.
   * */
  return UBSize / num_in_ub;
}

bool TransdataBorrow::OnceTiling() {
  // ptrI and ptrO split same axis: [N,H,C,16] -> [N,16,H,C] [N,C1,H0,C0,H1] -> [N,C1,H1,H0,C0]
  // handle_c1: make c1\c0 in ub-internal while c1 belong to ub-external and don't split c1\c0.
  // calc-bound need do two kind ?
  bool init_success = false;
  bool handle_c1 = compileInfo.is_forward && permute->at(c1_index) < ptrI && c1_index < ptrO;
  num_in_ub = Prod(tiling_input.shape, ptrI + 1, tiling_input.size);
  for (size_t idx = ptrO + 1; idx < tiling_output.size; idx++) {
    num_in_ub = permute->at(idx) < ptrI ? num_in_ub * tiling_output.shape[idx] : num_in_ub;
  }
  num_in_ub = handle_c1 ? num_in_ub * tiling_input.shape[permute->at(c1_index)] : num_in_ub;
  if (num_in_ub == 0) {
    return false;
  }

  // In BH, search-space should be in [1, maximum_space_bound]
  // In BN, search-space should be in [1, maximum_theory_bound]
  // maximum_space_bound >= maximum_theory_bound
  int64_t maximum_space_bound = computeType == BORROW_N_SCH ? CalcBNBound() : CalcBHBound();
  int64_t total_num = Prod(tiling_input.shape, 0, tiling_input.size);
  int64_t factor_limit = output.shape[ptrO > x1_index ? ptrO - 1 : ptrO];
  int64_t begin = 1;
  int64_t stride = 1;
  bool is_bh_backward = computeType == BORROW_H_SCH && (!compileInfo.is_forward);
  if (is_bh_backward) {
    factor_limit = SetAlign(factor_limit, align_size);
    begin = align_size;
    stride = align_size;
  }

  for (int64_t factor = begin; coef * factor <= maximum_space_bound; factor += stride) {
    // In BH + backward, factor should be division by 16 (FP16).
    // Only in BH + backward, factor can more than real_value.
    if (factor > factor_limit) {
      break;
    }
    // Check tail
    GetOutputRealTail(ptrO, coef * factor, mte3);
    bool tail_is_legal = mte3.tailLen == 0 || mte3.tailLen >= align_size;
    // Check main
    core = total_num / num_in_ub / tiling_input.shape[ptrI] * CeilDiv(tiling_input.shape[ptrI], coef * factor);
    bool main_is_legal = mte3.mainLen >= align_size || core == 1;
    if (tail_is_legal && main_is_legal) {
      factorI = coef * factor;
      init_success = true;
      if (mte3.mainLen >= mte_rate) {
        break;
      }
    }
  }

  // Check init
  if (!init_success) {
    return false;
  }

  // Return
  if (core <= compileInfo.core_num) {
    factorO = factorI;
    return true;
  }

  // Adjust: extra core can improve UB usage
  for (int64_t factor = factorI / coef; coef * factor <= maximum_space_bound; factor += stride) {
    if (factor > factor_limit) {
      break;
    }
    // Check tail
    GetOutputRealTail(ptrO, coef * factor, mte3);
    bool tail_is_legal = mte3.tailLen == 0 || mte3.tailLen >= align_size;
    if (tail_is_legal) {
      core = total_num / num_in_ub / tiling_input.shape[ptrI] * CeilDiv(tiling_input.shape[ptrI], coef * factor);
      factorI = coef * factor;
      if (core < compileInfo.core_num) {
        break;
      }
    }
  }
  factorO = factorI;
  return true;
}

void TransdataBorrow::CompareTiling() {
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

void TransdataBorrow::AdjustUBFactorForward() {
  // Forward's out is align, make dim can be divisible by factor.
  int64_t dimBound = tiling_output.shape[tilingInfo.ub_1_idx] / coef;
  int64_t factor = tilingInfo.ub_1_factor / coef;
  int64_t n_factor = REFINE(dimBound, factor);
  if (computeType == BORROW_H_SCH && n_factor % STRIDE_2 == 0) {
    int64_t loop_a = CeilDiv(dimBound, n_factor - 1);
    int64_t loop_b = CeilDiv(dimBound, n_factor + 1);
    // factor is maximum-factor
    n_factor = loop_a <= loop_b || n_factor == factor ? n_factor - 1 : n_factor + 1;
  }
  tilingInfo.ub_1_factor = n_factor * coef;
  tilingInfo.ub_0_factor = tilingInfo.ub_1_factor;
}

void TransdataBorrow::AdjustUBFactorBackward() {
  // Backward: try to make MTE3 align
  int64_t factor = tilingInfo.ub_1_factor;
  for (; factor >= 1; factor--) {
    GetOutputRealTail(tilingInfo.ub_1_idx, factor, mte3);
    bool main_align = mte3.mainLen % align_size == 0;
    bool tail_align = mte3.tailLen % align_size == 0;
    // best
    if (main_align && tail_align) {
      tilingInfo.ub_1_factor = factor;
      break;
    }
    // better (choose bigger main)
    if (main_align && mte3.tailLen >= align_size) {
      GetOutputRealTail(tilingInfo.ub_1_idx, tilingInfo.ub_1_factor, mte3);
      bool had_aligned = mte3.mainLen % align_size == 0;
      tilingInfo.ub_1_factor = had_aligned ? tilingInfo.ub_1_factor : factor;
    }
  }
  tilingInfo.ub_0_factor = tilingInfo.ub_1_factor;
}

void TransdataBorrow::GetOutputRealTail(size_t ptr, int64_t factor, MTEInfo &mte) {
  /* FP16 as Eg.
   * Backward + BorrowN:
   * 1. output is [N,H,C], tiling_output is [No,Ni,H,C].
   * 2. [H,C] is serial that be burst_len.
   * Backward + BorrowH:
   * 1. output is [N,H,C], tiling_output is [No,ho,hi,C], hi = m*16.
   * 2. factor is 16*hi, if factor more than H, factor=H.
   * 3. factor*C is serial that be burst_len.
   * Forward + BorrowN:
   * 1. output is [N, C1, H, C0], tiling_output is [No, Ni, C1, H, C0].
   * 2. [H, C0] is serial that be burst_len, C1 would be n_burst.
   * Forward + BorrowH:
   * 1. output is [N, C1, H, C0], tiling_output is [N, C1, ho, hi, C0].
   * 2. factor split ho, factor*hi*c0 is serial that be burst_len, C1 would be n_burst.
   * */
  // mapping split-axis based on res-tensor
  size_t out_ptr = ptr > x1_index ? ptr - 1 : ptr;
  // split ho by factor equals to split H by factor * hi
  factor = ptr == x1_index ? factor * tiling_output.shape[x0_index] : factor;
  // split actually occur on the res that make factor don't exceed dim-bound.
  factor = output.shape[out_ptr] > factor ? factor : output.shape[out_ptr];
  int64_t baseLen = Prod(output.shape, out_ptr + 1, output.size);
  // static check
  factor = factor == 0 ? 1 : factor;
  // BH: ho hi fused as Hx in [N,C1,ho,hi,C0] or [N,ho,hi,C], and hi C0 is serial.
  // BN: No Ni fused as Nx in [No,Ni,C1,H,C0] or [No,Ni,H,C], but Ni H is not serial.
  bool cond = computeType == BORROW_N_SCH ? ptr <= x1_index : ptr < x1_index;
  mte.mainLen = cond ? baseLen : factor * baseLen;
  mte.tailLen = cond ? baseLen : (output.shape[out_ptr] % factor) * baseLen;
}

void TransdataBorrow::DiscriminationAxisType(AxisType *type_array, size_t length) {
  // Assure type of axis is belong to internal of UB or not.
  // type_array based on output
  for (size_t i = tilingInfo.ub_0_idx + 1; i < length; i++) {
    // deal input
    type_array[VectorIndex(*permute, i)] = UB_INTERNAL;
  }
  for (size_t i = tilingInfo.ub_1_idx + 1; i < length; i++) {
    // deal output
    type_array[i] = UB_INTERNAL;
  }
  // make c1\c0 in ub-internal while forward
  if (compileInfo.is_forward && type_array[c1_index] == UB_EXTERNAL) {
    type_array[c1_index] = UB_INTERNAL;
  }
  // part of split-axis belong to ub_external, part belong to ub_internal.
  type_array[VectorIndex(*permute, tilingInfo.ub_0_idx)] = UB_FIRST_SPLIT;
  type_array[tilingInfo.ub_1_idx] = UB_SECOND_SPLIT;
}

bool TransdataBorrow::ChooseHelpInfo() {
  // BorrowX discriminate X as H or N which used different info.
  const std::vector <size_t> *info_c;
  if (computeType == BORROW_N_SCH) {
    x1_index = compileInfo.bn_x1x0[0];
    x0_index = compileInfo.bn_x1x0[1];
    permute = &compileInfo.bn_permute;
    info_c = &compileInfo.bn_c1c0;
  } else {
    x1_index = compileInfo.bh_x1x0[0];
    x0_index = compileInfo.bh_x1x0[1];
    permute = &compileInfo.bh_permute;
    info_c = &compileInfo.bh_c1c0;
  }

  if (compileInfo.is_forward) {
    c1_index = info_c->at(0);
    c0_index = info_c->at(1);
  } else {
    c_index = info_c->at(0);
  }
  return true;
}

bool TransdataBorrow::Init() {
  /* Support Forward (eg:NHC -> NHC1C0 -> NC1HC0)
   * (N.o,16,H,C) -> (N.o,H,C,16) -> (N.o,H,C1,C0,16) -> (N.o,C1,H,C0,16) -> (N.o,16,C1,H,C0) BN
   * (N,h1,h0,C) -> (N,h0,C,h1) -> (N,h0,C1,C0,h1) -> (N,C1,h0,C0,h1) -> (N,C1,h1,h0,c0) BH
   * Support Backward (eg:NC1HC0 -> NHC1C0 -> NHC)
   * (N.o,16,C1,H,C0) -> (N.o,C1,H,C0,16) ->(N.o,H,C1,C0,16) -> (N.o,H,C,16) -> (N.o,16,H,C) BN
   * (N,C1,h1,h0,C0) -> (N,C1,h0,C0,h1) -> (N,h0,C1,C0,h1) -> (N,h0,C,h1) -> (N,h1,h0,C) BH
   * The algorithm would choose last-transpose as tiling(just like (N,h0,C,h1) -> (N,h1,h0,C)),
   * but tensors of last-transpose maybe not maximum size in stream that need to do pad in tiling_tensors.
   * */
  // Create tiling_out [N.o,16,C1,H,C0] or [N,C1,H1,H0,C0]
  align_size = compileInfo.align_size < BLOCK / FP16_BYTE ? BLOCK / FP16_BYTE : compileInfo.align_size;
  size_t ptr = 0;
  for (size_t i = 0; i < output.size; i++) {
    if (i == x1_index) {
      if (computeType == BORROW_N_SCH) {
        tiling_output.shape[ptr + 1] = align_size;
        tiling_output.shape[ptr] = CeilDiv(SetAlign(output.shape[i], align_size), align_size);
      } else {
        tiling_output.shape[ptr + 1] = 1;
        int64_t var = compileInfo.is_forward ? align_size : align_size * align_size;
        tiling_output.shape[ptr] = SetAlign(output.shape[i], var);
      }
      ptr++;
    } else {
      tiling_output.shape[ptr] = output.shape[i];
      if (!compileInfo.is_forward && compileInfo.src_pad_mode[i] != 0) {
        tiling_output.shape[ptr] = SetAlign(output.shape[i], compileInfo.src_pad_var[i]);
      }
    }
    ptr++;
  }
  tiling_output.SetSize(ptr);
  return InferTilingInput();
}

bool TransdataBorrow::IsConstRuntime() {
  if (compileInfo.is_const && (!compileInfo.is_const_compile)) {
    std::string pattern_str = std::to_string(CONST_KEY);
    tilingInfo.blk_dim = compileInfo.const_block_dims.at(pattern_str);
    return true;
  }
  return false;
}

bool TransdataBorrow::Strategy() {
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

bool TransdataBorrow::BNFilter(size_t ptrA, size_t ptrB) {
  // not exceed UBSize [n,c1,h,c0,16] -> [n,16,c1,h,c0] (forward + BN).
  // not exceed UBSize [n,h,c,16] -> [n,16,h,c] (backward + BN).
  // forward would make c1\c0 in ub-internal.
  int64_t num = Prod(tiling_input.shape, ptrA + 1, tiling_input.size);
  for (size_t idx = ptrB + 1; idx < tiling_output.size; idx++) {
    num = permute->at(idx) < ptrA ? num * tiling_output.shape[idx] : num;
  }
  // make c1\c0 in ub-internal
  num = compileInfo.is_forward && permute->at(c1_index) < ptrA && c1_index < ptrB ?
        num * tiling_output.shape[c1_index] : num;
  // make storage-align
  num = SetAlign(num, align_size * align_size * STRIDE_3);
  return num <= UBSize;
}

bool TransdataBorrow::BHFilter(size_t ptrA, size_t ptrB) {
  // not exceed UBSize [n,c1,h0,c0,h1]->[nn,c1,h1,h0,c0] (forward+BH).
  // not exceed UBSize [n,h0,c,h1]->[n,h1,h0,c] (backward+BH).
  // forward would make c1\c0 in ub-internal.
  int64_t num = Prod(tiling_output.shape, x1_index + 1, tiling_output.size);
  num = ptrB <= x1_index ? num : num / Prod(tiling_output.shape, x1_index + 1, ptrB + 1);
  num = SetAlign(num, align_size * STRIDE_3);
  num *= Prod(tiling_output.shape, ptrB + 1, x1_index + 1);
  for (size_t idx = ptrA + 1; idx < tiling_input.size; idx++) {
    num = static_cast<size_t>(VectorIndex(*permute, idx)) < ptrB ? num * tiling_input.shape[idx] : num;
  }
  num = compileInfo.is_forward && permute->at(c1_index) < ptrA && c1_index < ptrB ?
        num * tiling_output.shape[c1_index] : num;
  return num <= UBSize;
}

bool TransdataBorrow::Filter() {
  /* Filter-BN:
   * 1. don't split x0,c1,c0(forward), don't split x0,c(backward).
   * 2. avoid bank-conflict(BC) in twice vnchwconv, do-straoge-align by 3*16(FP16).
   * Filter-BH:
   * 1. don't split x0,c1,c0(forward), don't split x0,c(backward).
   * 2. sch only avoid BC in first vnchwconv.
   * 3. BH need ub split in x1.
   * */
  size_t length = tiling_input.size;
  for (size_t i = 0; i < length; i++) {
    bool split_x0 = i == permute->at(x0_index);
    bool split_c = compileInfo.is_forward ? i == permute->at(c0_index) || i == permute->at(c1_index) :
                   i == permute->at(c_index);
    if (split_x0 || split_c) {
      // Common Limit no matter BN or BH.
      continue;
    }
    if (computeType == BORROW_H_SCH && i != permute->at(x1_index)) {
      // BH Limit
      continue;
    }
    // Limit of UBSize
    size_t ptrB = static_cast<size_t>(VectorIndex(*permute, i));
    if (computeType == BORROW_H_SCH ? BHFilter(i, ptrB) : BNFilter(i, ptrB)) {
      split_array[array_size].Set(i, ptrB);
      array_size++;
    }
  }
  return array_size > 0;
}

bool TransdataBorrow::UBTilingProcess() {
  bool tiling_success = false;
  while (array_size >= 1) {
    ptrI = split_array[array_size - 1].ptrA;
    ptrO = split_array[array_size - 1].ptrB;
    if (!OnceTiling()) {
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
    if (computeType == BORROW_N_SCH) {
      AdjustUBFactorBackward();
    }
  }
  return tiling_success;
}

bool TransdataBorrow::UBTiling() {
  /* In Borrow-N, while tiling_input is (N.o,H,C,16), tiling_output is (N.o,16,H,C).
   * In Borrow-N, while tiling_input is (N.o,C1,H,C,16), tiling_output is (N.o,16,C1,H,C).
   * In Borrow-H, while tiling_input is (N,h0,C,h1), tiling_output is (N,h1,h0,C).
   * In Borrow-H, while tiling_input is (N,C1,h0,C0,h1), tiling_output is (N,c1,h1,h0,C0).
   * Only support once-tiling in UB.
   * */
  V_OP_TILING_CHECK(Filter(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Filter Failed"), return false);
  V_OP_TILING_CHECK(UBTilingProcess(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "UB-Tiling Failed"), return false);
  return true;
}

bool TransdataBorrow::BlockTiling() {
  AxisType type_array[tiling_output.size] = {UB_EXTERNAL};
  DiscriminationAxisType(type_array, tiling_output.size);
  BlkTilingProcess(type_array, tiling_output);
  return true;
}

void TransdataBorrow::BlkTilingProcess(const AxisType *type_array, const Shape &res) {
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
  if (!exceed_limit) {
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

bool TransdataBorrow::WriteTilingData() {
  if (compileInfo.is_const && (!compileInfo.is_const_compile)) {
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
    run_info.AddTilingData(static_cast<int32_t>(VectorIndex(*permute, tilingInfo.ub_0_idx)));
    run_info.AddTilingData(static_cast<int32_t>(tilingInfo.ub_1_idx));
    run_info.AddTilingData(static_cast<int32_t>(tilingInfo.blk_factor));
    if (computeType == BORROW_N_SCH) {
      run_info.AddTilingData(static_cast<int32_t>(tilingInfo.ub_0_factor));
      run_info.AddTilingData(static_cast<int32_t>(tilingInfo.ub_1_factor));
    } else {
      run_info.AddTilingData(static_cast<int32_t>(tilingInfo.ub_0_factor / coef));
      run_info.AddTilingData(static_cast<int32_t>(tilingInfo.ub_1_factor / coef));
    }
    run_info.AddTilingData(static_cast<int32_t>(transposeWork));
    run_info.AddTilingData(static_cast<int32_t>(avoidBCWork));
    run_info.AddTilingData(static_cast<uint32_t>(tilingInfo.blk_dim));
    return true;
  }

  // dynamic
  run_info.SetBlockDim(static_cast<uint32_t>(tilingInfo.blk_dim));
  run_info.SetTilingKey(static_cast<uint32_t>(CalcTilingKey()));
  // convert dim which is input after fused
  const Shape *res_shape = compileInfo.is_forward ? &input : &output;
  for (size_t i = 0; i < res_shape->size; i++) {
    bool skip_dim = is_reinterpret && i == res_shape->size - 1;
    if (!skip_dim) {
      run_info.AddTilingData(static_cast<int32_t>(res_shape->shape[i]));
      OP_LOGD(op_type.c_str(), "input shape : %d", res_shape->shape[i]);
    }
  }
  // convert factor
  if (computeType == BORROW_N_SCH) {
    run_info.AddTilingData(static_cast<int32_t>(tilingInfo.ub_0_factor));
  } else {
    run_info.AddTilingData(static_cast<int32_t>(tilingInfo.ub_0_factor / coef));
  }
  run_info.AddTilingData(static_cast<int32_t>(tilingInfo.blk_factor));
  return true;
}

bool TransdataBorrow::CalcTiling() {
  V_OP_TILING_CHECK(ChooseHelpInfo(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ChooseHelpInfo Failed"),
                    return false);
  V_OP_TILING_CHECK(Init(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Init Failed"), return false);
  if (!IsConstRuntime()) {
    V_OP_TILING_CHECK(Strategy(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ChooseStrategy Failed"), return false);
    V_OP_TILING_CHECK(UBTiling(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "UBTiling Failed"), return false);
    V_OP_TILING_CHECK(BlockTiling(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "BlockTiling Failed"), return false);
  }
  return true;
}

void TransdataBorrow::SetAttr(size_t type, size_t work) {
  // handle attr
  computeType = type;
  transposeWork = work;
}

bool TransdataBorrow::DoTiling() {
  // main process
  bool ret = CalcTiling();
  return ret && WriteTilingData();
}
}  // namespace optiling
