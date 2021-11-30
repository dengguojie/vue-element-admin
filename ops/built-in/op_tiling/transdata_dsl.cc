/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file trans_data_dsl.cpp
 * \brief
 */
#include <math.h>
#include "transdata_dsl.h"
#include "error_log.h"
#include "vector_tiling_log.h"
#include "graph/utils/op_desc_utils.h"

namespace optiling {

int64_t TransdataBase::LimitMap(int64_t factor) {
  switch(BLOCK / factor) {
    case FP32:
      return 64;
    case FP16:
      return 256;
    case INT8:
      return 1024;
    default:
      return 1;
  }
}

bool TransdataBase::CommonAlignLimit(int64_t dim_len, int64_t ub_size) {
  int64_t factor = compileInfo.align_size;
  int64_t num_bit = BLOCK / factor;
  bool is_not_align = dim_len % factor != 0;
  bool is_legal_len = dim_len <= PACKET_SENDING_RATE / num_bit;
  is_legal_len = is_legal_len and (dim_len <= ub_size / num_bit / LimitMap(factor) / factor * factor);
  return is_legal_len and is_not_align;
}

int64_t TransdataBase::SetAlign(int64_t value, int64_t align_factor) {
  return (value + align_factor - 1) / align_factor * align_factor;
}

int64_t TransdataBase::Prod(int64_t *input, int64_t ptr, int64_t length) {
  int64_t base = 1;
  if (ptr >= length) {
    return base;
  }
  for (int64_t idx = ptr; idx < length; idx++) {
    base *= input[idx];
  }
  return base;
}

int32_t TransdataBase::CalcTilingKey() {
  using namespace std;
  int64_t db = 0;
  int64_t is_forward = compileInfo.is_forward == 1 ? 2 : 3;
  int64_t pos[7] = {db, is_forward, computeType, shapeType, tilingInfo.blk_idx,
                    compileInfo.permute[tilingInfo.ub_0_idx], tilingInfo.ub_1_idx};
  pos[5] = computeType == 1 ? tilingInfo.ub_0_idx : pos[5];
  int64_t val[7] = {1000000000, 100000000, 10000000, 100000, 10000, 1000, 100};
  int32_t key = 0;
  for (size_t i = 0; i < 7; i++) {
    key += pos[i] * val[i];
  }
  return key;
}

void TransdataBase::GetOutPutRealTail(int64_t ptr, int64_t factor, TransdataDSLMTEInfo *mte) {
  // work in backward
  // reshape
  int64_t realPtr = reshape_mapping_output[ptr];
  int64_t baseLen = Prod(output_shape, realPtr + 1, tiling_length - 1);
  factor = realPtr == c1_idx ? factor * compileInfo.pad_align_size : factor;
  factor = output_shape[realPtr] > factor ? factor : output_shape[realPtr];

  mte->mainLen = factor * baseLen;
  mte->tailLen = (output_shape[realPtr] % factor) * baseLen;
}

void TransdataBase::StorageAlign(int64_t *new_input, int64_t *new_out, int64_t *input) {
  // init
  for (int64_t i = 0; i < tiling_length; i++) {
    new_input[i] = input[i];
  }
  // align for mte2(pad had been done)
  new_input[tiling_length - 1] = SetAlign(new_input[tiling_length - 1], compileInfo.align_size);
  // align for different instruction (only support 5HD and NZ)
  if (is_last_transpose) {
    bool is_fp16 = BLOCK / compileInfo.align_size == FP16;
    bool is_int8 = BLOCK / compileInfo.align_size == INT8;
    if (is_fp16 or is_int8) {
      new_input[tiling_length-2] = SetAlign(new_input[tiling_length-2], compileInfo.align_size);
    } else {
      //FP32(128),INT32(128),INT64(64)
      new_input[tiling_length-2] = SetAlign(new_input[tiling_length-2], 16 * compileInfo.align_size);
    }
  }

  // update new_out
  for (int64_t i = 0; i < tiling_length; i++) {
    new_out[i] = new_input[compileInfo.permute[i]];
  }
}

bool TransdataBase::CheckValidSplit(int64_t *input, int64_t *output, int64_t ptrA, int64_t ptrB) {
  // prtA: index of input
  // ptrB: index of output
  // new_c0 is based on input
  bool valid_split = compileInfo.permute[ptrB] <= ptrA and compileInfo.permute[ptrA] <= ptrB;
  int64_t base = Prod(input, ptrA + 1, tiling_length);
  for (int64_t idx = ptrB + 1; idx < tiling_length; idx++) {
    base = compileInfo.permute[idx] < ptrA ? base * output[idx] : base;
  }
  bool not_exceed_ub = base <= UBSize;
  if (isDataMove) {
    return valid_split && not_exceed_ub;
  }

  int64_t new_c0 = c0_idx;
  int64_t new_c1 = c1_idx;
  if (not compileInfo.is_forward) {
    new_c0 = compileInfo.permute[c0_idx];
    new_c1 = compileInfo.permute[c1_idx];
  }

  // not support split c0
  bool not_split_c0 = new_c0 != ptrA and compileInfo.permute[new_c0] != ptrB;
  // common align need tiling not split last dim
  bool valid_common_align = shapeType != 1;
  if(shapeType == 1) {
    if (compileInfo.is_forward) {
      // if input is [n,h,c], split c1 make h and c not serial
      if (new_c0 != tiling_length - 1) {
        // input is n c1 c0 h (n,c,h)
        valid_common_align = compileInfo.permute[ptrB] != tiling_length - 1 and ptrA != tiling_length - 1;
      } else {
        // input is n h c1 c0 (n,h,c)
        valid_common_align = compileInfo.permute[ptrB] < tiling_length - 2 and ptrA < tiling_length - 2;
      }
    } else {
      // if output is [n,h,c], split c1 make h and c not serial
      if (compileInfo.permute[new_c0] != tiling_length - 1) {
        // out is n c1 c0 h
        valid_common_align = compileInfo.permute[ptrA] != tiling_length - 1 and ptrB != tiling_length - 1;
      } else {
        // out is n h c1 c0
        valid_common_align = compileInfo.permute[ptrA] != tiling_length - 2 and ptrB != tiling_length - 2;
      }
    }
  }

  // template avoid emit_insn of pass
  bool is_fp32 = BLOCK / compileInfo.align_size == FP32;
  if (is_last_transpose and is_fp32 and input[tiling_length - 2] > 128) {
    if (ptrA < tiling_length - 2) {
      return false;
    }
  }

  return valid_split and not_split_c0 and not_exceed_ub and valid_common_align;
}

bool TransdataBase::InferOutput() {
  // forward computation: input infer output
  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  ge::ConstGeTensorDescPtr input_desc = op_desc->GetInputDescPtr(0);
  auto input_ge_shape = input_desc->GetShape();
  size_t ori_length = input_ge_shape.GetDimNum();
  for (size_t i = 0; i < ori_length; i++) {
    input_shape[i] = input_ge_shape.GetDim(i);
  }
  return DoFusing(input_shape, output_shape, ori_length);
}

bool TransdataBase::InferInput() {
  // backward computation: output infer input
  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc(0);
  auto output_ge_shape = output_desc->MutableShape();
  size_t ori_length = output_ge_shape.GetDimNum();
  for (size_t i = 0; i < ori_length; i++) {
    output_shape[i] = output_ge_shape.GetDim(i);
  }
  return DoFusing(output_shape, input_shape, ori_length);
}

bool TransdataBase::DoFusing(int64_t *input, int64_t *output, size_t ori_length) {
  /*
   * If backward, input is output_shape, output is input_shape.
   * If forward, input is input_shape, output is output_shape.
   * ori_length is valid size of input.
   * Infer Regulation (Forward):
   *    input_shape __fuse__ input_shape (const had been fused in classifier)
   *    input_shape __pad__ pad_shape
   *    pad_shape __reshape__ reshape
   *    reshape __transpose__ output_shape
   */
  size_t length_input = compileInfo.src_pad.size();
  size_t length_output = compileInfo.permute.size();
  size_t root_ptr = 0;
  int64_t pad_mark = 0;
  int64_t align = compileInfo.pad_align_size;

  // __fuse__
  if (not compileInfo.is_const) {
    while (root_ptr + 1 <= length_input) {
      if (root_ptr == length_input - 1 and compileInfo.src_fuse[root_ptr] == ori_length - 1) {
        input[root_ptr] = input[compileInfo.src_fuse[root_ptr]];
        break;
      }
      int64_t begin = compileInfo.src_fuse[root_ptr];
      int64_t end = root_ptr + 2 <= length_input ? compileInfo.src_fuse[root_ptr + 1] : ori_length;
      int64_t base = 1;
      for (int64_t i = begin; i < end; i++) {
        base *= input[i];
      }
      input[root_ptr] = base;
      root_ptr ++;
    }
  }

  // __pad__ and __reshape__
  // pad_mark is 0 that do nothing
  // pad_mark is 1 that do padding
  // pad_mark is 2 that split dim to C1 and C0
  root_ptr = 0;
  for (size_t i = 0; i < length_input; i++) {
    pad_mark = compileInfo.src_pad[i];
    if (pad_mark == 0) {
      reshape[root_ptr] = input[i];
      reshape_mapping_output[root_ptr] = i;
      root_ptr++;
    } else if (pad_mark == 1) {
      reshape[root_ptr] = SetAlign(input[i], align);
      reshape_mapping_output[root_ptr] = i;
      root_ptr++;
    } else {
      isDataMove = false;
      reshape[root_ptr] = SetAlign(input[i], align) / align;
      reshape[root_ptr + 1] = align;
      reshape_mapping_output[root_ptr] = i;
      reshape_mapping_output[root_ptr + 1] = i;
      // if forward, index of c0,c1 are based on reshape that reshape as input
      // if backward, index of c0,c1 are based on reshape that reshape as output
      c1_idx = root_ptr;
      c0_idx = root_ptr + 1;
      root_ptr += 2;
    }
  }

  // __transpose__
  for (size_t i = 0; i < length_output; i++) {
    output[compileInfo.permute[i]] = reshape[i];
  }
  return true;
}

void TransdataBase::FindAxisInUB(int64_t *axis_in_ub) {
  /*
   * 0 : ub_outer
   * 1 : ub_inner
   * 2 : first_split
   * 3 : second_split
   * base : output
   */
  // input
  for (int64_t idx = tilingInfo.ub_0_idx + 1; idx < tiling_length; idx++) {
    axis_in_ub[compileInfo.permute[idx]] = 1;
  }
  // output
  for (int64_t idx = tilingInfo.ub_1_idx + 1; idx <tiling_length; idx++) {
    axis_in_ub[idx] = 1;
  }
  // deal ub split axis
  axis_in_ub[compileInfo.permute[tilingInfo.ub_0_idx]] = 2;
  axis_in_ub[tilingInfo.ub_1_idx] = 3;
}

void TransdataBase::ForwardBlockProcess(int64_t *axis_in_ub) {
  // update factor by output_shape(not storage align and transpose align)
  if (tilingInfo.ub_0_factor > output_shape[compileInfo.permute[tilingInfo.ub_0_idx]]) {
    tilingInfo.ub_0_factor = output_shape[compileInfo.permute[tilingInfo.ub_0_idx]];
  }
  if (tilingInfo.ub_1_factor > output_shape[tilingInfo.ub_1_idx]) {
    tilingInfo.ub_1_factor = output_shape[tilingInfo.ub_1_idx];
  }

  // find split index
  // last dim of forward's output is C0 that make output >= BLOCK
  int64_t base = 1;
  int64_t first_factor = tilingInfo.ub_0_factor;
  int64_t second_factor = tilingInfo.ub_1_factor;
  int64_t slide_idx = 0;
  int64_t block_idx = 0;
  bool exceed_limit = false;

  // find split idx
  while(slide_idx < tiling_length) {
    if (base >= compileInfo.core_num) {
      exceed_limit = true;
      break;
    }
    if (axis_in_ub[slide_idx] == 1) {
      slide_idx += 1;
      continue;
    }
    if (axis_in_ub[slide_idx] == 0) {
      base *= output_shape[slide_idx];
    } else {
      base *= axis_in_ub[slide_idx] == 2 ? SetAlign(output_shape[slide_idx], first_factor) / first_factor:
              SetAlign(output_shape[slide_idx], second_factor) / second_factor;
    }
    slide_idx += 1;
    block_idx = slide_idx;
  }

  // assure value
  int64_t dim_bound = axis_in_ub[block_idx - 1] == 0 ? output_shape[block_idx - 1] :
                      axis_in_ub[block_idx - 1] == 2 ?
                      SetAlign(output_shape[block_idx - 1], first_factor) / first_factor :
                      SetAlign(output_shape[block_idx - 1], second_factor) / second_factor;

  tilingInfo.blk_idx = block_idx - 1;
  if (not exceed_limit) {
    tilingInfo.blk_dim = base;
    tilingInfo.blk_factor = 1;
  } else {
    base = base / dim_bound;
    for (int64_t factor = 1; factor <= dim_bound; factor++) {
      int64_t outer = base * ((dim_bound + factor - 1) / factor);
      if (outer <= compileInfo.core_num) {
        tilingInfo.blk_dim = outer;
        tilingInfo.blk_factor = factor;
        break;
      }
    }
  }
}

void TransdataBase::BackwardBlockProcess(int64_t *axis_in_ub) {
  // update factor by reshape (not storage align and transpose align)
  if (tilingInfo.ub_0_factor > reshape[compileInfo.permute[tilingInfo.ub_0_idx]]) {
    tilingInfo.ub_0_factor = reshape[compileInfo.permute[tilingInfo.ub_0_idx]];
  }
  if (tilingInfo.ub_1_factor > reshape[tilingInfo.ub_1_idx]) {
    tilingInfo.ub_1_factor = reshape[tilingInfo.ub_1_idx];
  }

  // find split index
  // last dim of forward's output is C0 that make output >= BLOCK
  int64_t base = 1;
  int64_t first_factor = tilingInfo.ub_0_factor;
  int64_t second_factor = tilingInfo.ub_1_factor;
  int64_t slide_idx = 0;
  int64_t block_idx = 0;
  bool exceed_limit = false;

  // find split idx
  while(slide_idx < tiling_length) {
    if (axis_in_ub[slide_idx] == 1) {
      slide_idx += 1;
      continue;
    }
    if (axis_in_ub[slide_idx] == 0) {
      base *= reshape[slide_idx];
    } else {
      base *= axis_in_ub[slide_idx] == 2 ? SetAlign(reshape[slide_idx], first_factor) / first_factor:
              SetAlign(reshape[slide_idx], second_factor) / second_factor;
    }

    block_idx = slide_idx;
    if (base >= compileInfo.core_num) {
      exceed_limit = true;
      break;
    }
    slide_idx += 1;
  }

  // assure value
  int64_t dim_bound = axis_in_ub[block_idx] == 0 ? reshape[block_idx] :
                      axis_in_ub[block_idx] == 2 ?
                      SetAlign(reshape[block_idx], first_factor) / first_factor :
                      SetAlign(reshape[block_idx], second_factor) / second_factor;

  tilingInfo.blk_idx = block_idx;
  if (not exceed_limit) {
    tilingInfo.blk_dim = base;
    tilingInfo.blk_factor = 1;
  } else {
    base = base / dim_bound;
    for (int64_t factor = 1; factor <= dim_bound; factor++) {
      int64_t outer = base * ((dim_bound + factor - 1) / factor);
      if (outer <= compileInfo.core_num) {
        tilingInfo.blk_dim = outer;
        tilingInfo.blk_factor = factor;
        break;
      }
    }
  }
}

bool TransdataBase::BaseBlockTiling() {
  int64_t axis_in_ub[tiling_length];
  memset(axis_in_ub, 0, sizeof(axis_in_ub));
  if (compileInfo.is_forward) {
    FindAxisInUB(axis_in_ub);
    ForwardBlockProcess(axis_in_ub);
  } else {
    FindAxisInUB(axis_in_ub);
    BackwardBlockProcess(axis_in_ub);
  }
  return true;
}

bool TransdataBase::BaseUBTiling() {
  /*
   * <Forward> ------------------<Backward>
   * input-<PAD>-pad_shape-<RESHAPE>-<TRANSPOSE>-out
   */
  int64_t tiling_input[tiling_length];
  int64_t tiling_output[tiling_length];
  if (compileInfo.is_forward) {
    // reshape -> out
    StorageAlign(tiling_input, tiling_output, reshape);
    V_OP_TILING_CHECK(UBTilingFilter(tiling_input, tiling_output), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "UBTilingFilter Failed"), return false);
    V_OP_TILING_CHECK(UBTilingForwardProcess(tiling_input, tiling_output), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "UBTilingForwardProcess Failed"), return false);
  } else {
    // input -> reshape -> ac_out
    StorageAlign(tiling_input, tiling_output, input_shape);
    V_OP_TILING_CHECK(UBTilingFilter(tiling_input, tiling_output), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "UBTilingFilter Failed"), return false);
    V_OP_TILING_CHECK(UBTilingBackwardProcess(tiling_input, tiling_output), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "UBTilingBackwardProcess Failed"), return false);
  }
  return true;
}

bool TransdataBase::UBTilingFilter(int64_t *input, int64_t *output) {
  // ptrA: index of input
  // ptrB: index of output
  ub_tiling_num = 0;
  for (int64_t ptrA = tiling_length - 1; ptrA >= 0; ptrA--) {
    for (int64_t ptrB = tiling_length - 1; ptrB >= 0; ptrB--) {
      if (CheckValidSplit(input, output, ptrA, ptrB)) {
        possible_ub_tiling[ub_tiling_num*2] = ptrA;
        possible_ub_tiling[ub_tiling_num*2+1] = ptrB;
        ub_tiling_num++;
      }
    }
  }
  return ub_tiling_num > 0;
}

bool TransdataBase::UBTilingForwardProcess(int64_t *input, int64_t *output) {
  /*
   * 1. In forward, output is mte3 that is align, input is mte2 that maybe not align.
   * 2. Firstly make mte2
   * 3. mte3.mainLen and mte3.tailLen are bigger than align size.
   */
  int64_t ptrI = -1;
  int64_t ptrO = -1;
  int64_t ub_0_factor = -1;
  int64_t ub_1_factor = -1;
  int64_t core = 1;
  float percent = 0;
  bool run_out_ub = false;
  bool split_once = false;

  int64_t num_in_ub = 0;
  int64_t total_num = Prod(input, 0, tiling_length);
  int64_t all_times = ub_tiling_num;
  int64_t mte_standard = 256 / (BLOCK / compileInfo.align_size);

  // update template value
  bool mte3_arrive_256B = false;
  bool mte2_arrive_256B = false;

  while (all_times >= 1) {
    // PreProcess
    ptrI = possible_ub_tiling[(all_times - 1) * 2];
    ptrO = possible_ub_tiling[(all_times - 1) * 2 + 1];
    mte2.virLen = Prod(input, ptrI + 1, tiling_length);
    mte3.virLen = Prod(output, ptrO + 1, tiling_length);
    num_in_ub = mte2.virLen;
    for (int64_t idx = ptrO + 1; idx < tiling_length; idx++) {
      num_in_ub = compileInfo.permute[idx] < ptrI ? num_in_ub * output[idx] : num_in_ub;
    }

    // MainTiling
    if (compileInfo.permute[ptrI] == ptrO) {
      // Step0: Init MTE3/MTE2
      split_once = true;
      run_out_ub = output[ptrO] * num_in_ub >= UBSize;
      // split axis which do transpose need special align (m,n)->(n,m)
      int64_t boundO = run_out_ub ? UBSize / num_in_ub : output[ptrO];
      if (is_last_transpose) {
        if (ptrO == tiling_length - 2) {
          boundO = boundO / compileInfo.align_size * compileInfo.align_size;
        } else if (ptrO == tiling_length - 1) {
          bool is_fp16 = BLOCK / compileInfo.align_size == FP16;
          bool is_int8 = BLOCK / compileInfo.align_size == INT8;
          boundO = is_fp16 or is_int8 ?
                   boundO / compileInfo.align_size * compileInfo.align_size :
                   16 * compileInfo.align_size;
        }
      }

      for (int64_t factor = 1; factor <= boundO; factor++) {
        core = total_num / num_in_ub / output[ptrO] * ((output[ptrO] + factor - 1) / factor);
        mte3_arrive_256B = factor * mte3.virLen >= mte_standard;
        mte2_arrive_256B = factor * mte2.virLen >= mte_standard;
        ub_0_factor = factor;
        ub_1_factor = factor;
        if (mte3_arrive_256B and mte2_arrive_256B) {
          break;
        }
      }

      // Step1: Adjust MTE3/MTE2
      core = total_num / num_in_ub / output[ptrO] * ((output[ptrO] + ub_1_factor - 1) / ub_1_factor);
      if (core > compileInfo.core_num) {
        // extra core can improve UB usage
        for (int64_t factor = ub_1_factor; factor <= boundO; factor++) {
          core = total_num / num_in_ub / output[ptrO] * ((output[ptrO] + factor - 1) / factor);
          if (core == compileInfo.core_num) {
            // use current value
            ub_0_factor = factor;
            ub_1_factor = factor;
            break;
          } else if (core < compileInfo.core_num) {
            // used current - 1
            core = total_num / num_in_ub / output[ptrO] * ((output[ptrO] + ub_1_factor - 1) / ub_1_factor);
            break;
          }
          ub_0_factor = factor;
          ub_1_factor = factor;
        }
      }

      percent = float(ub_1_factor * num_in_ub) / float(UBSize);
      mte2.virLen = ub_0_factor * mte2.virLen;
      mte3.virLen = ub_1_factor * mte3.virLen;
    } else {
      /*
       * step0: Init MTE2/MTE3, let MTE2/MTE3 arrive 256B
       * step1: Fix MTE3, Adjust MTE2
       * step2: Fix MTE2, Adjust MTE3
       * */
      split_once = false;
      int64_t boundF = UBSize / num_in_ub;
      int64_t boundI = boundF > input[ptrI] ? input[ptrI] : boundF;
      int64_t boundO = boundF > output[ptrO] ? output[ptrO] : boundF;
      int64_t base_core = total_num / num_in_ub / output[ptrO] / input[ptrI];
      int64_t ub_0_outer = 1;
      int64_t ub_1_outer = 1;

      // step0: Init MTE2
      for (int64_t factor = 1; factor <= boundI; factor++) {
        ub_0_factor = factor;
        mte2_arrive_256B = factor * mte2.virLen >= mte_standard;
        if (mte2_arrive_256B) {
          break;
        }
      }

      // step0: Init MTE3
      ub_0_outer = (input[ptrI] + ub_0_factor - 1) / ub_0_factor;
      boundO = boundF / ub_0_factor;
      boundO = boundO > output[ptrO] ? output[ptrO] : boundO;
      for (int64_t factor = 1; factor <= boundO; factor++) {
        ub_1_outer = (output[ptrO] + factor - 1) / factor;
        core = base_core * ub_0_outer * ub_1_outer;
        mte3_arrive_256B = factor * mte3.virLen >= mte_standard;
        ub_1_factor = factor;
        if (mte3_arrive_256B) {
          break;
        }
      }

      // step1: Adjust MTE2
      ub_0_outer = (input[ptrI] + ub_0_factor - 1) / ub_0_factor;
      ub_1_outer = (output[ptrO] + ub_1_factor - 1) / ub_1_factor;
      core = base_core * ub_0_outer * ub_1_outer;
      boundI = boundF / ub_1_factor;
      boundI = boundI > input[ptrI] ? input[ptrI] : boundI;
      if (core > compileInfo.core_num) {
        // extra core can improve UB usage
        for (int64_t factor = ub_0_factor; factor <= boundI; factor++) {
          ub_0_outer = (input[ptrI] + factor - 1) / factor;
          core = base_core * ub_0_outer * ub_1_outer;
          if (core == compileInfo.core_num) {
            ub_0_factor = factor;
            break;
          } else if (core < compileInfo.core_num) {
            ub_0_outer = (input[ptrI] + ub_0_factor - 1) / ub_0_factor;
            core = base_core * ub_0_outer * ub_1_outer;
            break;
          }
          ub_0_factor = factor;
        }
      }

      // step2: Adjust MTE3
      ub_0_outer = (input[ptrI] + ub_0_factor - 1) / ub_0_factor;
      ub_1_outer = (output[ptrO] + ub_1_factor - 1) / ub_1_factor;
      core = base_core * ub_0_outer * ub_1_outer;
      boundO = boundF / ub_0_factor;
      boundO = boundO > output[ptrO] ? output[ptrO] : boundO;
      if (core > compileInfo.core_num) {
        // extra core can improve UB usage
        for (int64_t factor = ub_1_factor; factor <= boundO; factor++) {
          ub_1_outer = (output[ptrO] + factor - 1) / factor;
          core = base_core * ub_1_outer * ub_0_outer;
          if (core == compileInfo.core_num) {
            ub_1_factor = factor;
            break;
          } else if (core < compileInfo.core_num) {
            ub_1_outer = (output[ptrO] + ub_1_factor - 1) / ub_1_factor;
            core = base_core * ub_1_outer * ub_0_outer;
            break;
          }
          ub_1_factor = factor;
        }
      }

      // Result
      mte2.virLen = ub_0_factor * mte2.virLen;
      mte3.virLen = ub_1_factor * mte3.virLen;
      percent = float(ub_0_factor * ub_1_factor * num_in_ub) / float(UBSize);
    }

    // Init
    if (tilingInfo.percent == -1) {
      UBTilingUpdate(ptrI, ptrO, ub_0_factor, ub_1_factor, mte2.virLen, mte3.virLen, percent, core);
      all_times--;
      continue;
    }

    // Compare
    int64_t new_core = percent >= 0.95 and core > compileInfo.core_num ? compileInfo.core_num : core;
    int64_t old_core = tilingInfo.percent >= 0.95 and tilingInfo.core > compileInfo.core_num ? compileInfo.core_num : tilingInfo.core;
    int64_t new_core_distance = abs(new_core - compileInfo.core_num);
    int64_t old_core_distance = abs(old_core - compileInfo.core_num);

    if (new_core_distance < old_core_distance) {
      UBTilingUpdate(ptrI, ptrO, ub_0_factor, ub_1_factor, mte2.virLen, mte3.virLen, percent, core);
    } else if (new_core_distance == old_core_distance) {
      if (split_once and not tilingInfo.split_once) {
        UBTilingUpdate(ptrI, ptrO, ub_0_factor, ub_1_factor, mte2.virLen, mte3.virLen, percent, core);
      } else if (split_once and tilingInfo.split_once) {
        if (mte3.virLen > tilingInfo.mte3_burst_len) {
          UBTilingUpdate(ptrI, ptrO, ub_0_factor, ub_1_factor, mte2.virLen, mte3.virLen, percent, core);
        }
      }
    }
    all_times--;
  }

  if (tilingInfo.split_once) {
    // deal factor
    // deal pure data move
    int64_t mainFactor = tilingInfo.ub_1_factor;
    int64_t dimBound = output[tilingInfo.ub_1_idx];
    int64_t tailFactor = dimBound % mainFactor;
    float tail_percent = (float)(tailFactor) / (float)(mainFactor);
    float fine_tuning_threshold = 0.8;
    if (tail_percent == 0 or tail_percent >= 0.8) {
      return true;
    }
    int loop = dimBound / mainFactor + 1;
    mainFactor = (dimBound % loop) ? (dimBound / loop + 1) : (dimBound / loop);
    tilingInfo.ub_0_factor = mainFactor;
    tilingInfo.ub_1_factor = mainFactor;
  }
  return true;
}

bool TransdataBase::UBTilingBackwardProcess(int64_t *input, int64_t *output) {
  /*
   * 1. In backward, input is mte2 that is align, output is mte3 that may be not align.
   * 2. Firstly make mte3
   * 3. MTE3 need protect mte3.mainLen Core and mte3.tailLen bigger than align_size.
   * */
  int64_t ptrI = -1;
  int64_t ptrO = -1;
  int64_t ub_0_factor = -1;
  int64_t ub_1_factor = -1;
  int64_t core = 1;
  float percent = 0;
  bool run_out_ub = false;
  bool init_success = false;
  bool split_once = false;

  int64_t num_in_ub = 0;
  int64_t total_num = Prod(input, 0, tiling_length);
  int64_t all_times = ub_tiling_num;
  int64_t mte_standard = 256 / (BLOCK / compileInfo.align_size);

  // update template value
  bool tail_is_legal = false;
  bool multi_core_is_legal = false;
  bool mte3_arrive_256B = false;
  bool mte2_arrive_256B = false;

  while (all_times >= 1) {
    // PreProcess
    init_success = false;
    ptrI = possible_ub_tiling[(all_times - 1) * 2];
    ptrO = possible_ub_tiling[(all_times - 1) * 2 + 1];
    mte2.virLen = Prod(input, ptrI + 1, tiling_length);
    mte3.virLen = Prod(output, ptrO + 1, tiling_length);
    num_in_ub = mte2.virLen;
    for (int64_t idx = ptrO + 1; idx < tiling_length; idx++) {
      num_in_ub = compileInfo.permute[idx] < ptrI ? num_in_ub * output[idx] : num_in_ub;
    }

    // MainTiling
    if (compileInfo.permute[ptrI] == ptrO) {
      // Step0: Init MTE3/MTE2
      split_once = true;
      run_out_ub = output[ptrO] * num_in_ub >= UBSize;
      // split axis which do transpose need special align (m,n)->(n,m)
      int64_t boundO = run_out_ub ? UBSize / num_in_ub : output[ptrO];
      if (is_last_transpose) {
        if (ptrO == tiling_length - 2) {
          boundO = boundO / compileInfo.align_size * compileInfo.align_size;
        } else if (ptrO == tiling_length - 1) {
          bool is_fp16 = BLOCK / compileInfo.align_size == FP16;
          bool is_int8 = BLOCK / compileInfo.align_size == INT8;
          boundO = is_fp16 or is_int8 ?
                   boundO / compileInfo.align_size * compileInfo.align_size :
                   16 * compileInfo.align_size;
        }
      }

      for (int64_t factor = 1; factor <= boundO; factor++) {
        // mte3_tail should bigger than 16(float16) or is 0, must used real output
        GetOutPutRealTail(ptrO, factor, &mte3);
        core = total_num / num_in_ub / output[ptrO] * ((output[ptrO] + factor - 1) / factor);
        tail_is_legal = mte3.tailLen == 0 or mte3.tailLen >= compileInfo.align_size;
        multi_core_is_legal = mte3.mainLen >= compileInfo.align_size or core == 1;
        mte3_arrive_256B = factor * mte3.virLen >= mte_standard;
        mte2_arrive_256B = factor * mte2.virLen >= mte_standard;
        if (tail_is_legal and multi_core_is_legal) {
          init_success = true;
          ub_0_factor = factor;
          ub_1_factor = factor;
          if (mte3_arrive_256B and mte2_arrive_256B) {
            break;
          }
        }
      }

      // StepX: Init success
      if (not init_success) {
        all_times--;
        continue;
      }

      // Step1: Adjust MTE3/MTE2
      core = total_num / num_in_ub / output[ptrO] * ((output[ptrO] + ub_1_factor - 1) / ub_1_factor);
      if (core > compileInfo.core_num) {
        // extra core can improve UB usage
        for (int64_t factor = ub_1_factor; factor <= boundO; factor++) {
          // mte3_tail should bigger than 16 (float16) or is 0
          GetOutPutRealTail(ptrO, factor, &mte3);
          tail_is_legal = mte3.tailLen == 0 or mte3.tailLen >= compileInfo.align_size;
          if (tail_is_legal) {
            core = total_num / num_in_ub / output[ptrO] * ((output[ptrO] + factor - 1) / factor);
            if (core == compileInfo.core_num) {
              // use current value
              ub_0_factor = factor;
              ub_1_factor = factor;
              break;
            } else if (core < compileInfo.core_num) {
              // used current - 1
              core = total_num / num_in_ub / output[ptrO] * ((output[ptrO] + ub_1_factor - 1) / ub_1_factor);
              break;
            }
            ub_0_factor = factor;
            ub_1_factor = factor;
          }
        }
      }

      percent = float(ub_1_factor * num_in_ub) / float(UBSize);
      mte2.virLen = ub_0_factor * mte2.virLen;
      mte3.virLen = ub_1_factor * mte3.virLen;
    } else {
      /*
       * step0: Init MTE2/MTE3, let MTE2/MTE3 arrive 256B
       * step1: Fix MTE2, Adjust MTE3
       * step2: Fix MTE3, Adjust MTE2
       * */
      split_once = false;
      int64_t boundF = UBSize / num_in_ub;
      int64_t boundI = boundF > input[ptrI] ? input[ptrI] : boundF;
      int64_t boundO = boundF > output[ptrO] ? output[ptrO] : boundF;
      int64_t base_core = total_num / num_in_ub / output[ptrO] / input[ptrI];
      int64_t ub_0_outer = 1;
      int64_t ub_1_outer = 1;

      // step0: Init MTE2
      for (int64_t factor = 1; factor <= boundI; factor++) {
        ub_0_factor = factor;
        mte2_arrive_256B = factor * mte2.virLen >= mte_standard;
        if (mte2_arrive_256B) {
          break;
        }
      }

      // step0: Init MTE3
      ub_0_outer = (input[ptrI] + ub_0_factor - 1) / ub_0_factor;
      boundO = boundF / ub_0_factor;
      boundO = boundO > output[ptrO] ? output[ptrO] : boundO;
      for (int64_t factor = 1; factor <= boundO; factor++) {
        GetOutPutRealTail(ptrO, factor, &mte3);
        ub_1_outer = (output[ptrO] + factor - 1) / factor;
        core = base_core * ub_0_outer * ub_1_outer;
        tail_is_legal = mte3.tailLen == 0 or mte3.tailLen >= compileInfo.align_size;
        multi_core_is_legal = mte3.mainLen >= compileInfo.align_size or core == 1;
        mte3_arrive_256B = factor * mte3.virLen >= mte_standard;
        if (tail_is_legal and multi_core_is_legal) {
          init_success = true;
          ub_1_factor = factor;
          if (mte3_arrive_256B) {
            break;
          }
        }
      }

      // StepX: Init success
      if (not init_success) {
        all_times--;
        continue;
      }

      // step1: Adjust MTE3
      ub_0_outer = (input[ptrI] + ub_0_factor - 1) / ub_0_factor;
      ub_1_outer = (output[ptrO] + ub_1_factor - 1) / ub_1_factor;
      core = base_core * ub_0_outer * ub_1_outer;
      if (core > compileInfo.core_num) {
        // extra core can improve UB usage
        for (int64_t factor = ub_1_factor; factor <= boundO; factor++) {
          GetOutPutRealTail(ptrO, factor, &mte3);
          tail_is_legal = mte3.tailLen == 0 or mte3.tailLen >= compileInfo.align_size;
          ub_1_outer = (output[ptrO] + factor - 1) / factor;
          core = base_core * ub_1_outer * ub_0_outer;
          if (tail_is_legal) {
            if (core == compileInfo.core_num) {
              ub_1_factor = factor;
              break;
            } else if (core < compileInfo.core_num) {
              ub_1_outer = (output[ptrO] + ub_1_factor - 1) / ub_1_factor;
              core = base_core * ub_1_outer * ub_0_outer;
              break;
            }
            ub_1_factor = factor;
          }
        }
      }

      // step2: Adjust MTE2
      ub_0_outer = (input[ptrI] + ub_0_factor - 1) / ub_0_factor;
      ub_1_outer = (output[ptrO] + ub_1_factor - 1) / ub_1_factor;
      core = base_core * ub_0_outer * ub_1_outer;
      boundI = boundF / ub_1_factor;
      boundI = boundI > input[ptrI] ? input[ptrI] : boundI;
      if (core > compileInfo.core_num) {
        // extra core can improve UB usage
        for (int64_t factor = ub_0_factor; factor <= boundI; factor++) {
          ub_0_outer = (input[ptrI] + factor - 1) / factor;
          core = base_core * ub_0_outer * ub_1_outer;
          if (core == compileInfo.core_num) {
            ub_0_factor = factor;
            break;
          } else if (core < compileInfo.core_num) {
            ub_0_outer = (input[ptrI] + ub_0_factor - 1) / ub_0_factor;
            core = base_core * ub_0_outer * ub_1_outer;
            break;
          }
          ub_0_factor = factor;
        }
      }

      // Result
      mte2.virLen = ub_0_factor * mte2.virLen;
      mte3.virLen = ub_1_factor * mte3.virLen;
      percent = float(ub_0_factor * ub_1_factor * num_in_ub) / float(UBSize);
    }

    // Init
    if (tilingInfo.percent == -1) {
      UBTilingUpdate(ptrI, ptrO, ub_0_factor, ub_1_factor, mte2.virLen, mte3.virLen, percent, core);
      all_times--;
      continue;
    }

    // Compare
    int64_t new_core = percent >= 0.95 and core > compileInfo.core_num ? compileInfo.core_num : core;
    int64_t old_core = tilingInfo.percent >= 0.95 and tilingInfo.core > compileInfo.core_num ? compileInfo.core_num : tilingInfo.core;
    int64_t new_core_distance = abs(new_core - compileInfo.core_num);
    int64_t old_core_distance = abs(old_core - compileInfo.core_num);

    if (new_core_distance < old_core_distance) {
      UBTilingUpdate(ptrI, ptrO, ub_0_factor, ub_1_factor, mte2.virLen, mte3.virLen, percent, core);
    } else if (new_core_distance == old_core_distance) {
      if (split_once and not tilingInfo.split_once) {
        UBTilingUpdate(ptrI, ptrO, ub_0_factor, ub_1_factor, mte2.virLen, mte3.virLen, percent, core);
      } else if (split_once and tilingInfo.split_once) {
        if (mte3.virLen > tilingInfo.mte3_burst_len) {
          UBTilingUpdate(ptrI, ptrO, ub_0_factor, ub_1_factor, mte2.virLen, mte3.virLen, percent, core);
        }
      }
    }
    all_times--;
  }
  return true;
}

void TransdataBase::UBTilingUpdate(int64_t ptrA, int64_t ptrB, int64_t factorA, int64_t factorB, int64_t mte2,
                                   int64_t mte3, float percent, int64_t core) {
  tilingInfo.ub_0_idx = ptrA;
  tilingInfo.ub_1_idx = ptrB;
  tilingInfo.ub_0_factor = factorA;
  tilingInfo.ub_1_factor = factorB;
  tilingInfo.mte2_burst_len = mte2;
  tilingInfo.mte3_burst_len = mte3;
  tilingInfo.percent = percent;
  tilingInfo.core = core;
  tilingInfo.split_once = compileInfo.permute[ptrA] == ptrB;
}

bool TransdataBase::GetCompileInfo() {
  tiling_length = compileInfo.permute.size();
  if (tiling_length < 1) {
    return false;
  }
  is_last_transpose = compileInfo.permute[tiling_length - 1] != tiling_length - 1;
  return true;
}

bool TransdataBase::IsConstRuntime() {
  if (compileInfo.is_const && (not compileInfo.is_const_compile)) {
    std::string pattern_str = std::to_string(CONST_KEY);
    tilingInfo.blk_dim = compileInfo.const_block_dims.at(pattern_str);
    return true;
  }
  return false;
}

bool TransdataBase::GetInputOutput() {
  /*
   * if compileInfo.is_forward is 1, infer reshape\output by input
   * if compileInfo.is_forward is 0, infer reshape\input by output
   */
  bool ret = compileInfo.is_forward ? InferOutput() : InferInput();
  return ret;
}

bool TransdataBase::ChooseStrategy() {
  size_t length = compileInfo.src_pad.size();
  int64_t last_dim_length = compileInfo.is_forward ? input_shape[length - 1] : output_shape[length - 1];
  bool last_dim_not_align = last_dim_length % compileInfo.align_size != 0;
  bool is_fp16 = compileInfo.align_size == 16;

  // Init computeType and ShapeType
  computeType = BaseSch;
  shapeType = CommonAlignBranch;
  // Choose UB Info
  if (compileInfo.ub_info.size() < computeType + 1) {
    return false;
  }
  if (compileInfo.ub_info[computeType].size() < shapeType + 1) {
    return false;
  }

  // Choose shapeType
  shapeType = CommonAlignLimit(last_dim_length, compileInfo.ub_info[computeType][shapeType]) ?
              CommonAlignBranch : StorageAlignBranch;
  // Choose UBInfo
  UBSize = compileInfo.ub_info[computeType][shapeType];
  return true;
}

bool TransdataBase::CalcTiling() {
  V_OP_TILING_CHECK(GetCompileInfo(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetCompileInfo Failed"), return false);
  if (not IsConstRuntime()) {
    V_OP_TILING_CHECK(GetInputOutput(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetInputOutput Failed"), return false);
    V_OP_TILING_CHECK(ChooseStrategy(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ChooseStrategy Failed"), return false);
    if (computeType == BaseSch) {
      V_OP_TILING_CHECK(BaseUBTiling(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "BaseUBTiling Failed"), return false);
      V_OP_TILING_CHECK(BaseBlockTiling(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "BaseBlockTiling Failed"), return false);
    } else {
      return false;
    }
  }
  return true;
}

bool TransdataBase::WriteTilingData() {
  if (compileInfo.is_const && (not compileInfo.is_const_compile)) {
    // const runtime
    run_info.SetBlockDim(tilingInfo.blk_dim);
    run_info.SetTilingKey(CONST_KEY);
    return true;
  }

  if (compileInfo.is_const && compileInfo.is_const_compile) {
    run_info.AddTilingData((int32_t)computeType);
    run_info.AddTilingData((int32_t)shapeType);
    run_info.AddTilingData((int32_t)tilingInfo.blk_idx);
    run_info.AddTilingData((int32_t)compileInfo.permute[tilingInfo.ub_0_idx]);
    run_info.AddTilingData((int32_t)tilingInfo.ub_1_idx);
    run_info.AddTilingData((int32_t)tilingInfo.blk_factor);
    run_info.AddTilingData((int32_t)tilingInfo.ub_0_factor);
    run_info.AddTilingData((int32_t)tilingInfo.ub_1_factor);
    run_info.SetBlockDim((uint32_t)tilingInfo.blk_dim);
    return true;
  }

  // dynamic
  run_info.SetBlockDim((uint32_t)tilingInfo.blk_dim);
  int32_t tiling_key = CalcTilingKey();
  run_info.SetTilingKey((uint32_t)tiling_key);

  // convert dim which is input after fused
  int64_t * target_shape = compileInfo.is_forward ? input_shape : output_shape;
  for (int64_t i = 0; i < compileInfo.unknown_dims.size(); i++) {
    run_info.AddTilingData((int32_t)target_shape[compileInfo.unknown_dims[i]]);
    OP_LOGD(op_type.c_str(), "input shape : %d", target_shape[i]);
  }

  run_info.AddTilingData((int32_t)tilingInfo.blk_factor);
  if (tilingInfo.split_once) {
    run_info.AddTilingData((int32_t)tilingInfo.ub_1_factor);
  } else {
    run_info.AddTilingData((int32_t)tilingInfo.ub_0_factor);
    run_info.AddTilingData((int32_t)tilingInfo.ub_1_factor);
  }
  return true;
}

bool TransdataBase::DoTiling() {
  // main process
  bool ret = CalcTiling();
  return ret && WriteTilingData();
}

bool CompileInfoTransdataDSL::Check() {
  V_OP_TILING_CHECK((core_num > 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(transdata_op_type, "core_num is %ld that is illegal",
                                                    core_num),
                    return false);
  V_OP_TILING_CHECK((pad_align_size > 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(transdata_op_type, "pad_align_size is %ld that is illegal",
                                                    pad_align_size),
                    return false);
  V_OP_TILING_CHECK((align_size > 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(transdata_op_type, "align_size is %ld that is illegal",
                                                    align_size),
                    return false);
  return true;
}

CompileInfoTransdataDSL::CompileInfoTransdataDSL(const std::string& op_type, const nlohmann::json& parsed_json_obj) {
  transdata_op_type = op_type;
  OP_LOGD(transdata_op_type.c_str(), "transdata compile info construct func running");

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
  src_pad = parsed_json_obj.at("_src_pad").get<std::vector<int64_t>>();
  src_fuse = parsed_json_obj.at("_src_fuse").get<std::vector<int64_t>>();
  permute = parsed_json_obj.at("_permute").get<std::vector<int64_t>>();
  unknown_dims = parsed_json_obj.at("_unknown_dims").get<std::vector<int64_t>>();
  ub_info = parsed_json_obj.at("_ub_info").get<std::vector<std::vector<int64_t>>>();

  // collect vars
  if (is_const and not is_const_compile) {
    // const runtime
    const_block_dims = parsed_json_obj.at("_const_block_dims").get<std::unordered_map<std::string, int32_t>>();
  }

  check_success = Check();
}

bool TransdataTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const {
  OP_LOGD(op_type.c_str(), "Transdata DSL tiling running");
  TransdataBase transdata(op_type, op_paras, compileInfo, run_info);
  return transdata.DoTiling();
}

bool TransdataTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info,
                                      const OpInfo& op_info) const {
  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Transdata DSL custom tiling is not supported yet");
  return false;
}

std::shared_ptr<AutoTilingHandler> CreateTransdataTilingHandler(const std::string& op_type,
                                                                const std::string& pattern,
                                                                const nlohmann::json& parsed_compile_info) {
  auto compile_info_ptr = std::make_shared<TransdataTilingHandler>(op_type, pattern, parsed_compile_info);
  if (!compile_info_ptr->ParsedSuccess()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Transdata parse compile info failed");
    return std::shared_ptr<AutoTilingHandler>(nullptr);
  }

  return compile_info_ptr;
}

}// namespace optiling