/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
 * \file pad_common.cpp
 * \brief
 */
#include "pad_common.h"
#include "error_log.h"

// block size (byte)
const int32_t MINI_UNIT = 32;

namespace optiling {
int padCommon::_numBit(const std::string& dtype) {
  // Only Support FP16 and FP32.
  int numBit = 2;
  if (dtype == "float" || dtype == "int32") {
    numBit = 4;
  }
  return numBit;
}

int64_t padCommon::_prod(int64_t index, const std::vector<int64_t>& shape) {
  int64_t init = 1;
  init = std::accumulate(shape.begin() + index, shape.end(), init, std::multiplies<int64_t>());
  return init;
}

int64_t padCommon::_accumulate(int64_t index, const std::vector<int64_t>* shape_ptr, int64_t init) {
  init = std::accumulate(shape_ptr->begin(), shape_ptr->begin() + index, init);
  init = std::abs(init);
  return init;
}

int64_t padCommon::_align(int64_t value, int64_t align_vol) const {
  if (align_vol == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING("pad_common", "align_vol = 0 is not support");
      return -1;
  }
  return (value % align_vol != 0) ? (value / align_vol + 1) * align_vol : value;
}

void padCommon::InitTilingParams(PadDTilingParams& params, int num) {
  std::vector<int64_t> vec_init(num, 0);
  params.top_vol = vec_init;
  params.top_address = vec_init;
  params.top_div_core = vec_init;
  params.top_total_core = vec_init;
  params.top_core_vol_0 = vec_init;
  params.top_core_vol_1 = vec_init;
  params.top_core_gap_0 = vec_init;
  params.top_core_gap_1 = vec_init;

  params.bottom_vol = vec_init;
  params.bottom_address = vec_init;
  params.bottom_div_core = vec_init;
  params.bottom_total_core = vec_init;
  params.bottom_core_vol_0 = vec_init;
  params.bottom_core_vol_1 = vec_init;
  params.bottom_core_gap_0 = vec_init;
  params.bottom_core_gap_1 = vec_init;

  std::vector<std::vector<int64_t>> vec2_init(num, std::vector<int64_t>(2));
  params.recur_inShape = vec_init;
  params.recur_outShape = vec_init;
  params.recur_padding = vec2_init;
  params.recur_model = vec_init;
  params.recur_dup_mk = vec_init;
  params.recur_gm2buf_mk = vec_init;
  params.prod_recurIn = vec_init;
  params.prod_recurOut = vec_init;
}

int padCommon::CheckBranch(const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape,
                           const std::vector<std::vector<int64_t>>& padding, int numBit, int model) {
  int64_t axis = inShape.size() - 1;
  int64_t input = 0;
  while (axis >= 0) {
    int64_t in_padding = padding[axis][model] * _prod(axis + 1, outShape);
    input = _prod(axis, inShape);
    if (in_padding > 0) {
      if (in_padding * numBit % MINI_UNIT != 0 || input * numBit % MINI_UNIT != 0) {
        return 0;
      }
      return 1;
    }
    axis -= 1;
  }
  // not padding
  if (input * numBit % MINI_UNIT != 0) {
    return 0;
  }
  return 1;
}

void padCommon::fused(std::vector<int64_t>& out, std::vector<int64_t> in, int64_t distance, int64_t begin,
                      std::vector<std::vector<int64_t>>& out_pad, std::vector<std::vector<int64_t>>& in_pad,
                      bool mark) {
  int64_t temp = 1;
  int64_t pad_left = in_pad[begin][0];
  int64_t pad_right = in_pad[begin][1];
  if (distance > 0) {
    for (int64_t i = begin; i < begin + distance; i++) {
      temp *= in[i];
    }
    out.push_back(temp);
    if (mark) {
      out_pad.push_back(std::vector<int64_t>({pad_left * temp / in[begin], pad_right * temp / in[begin]}));
    }
  } else {
    out.push_back(in[begin]);
    if (mark) {
      out_pad.push_back(in_pad[begin]);
    }
  }
}

void padCommon::fused_special(std::vector<int64_t>& out, std::vector<int64_t> in, int64_t head_ptr,
                              std::vector<std::vector<int64_t>>& out_pad, std::vector<std::vector<int64_t>> in_pad,
                              bool mark) {
  std::vector<int64_t> sample = {0, 0};
  if (head_ptr != 0) {
    int64_t temp = 1;
    if (in_pad[head_ptr] != sample) {
      for (int64_t i = head_ptr - 1; i >= 0; i--) {
        temp *= in[i];
      }
      out.push_back(temp);
    } else {
      for (int64_t i = head_ptr; i >= 0; i--) {
        temp *= in[i];
      }
      out.push_back(temp);
    }
    if (mark) {
      out_pad.push_back(std::vector<int64_t>({0, 0}));
    }
  }
}

bool padCommon::FusedAxis(std::vector<int64_t>& n_inShape, std::vector<int64_t>& n_outShape,
                          std::vector<std::vector<int64_t>>& n_padding, const std::vector<int64_t>& inShape,
                          const std::vector<int64_t>& outShape, std::vector<std::vector<int64_t>> padding) {
  /*
   *  (0,0), (0,0), (1,1), (0,0), (1,1), (0,0), (0,0)
   *                                             | |
   *                                |             |
   *                  |             |
   * fused [head_ptr, tail_ptr] while tail_ptr point (0,0).
   * fused [head_ptr, tail_ptr) while tail_ptr point others.
   */

  int64_t length = inShape.size();
  int64_t head_ptr = length - 1;
  int64_t tail_ptr = length - 1;
  int64_t gap = 0;
  std::vector<int64_t> sample = {0, 0};

  for (int64_t i = length - 1; i >= 0; i--) {
    if (padding[i] != sample) {
      tail_ptr = head_ptr;
      head_ptr = i;

      // calc gap.
      if (padding[tail_ptr] != sample) {
        gap = tail_ptr - head_ptr;
        gap = (gap > 1) ? gap : 0;
      } else {
        gap = tail_ptr - head_ptr + 1;
      }

      // fill new shapes and padding.
      fused(n_inShape, inShape, gap, head_ptr, n_padding, padding, true);
      fused(n_outShape, outShape, gap, head_ptr, n_padding, padding, false);
    } else {
      if (length == 1) {
        n_inShape.push_back(inShape[0]);
        n_outShape.push_back(outShape[0]);
        n_padding.push_back(padding[0]);
      }
    }
  }

  // special solution.
  fused_special(n_inShape, inShape, head_ptr, n_padding, padding, true);
  fused_special(n_outShape, outShape, head_ptr, n_padding, padding, false);

  // reverse.
  std::reverse(n_inShape.begin(), n_inShape.end());
  std::reverse(n_outShape.begin(), n_outShape.end());
  std::reverse(n_padding.begin(), n_padding.end());

  if (n_inShape.size() != n_outShape.size() or n_inShape.size() != n_padding.size()) {
    VECTOR_INNER_ERR_REPORT_TILIING("PadDTiling", "Length of shape after fused error");
    return false;
  }

  return true;
}

void padCommon::GetDepth(const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape,
                         const std::vector<std::vector<int64_t>>& padding, int64_t& depth, int coreNum, int numBit,
                         int branch) {
  /*
   * "Depth": to decide: recursion layer will used which axises for cores.
   * */
  using namespace std;
  if (numBit == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING("pad_common", "numbit = 0 is not support");
      return;
  }
  int block_num = MINI_UNIT / numBit;
  int64_t size = inShape.size();
  int64_t pro_value = 0;

  // Eliminate "1" in head: [1,1,2,..,] -> [2,...]
  for (int i = 1; i <= size; i++) {
    pro_value = accumulate(inShape.begin(), inShape.begin() + i, 1, multiplies<int64_t>());
    depth = i;
    if (pro_value != 1) {
      break;
    }
  }

  // Determine whether to (depth + 1).
  // outBound: exceed boundary.
  // moreData: axis in [0, depth) could be splited for muilt cores in recursion layer.
  depth += 1;
  bool outBound = int64_t(inShape.size()) - depth >= 0;
  bool moreData = (!outBound) ? false : inShape[depth - 1] * _prod(depth, outShape) >= block_num;
  bool moreCore = inShape[depth - 2] < coreNum;
  depth = (outBound and moreData and moreCore) ? depth : depth - 1;

  // If branch is not align, pad of top or bottom maybe in (0,block_num) which
  // can't satisfy regulation of circulation layer.
  // In align, min(depth) = 1, In not align, min(depth) = 0.
  if (branch == 0) {
    int64_t top_vol = 0;
    int64_t bottom_vol = 0;
    while (depth > 0) {
      top_vol = _prod(depth, outShape) * padding[depth - 1][0];
      bottom_vol = _prod(depth, outShape) * padding[depth - 1][1];
      if ((top_vol == 0 || top_vol >= block_num) && (bottom_vol == 0 || bottom_vol >= block_num)) {
        break;
      }
      depth -= 1;
    }
  }
}

void padCommon::_calc_core_circulation(int branch, int index, const std::string& pattern, const int numBit,
                                       const int maxCore, const std::vector<int64_t>* vol,
                                       const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape,
                                       std::vector<int64_t>* total_core, std::vector<int64_t>* div_core,
                                       std::vector<int64_t>* core_vol_0, std::vector<int64_t>* core_vol_1,
                                       std::vector<int64_t>* core_gap_0, std::vector<int64_t>* core_gap_1) {
  /* Deal with info of circulation.
   * Branch is align(1), the first layer in circulation could be deal with MultiCore.
   * Branch is not align(0), the first layer in circulation could be deal with MultiCore.
   * In fact, "SingleCore" or "MultiCore" are same in performance while they used the same bus.
   * But cost of multi cmds in "MultiCore" would be less than "SingleCore".
   * Re:
   * In not align(0) and index is 0: data of div_core is more than core_vol_0.
   * core_gap_0 == core_gap_1: Because prod is always less than MaxCore in the func.
   * */
  int64_t virCore = 0;
  if (numBit == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING("pad_common", "numbit = 0 is not support");
      return;
  }
  int64_t block_num = MINI_UNIT / numBit;
  int64_t prod = std::accumulate(inShape.begin(), inShape.begin() + index, 1, std::multiplies<int64_t>());
  if (vol->at(index) > 0) {
    virCore = (prod != 1) ? inShape[index - 1] : vol->at(index) / block_num;
    total_core->at(index) = (virCore > maxCore) ? maxCore : virCore;

    div_core->at(index) =
        (virCore <= maxCore) ? virCore - 1 : (virCore % maxCore == 0) ? maxCore - 1 : virCore % maxCore - 1;

    int64_t loop_0 = (virCore % maxCore > 0) ? virCore / maxCore + 1 : virCore / maxCore;
    int64_t loop_1 = (div_core->at(index) + 1 != total_core->at(index)) ? loop_0 - 1 : loop_0;

    core_vol_0->at(index) = (prod != 1) ? loop_0 * vol->at(index) : block_num * loop_0;
    core_vol_1->at(index) = (prod != 1) ? loop_1 * vol->at(index) : block_num * loop_1;

    if (pattern == "top") {
      core_gap_0->at(index) = (prod != 1) ? _prod(index, outShape) : core_vol_0->at(index);
      core_gap_1->at(index) = (prod != 1) ? _prod(index, outShape) : core_vol_1->at(index);
    } else {
      core_gap_0->at(index) = (prod != 1) ? -_prod(index, outShape) : core_vol_0->at(index);
      core_gap_1->at(index) = (prod != 1) ? -_prod(index, outShape) : core_vol_1->at(index);
    }
  }
}

void padCommon::GetCirculateParams(const std::string& pattern, const int numBit, const int maxCore,
                                   const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape,
                                   const std::vector<std::vector<int64_t>>& padding, PadDTilingParams& params) {
  std::vector<int64_t>* vol;
  std::vector<int64_t>* address;
  std::vector<int64_t>* div_core;
  std::vector<int64_t>* total_core;
  std::vector<int64_t>* core_vol_0;
  std::vector<int64_t>* core_vol_1;
  std::vector<int64_t>* core_gap_0;
  std::vector<int64_t>* core_gap_1;
  int index = 0;
  int pos = 0;
  if (pattern == "top") {
    vol = &params.top_vol;
    address = &params.top_address;
    div_core = &params.top_div_core;
    total_core = &params.top_total_core;
    core_vol_0 = &params.top_core_vol_0;
    core_vol_1 = &params.top_core_vol_1;
    core_gap_0 = &params.top_core_gap_0;
    core_gap_1 = &params.top_core_gap_1;
  } else {
    vol = &params.bottom_vol;
    address = &params.bottom_address;
    div_core = &params.bottom_div_core;
    total_core = &params.bottom_total_core;
    core_vol_0 = &params.bottom_core_vol_0;
    core_vol_1 = &params.bottom_core_vol_1;
    core_gap_0 = &params.bottom_core_gap_0;
    core_gap_1 = &params.bottom_core_gap_1;
    pos = 1;
  }

  while (index < params.depth) {
    // vol, address
    vol->at(index) = _prod(index + 1, outShape) * padding[index][pos];
    address->at(index) =
        (pattern == "top") ? _accumulate(index, vol, 0) : _accumulate(index + 1, vol, -_prod(0, outShape));
    // others
    _calc_core_circulation(params.branch, index, pattern, numBit, maxCore, vol, inShape, outShape, total_core, div_core,
                           core_vol_0, core_vol_1, core_gap_0, core_gap_1);
    index += 1;
  }
}

void padCommon::SplitRL(int64_t& ptrR, int64_t& ptrL, int64_t maxCore, int64_t block, int64_t baseData,
                        int64_t baseCore) {
  // Regulation to split inShape[depth-1]:
  // 1.right * core_data >= 32B;
  // 2.left * virCore ~ maxCore;
  // 3.left * right == inShape[depth-1]
  int64_t total = ptrR;
  int64_t bef_ptrR = ptrR;
  int64_t bef_prtL = ptrL;

  for (int64_t i = 2; i <= total; i++) {
    if (total % i == 0) {
      bef_ptrR = ptrR;
      bef_prtL = ptrL;
      ptrR = total / i;
      ptrL = i;

      if (ptrR * baseData < block) {
        ptrR = bef_ptrR;
        ptrL = bef_prtL;
        break;
      } else {
        if (ptrL * baseCore >= maxCore) {
          break;
        }
      }
    }
  }
}

void padCommon::DupTilingMax(PadDTilingParams& params, int64_t idx) {
  int64_t CirDupVol = 0;
  int64_t MovDupVol = 0;
  int64_t CirPos = 0;
  int64_t MovPos = 0;
  int64_t SortVol = (idx == int64_t(params.recur_inShape.size())) ? 0 : params.prod_recurOut[idx];

  // Max CirDupVol
  for (int64_t i = 0; i <= params.depth - 1; i++) {
    CirDupVol = (params.top_vol[i] >= params.bottom_vol[i]) ? params.top_vol[i] : params.bottom_vol[i];
    if (CirDupVol > 0) {
      CirPos = i;
      GELOGD("CirPos: [%d].", CirPos);
      break;
    }
  }

  // Max MovDupVol
  int64_t begin = (params.depth > 0) ? params.depth - 1 : 0;
  int64_t end = idx;
  for (int64_t i = begin; i < end; i++) {
    MovDupVol = (params.recur_padding[i][0] >= params.recur_padding[i][1]) ? params.recur_padding[i][0]
                                                                           : params.recur_padding[i][1];
    if (MovDupVol > 0) {
      MovDupVol = (i == int64_t(params.recur_inShape.size())) ? MovDupVol : MovDupVol * params.prod_recurOut[i + 1];
      MovPos = i;
      break;
    }
  }

  // Regulation For VecDup
  if (CirDupVol >= MovDupVol) {
    // Circulation deciedes Recursion vec_dup or not.
    // If CirDupVol >= SortVol, means that Recusion dosen't need to vec_dup.
    if (CirDupVol < SortVol) {
      params.recur_dup_mk[idx] = 1;
    }
  } else {
    // Recursion(Mov) decides Recursion(Sort) vec_dup or not.
    // MovDupVol > 0 in the branch, and Mov layer happened before Sort layer.
    params.recur_dup_mk[MovPos] = 1;
  }
}

void padCommon::GetRecurCore(PadDTilingParams& params, const std::vector<int64_t>& inShape,
                             const std::vector<int64_t>& outShape, const std::vector<std::vector<int64_t>>& padding,
                             int maxCore, int numBit, int ubSize) {
  /**
   eg:             inShape is [16, 16, 16, 256]
                                    |
   index = depth - 1               index
   baseCore = prod(shape[0:depth-1)) = 16
   baseData = prod(shape[depth:]) = 16 *256
   depth >= 1 (align)
   */
  using namespace std;
  int64_t depth = params.depth;
  if (numBit == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING("pad_common", "numbit = 0 is not support");
      return;
  }
  int64_t block = MINI_UNIT / numBit;
  int64_t baseCore = accumulate(inShape.begin(), inShape.begin() + depth - 1, 1, std::multiplies<int64_t>());
  int64_t size = inShape.size();
  int64_t baseData = (size != depth) ? _prod(depth, outShape) : block;
  if (depth <= 0){
    depth = 1;
  }
  // Record Status.
  // ptrR*ptrL == inShape[depth-1].
  // prtR*baseData >= block while ptrL is 1.
  int64_t ptrR = (size != depth) ? inShape[depth - 1] : inShape[depth - 1] / block;
  int64_t ptrL = 1;
  SplitRL(ptrR, ptrL, maxCore, block, baseData, baseCore);
  ptrR = (size != depth) ? ptrR : ptrR * block;

  // Return Scalar.
  int64_t virCore = ptrL * baseCore;
  params.total_core = (virCore > maxCore) ? maxCore : virCore;
  params.div_core = (virCore <= maxCore) ? virCore - 1 : (virCore % maxCore == 0) ? maxCore - 1 : virCore % maxCore - 1;
  params.cond = ptrL;
  params.loop_0 = (virCore % maxCore > 0) ? virCore / maxCore + 1 : virCore / maxCore;
  params.loop_1 = (params.div_core + 1 != params.total_core) ? params.loop_0 - 1 : params.loop_0;
  params.address = _accumulate(depth, &params.top_vol, 0);
  params.gap_1 = ptrR * _prod(depth, outShape);
  params.gap_0 = (depth >= 1) ? _prod(depth - 1, outShape) : 0;
  params.in_vol = ptrR * _prod(depth, inShape);

  // Return recur_inShape, recur_padding, recur_outShape.
  // Only split two dims as core
  params.recur_inShape = inShape;
  if (depth - 1 > 0) {
    for (int i = 0; i <= depth-2; i++){
      params.recur_inShape[i] = 1;
    }
  }
  params.recur_inShape[depth - 1] = ptrR;

  params.recur_padding = padding;
  for (int i = 0; i < depth; i++) {
    params.recur_padding[i] = {0, 0};
  }

  params.recur_outShape = outShape;
  if (depth - 1 > 0) {
      for (int i = 0; i <= depth-2; i++){
        params.recur_outShape[i] = 1;
      }
  }
  if (depth - 1 >= 0) {
    params.recur_outShape[depth - 1] = ptrR;
  }

  // Return recurIn, recurOut, model
  int64_t index = 0;
  while (index < size) {
    params.prod_recurIn[index] = _prod(index, params.recur_inShape);
    params.prod_recurOut[index] = _prod(index, params.recur_outShape);
    if ((params.prod_recurIn[index] + params.prod_recurOut[index]) <= ubSize) {
      params.recur_model[index] = 1;
    }
    index += 1;
  }

  // Return recur_gm2buf_mk (One-Time-Triggered OR Not)
  // index record postion of recur_model==[1] firstly.
  for (int64_t i = 0; i < size; i++) {
    if (params.recur_model[i] == 1) {
      params.recur_gm2buf_mk[i] = 1;
      index = i;
      break;
    }
    index = inShape.size();
  }

  // Return recur_dup_mk (One-Time-Triggered)
  // Vec_dup in Circulation layer will decide situation of Vec_dup in Recursion layer.
  // Vec_dup in Rercursion_Mov will effect Vec_dup in Rercursion_Sort.
  DupTilingMax(params, index);
}

void padCommon::GetRecurCorePro(PadDTilingParams& params, const std::vector<int64_t>& inShape,
                                const std::vector<int64_t>& outShape, const std::vector<std::vector<int64_t>>& padding,
                                int maxCore, int numBit, int ubSize) {
  /**
   eg:             inShape is [16, 16, 16, 256]
                                    |
   index = depth - 1               index
   baseCore = prod(shape[0:depth-1)) = 16
   baseData = prod(shape[depth:]) = 16 *256

   Re: diff from GetRecurCore(ALIGN)
   1. [64] -> [96] [[16,16]]: ALIGN split 64 as 4 core after circulation.
      [64] -> [97] [[17,16]]: NOT ALIGN make 64 as "SingleCore" after circulation.
      [63] -> [96] [.......]: NOT ALIGN make 63 as "SingleCore" after circulation.

   2. ALIGN: depth >= 1:
             depth == inShape.size: split inShape[depth-1] as core
      NOT ALIGN: depth >= 0
                 depth == 0: SingleCore.
                 depth == inShape.size: don't split last_dim for core
   */
  using namespace std;
  int64_t depth = params.depth;
  if (numBit == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("pad_common", "num_bit = 0 is not support");
    return;
  }
  int64_t block = MINI_UNIT / numBit;
  int64_t baseCore = 0;
  int64_t baseData = 0;
  int64_t virCore = 0;
  int64_t size = inShape.size();
  if (depth < 0){
    depth = 1;
  }
  if (depth == 0) {
    // not work in circulation.
    // Return Scalar.
    params.total_core = 1;
    params.div_core = 0;
    params.cond = 1;
    params.loop_0 = 1;
    params.loop_1 = 1;
    params.address = 0;
    params.gap_1 = 0;
    params.gap_0 = 0;
    params.in_vol = _prod(0, inShape);

    // Return recur_inShape, recur_padding, recur_outShape.
    // Only split two dims as core
    params.recur_inShape = inShape;
    params.recur_padding = padding;
    params.recur_outShape = outShape;
  } else {
    // A B C D
    // eg: depth = 3 size = 4
    baseCore = accumulate(inShape.begin(), inShape.begin() + depth - 1, 1, std::multiplies<int64_t>());
    baseData = _prod(depth, outShape);

    int64_t ptrR = inShape[depth - 1];
    int64_t ptrL = 1;
    if (depth < size) {
      SplitRL(ptrR, ptrL, maxCore, block, baseData, baseCore);
    }

    // Return Scalar.
    virCore = ptrL * baseCore;
    params.total_core = (virCore > maxCore) ? maxCore : virCore;
    params.div_core =
        (virCore <= maxCore) ? virCore - 1 : (virCore % maxCore == 0) ? maxCore - 1 : virCore % maxCore - 1;
    params.cond = ptrL;
    params.loop_0 = (virCore % maxCore > 0) ? virCore / maxCore + 1 : virCore / maxCore;
    params.loop_1 = (params.div_core + 1 != params.total_core) ? params.loop_0 - 1 : params.loop_0;
    params.address = _accumulate(depth, &params.top_vol, 0);
    params.gap_1 = ptrR * _prod(depth, outShape);
    params.gap_0 = _prod(depth - 1, outShape);
    params.in_vol = ptrR * _prod(depth, inShape);

    // Return recur_inShape, recur_padding, recur_outShape.
    // Only split two dims as core
    params.recur_inShape = inShape;
    if (depth - 1 > 0) {
      for (int i = 0; i <= depth-2; i++){
        params.recur_inShape[i] = 1;
      }
    }
    params.recur_inShape[depth - 1] = ptrR;

    params.recur_padding = padding;
    for (int i = 0; i < depth; i++) {
      params.recur_padding[i] = {0, 0};
    }

    params.recur_outShape = outShape;
    if (depth - 1 > 0) {
      for (int i = 0; i <= depth-2; i++){
        params.recur_outShape[i] = 1;
      }
    }
    params.recur_outShape[depth - 1] = ptrR;
  }

  // Return recurIn, recurOut, model
  // Not Align requires input data and sort data are align in UB.
  // The last_dim_limit decides sort or not in model of "Not Align".
  int last_dim_limit = 64;
  uint64_t length = inShape.size() - 1;
  int64_t index = 0;
  int64_t align_in = 0;
  int64_t align_out = 0;
  while (index < size) {
    params.prod_recurIn[index] = _prod(index, params.recur_inShape);
    params.prod_recurOut[index] = _prod(index, params.recur_outShape);

    if (inShape[length] <= last_dim_limit) {
      align_in = _align(params.prod_recurIn[index], block);
      align_out = _align(params.prod_recurOut[index], block);
      if (align_in + align_out <= ubSize) {
        params.recur_model[index] = 1;
      }
    }
    index += 1;
  }

  // Return recur_gm2buf_mk (One-Time-Triggered OR Not)
  // index record position of recur_model==[1] firstly.
  if (inShape[length] <= last_dim_limit) {
    for (int64_t i = 0; i < size; i++) {
      if (params.recur_model[i] == 1) {
        params.recur_gm2buf_mk[i] = 1;
        index = i;
        break;
      }
      index = inShape.size();
    }
  }
  // Return recur_dup_mk (One-Time-Triggered)
  // Vec_dup in Circulation layer will decide situation of Vec_dup in Recursion layer.
  // Vec_dup in Recursion_Mov will effect Vec_dup in Recursion_Sort.
  // recur_dup_mk only effect Recursion_Sort.
  if (inShape[length] <= last_dim_limit) {
    DupTilingMax(params, index);
  }
}

void padCommon::SetVectorParams(std::vector<std::vector<int64_t>>& vector_params, OpRunInfo& runInfo) {
  int num0 = vector_params.size();
  int num1 = vector_params[0].size();

  for (int i = 0; i < num0; i++) {
    for (int j = 0; j < num1; j++) {
      ByteBufferPut(runInfo.tiling_data, int64_t(vector_params[i][j]));
    }
  }
}

void padCommon::PrintRunningParams(const PadDTilingParams& params) {
  using namespace std;

  GELOGI("branch: [%d].", params.branch);
  GELOGI("depth: [%d].", params.depth);
  GELOGI("total_core: [%d]  .", params.total_core);
  GELOGI("div_core: [%d] .", params.div_core);
  GELOGI("in_vol: [%d] .", params.in_vol);
  GELOGI("loop_0: [%d] .", params.loop_0);
  GELOGI("loop_1: [%d] .", params.loop_1);
  GELOGI("gap_0: [%d] .", params.gap_0);
  GELOGI("gap_1: [%d] .", params.gap_1);
  GELOGI("cond: [%d] .", params.cond);
  GELOGI("address: [%d] .", params.address);

  vector< vector<int64_t> > vector_params = {
      params.top_vol,           params.top_address,       params.top_div_core,      params.top_total_core,
      params.top_core_vol_0,    params.top_core_vol_1,    params.top_core_gap_0,    params.top_core_gap_1,
      params.bottom_vol,        params.bottom_address,    params.bottom_div_core,   params.bottom_total_core,
      params.bottom_core_vol_0, params.bottom_core_vol_1, params.bottom_core_gap_0, params.bottom_core_gap_1,
      params.recur_model,       params.recur_dup_mk,      params.recur_gm2buf_mk,   params.prod_recurOut,
      params.prod_recurIn,      params.recur_inShape,     params.recur_outShape};

  vector<string> vector_name = {"top_vol",           "top_address",       "top_div_core",      "top_total_core",
                                "top_core_vol_0",    "top_core_vol_1",    "top_core_gap_0",    "top_core_gap_1",
                                "bottom_vol",        "bottom_address",    "bottom_div_core",   "bottom_total_core",
                                "bottom_core_vol_0", "bottom_core_vol_1", "bottom_core_gap_0", "bottom_core_gap_1",
                                "recur_model",       "recur_dup_mk",      "recur_gm2buf_mk",   "prod_recurOut",
                                "prod_recurIn",      "recur_inShape",     "recur_outShape"};

  int num0 = vector_params.size();
  int num1 = vector_params[0].size();

  string vec_str;
  for (int i = 0; i < num0; i++) {
    vec_str = "";
    for (int j = 0; j < num1; j++) {
      vec_str += std::to_string(vector_params[i][j]);
      vec_str += ",";
    }
    GELOGI("[%s]: [%s].", vector_name[i].c_str(), vec_str.c_str());
  }

  // recur_padding
  vec_str = "";
  for (int i = 0; i < int64_t(params.recur_padding.size()); i++) {
    vec_str += std::to_string(params.recur_padding[i][0]);
    vec_str += ",";
  }
  GELOGI("[recur_padding_top]: [%s].", vec_str.c_str());

  vec_str = "";
  for (int i = 0; i < int64_t(params.recur_padding.size()); i++) {
    vec_str += std::to_string(params.recur_padding[i][1]);
    vec_str += ",";
  }
  GELOGI("[recur_padding_bottom]: [%s].", vec_str.c_str());
}

void padCommon::SetRunningParams(const PadDTilingParams& params, OpRunInfo& runInfo) {
  using namespace std;
  // scalar_params
  // 256
  ByteBufferPut(runInfo.tiling_data, int64_t(params.branch));
  ByteBufferPut(runInfo.tiling_data, int64_t(params.depth));
  ByteBufferPut(runInfo.tiling_data, int64_t(params.total_core));
  ByteBufferPut(runInfo.tiling_data, int64_t(params.div_core));
  ByteBufferPut(runInfo.tiling_data, int64_t(params.in_vol));
  ByteBufferPut(runInfo.tiling_data, int64_t(params.loop_0));
  ByteBufferPut(runInfo.tiling_data, int64_t(params.loop_1));
  ByteBufferPut(runInfo.tiling_data, int64_t(params.gap_0));
  ByteBufferPut(runInfo.tiling_data, int64_t(params.gap_1));
  ByteBufferPut(runInfo.tiling_data, int64_t(params.cond));
  ByteBufferPut(runInfo.tiling_data, int64_t(params.address));

  // vector_params(recur_padding special)
  vector< vector<int64_t> > vector_params = {
      params.top_vol,           params.top_address,       params.top_div_core,      params.top_total_core,
      params.top_core_vol_0,    params.top_core_vol_1,    params.top_core_gap_0,    params.top_core_gap_1,
      params.bottom_vol,        params.bottom_address,    params.bottom_div_core,   params.bottom_total_core,
      params.bottom_core_vol_0, params.bottom_core_vol_1, params.bottom_core_gap_0, params.bottom_core_gap_1,
      params.recur_model,       params.recur_dup_mk,      params.recur_gm2buf_mk,   params.prod_recurOut,
      params.prod_recurIn,      params.recur_inShape,     params.recur_outShape};

  SetVectorParams(vector_params, runInfo);

  // recur_padding
  for (int i = 0; i < int64_t(params.recur_padding.size()); i++) {
    ByteBufferPut(runInfo.tiling_data, int64_t(params.recur_padding[i][0]));
  }

  for (int i = 0; i < int64_t(params.recur_padding.size()); i++) {
    ByteBufferPut(runInfo.tiling_data, int64_t(params.recur_padding[i][1]));
  }
}

bool padCommon::CheckTensor(const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape) {
  if (inShape.size() != outShape.size()) {
    VECTOR_INNER_ERR_REPORT_TILIING("PadDTiling", "CheckTensor Failed.");
    return false;
  }
  return true;
}
}  // namespace optiling
