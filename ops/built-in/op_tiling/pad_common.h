/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file pad_common.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_PAD_COMMON_H_
#define OPS_BUILT_IN_OP_TILING_PAD_COMMON_H_

#include <vector>
#include <string>

#include <nlohmann/json.hpp>
#include "graph/debug/ge_log.h"
#include "op_tiling.h"

#include "op_log.h"

namespace optiling {

struct PadDTilingParams {
  // Circulation
  int branch = 0;
  int64_t depth = 0;
  std::vector<int64_t> top_vol;
  std::vector<int64_t> top_address;
  std::vector<int64_t> top_div_core;
  std::vector<int64_t> top_total_core;
  std::vector<int64_t> top_core_vol_0;
  std::vector<int64_t> top_core_vol_1;
  std::vector<int64_t> top_core_gap_0;
  std::vector<int64_t> top_core_gap_1;

  std::vector<int64_t> bottom_vol;
  std::vector<int64_t> bottom_address;
  std::vector<int64_t> bottom_div_core;
  std::vector<int64_t> bottom_total_core;
  std::vector<int64_t> bottom_core_vol_0;
  std::vector<int64_t> bottom_core_vol_1;
  std::vector<int64_t> bottom_core_gap_0;
  std::vector<int64_t> bottom_core_gap_1;

  // Recursion
  int64_t total_core = 0;
  int64_t div_core = 0;
  int64_t in_vol = 0;
  int64_t loop_0 = 0;
  int64_t loop_1 = 0;
  int64_t gap_0 = 0;
  int64_t gap_1 = 0;
  int64_t cond = 0;
  int64_t address = 0;

  std::vector<int64_t> recur_inShape;
  std::vector<int64_t> recur_outShape;
  std::vector<std::vector<int64_t>> recur_padding;
  std::vector<int64_t> recur_model;
  std::vector<int64_t> recur_dup_mk;
  std::vector<int64_t> recur_gm2buf_mk;
  std::vector<int64_t> prod_recurIn;
  std::vector<int64_t> prod_recurOut;
};

class padCommon {
 public:
  int _numBit(const std::string& dtype);

  int64_t _prod(int64_t index, const std::vector<int64_t>& shape);

  int64_t _accumulate(int64_t index, const std::vector<int64_t>* shape_ptr, int64_t init);

  int64_t _align(int64_t value, int64_t align_vol);

  void InitTilingParams(PadDTilingParams& params, int num);

  int CheckBranch(const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape,
                  const std::vector<std::vector<int64_t>>& padding, int numBit, int model);

  void fused(std::vector<int64_t>& out, std::vector<int64_t> in, int64_t distance, int64_t begin,
             std::vector<std::vector<int64_t>>& out_pad, std::vector<std::vector<int64_t>>& in_pad, bool mark);

  void fused_special(std::vector<int64_t>& out, std::vector<int64_t> in, int64_t head_ptr,
                     std::vector<std::vector<int64_t>>& out_pad, std::vector<std::vector<int64_t>> in_pad, bool mark);

  bool FusedAxis(std::vector<int64_t>& n_inShape, std::vector<int64_t>& n_outShape,
                 std::vector<std::vector<int64_t>>& n_padding, const std::vector<int64_t>& inShape,
                 const std::vector<int64_t>& outShape, std::vector<std::vector<int64_t>> padding);

  void GetDepth(const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape,
                const std::vector<std::vector<int64_t>>& padding, int64_t& depth, int coreNum, int numBit, int branch);

  void _calc_core_circulation(int branch, int index, const std::string& pattern, int numBit, int maxCore,
                              const std::vector<int64_t>* vol, const std::vector<int64_t>& inShape,
                              const std::vector<int64_t>& outShape, std::vector<int64_t>* total_core,
                              std::vector<int64_t>* div_core, std::vector<int64_t>* core_vol_0,
                              std::vector<int64_t>* core_vol_1, std::vector<int64_t>* core_gap_0,
                              std::vector<int64_t>* core_gap_1);

  void GetCirculateParams(const std::string& pattern, int numBit, int maxCore, const std::vector<int64_t>& inShape,
                          const std::vector<int64_t>& outShape, const std::vector<std::vector<int64_t>>& padding,
                          PadDTilingParams& params);

  void SplitRL(int64_t& ptrR, int64_t& ptrL, int64_t maxCore, int64_t block, int64_t baseData, int64_t baseCore);

  void DupTilingMax(PadDTilingParams& params, int64_t idx);

  void GetRecurCore(PadDTilingParams& params, const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape,
                    const std::vector<std::vector<int64_t>>& padding, int maxCore, int numBit, int ubSize);

  void GetRecurCorePro(PadDTilingParams& params, const std::vector<int64_t>& inShape,
                       const std::vector<int64_t>& outShape, const std::vector<std::vector<int64_t>>& padding,
                       int maxCore, int numBit, int ubSize);

  void SetVectorParams(std::vector<std::vector<int64_t>>& vector_params, OpRunInfo& runInfo);

  void PrintRunningParams(const PadDTilingParams& params);

  void SetRunningParams(const PadDTilingParams& params, OpRunInfo& runInfo);

  bool CheckTensor(const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape);
};
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_PAD_COMMON_H_
