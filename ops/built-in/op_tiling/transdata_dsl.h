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
 * \file trans_data_dsl.h
 * \brief dynamic trans_data_dsl op tiling
 */

#ifndef TRANSDATA_DSL_H
#define TRANSDATA_DSL_H

#include <cmath>
#include <vector>
#include <string>

#include <nlohmann/json.hpp>
#include "vector_tiling.h"

namespace optiling {

static const int64_t OFFSET_2 = 2;
static const int64_t STRIDE_16 = 16;
static const int64_t STRIDE_2 = 2;
static const int64_t AXIS_OUT_UB = 0;
static const int64_t AXIS_IN_UB = 1;
static const int64_t AXIS_IN_FIRST_UB_SPLIT = 2;
static const int64_t AXIS_IN_SECOND_UB_SPLIT = 3;
static const float THRESHOLD = 0.95;

static const int64_t FP32_TRANSPOSE_LIMIT = 128;
static const int64_t MAX_DIM_NUM = 8;
static const int64_t BLOCK = 32;
static const int64_t FP16 = 2;
static const int64_t INT8 = 1;
static const int64_t FP32 = 4;
static const int64_t DEFAULT_VALUE = -1;
static const int64_t BaseSch = 0;
static const int64_t BorrowNSch = 1;
static const int64_t BorrowHSch = 2;
static const int64_t StorageAlignBranch = 0;
static const int64_t CommonAlignBranch = 1;
static const int64_t PACKET_SENDING_RATE = 256;

static const int64_t CONST_KEY = 123;
static const int64_t FORWARD_KEY = 2;
static const int64_t BACKWARD_KEY = 3;
static const size_t KEY_NUM = 7;

struct CompileInfoTransdataDSL {
  // construct func
  CompileInfoTransdataDSL() = default;
  CompileInfoTransdataDSL(const std::string& op_type, const nlohmann::json& compile_info);
  // check value
  bool Check() const;
  bool check_success{false};
  bool is_const_compile{false};
  bool is_const{false};
  bool is_forward{false};
  int64_t align_size{DEFAULT_VALUE};
  int64_t pad_align_size{DEFAULT_VALUE};
  int64_t core_num{DEFAULT_VALUE};
  std::vector<int64_t> src_pad;
  std::vector<int64_t> src_fuse;
  std::vector<int64_t> permute;
  std::vector<int64_t> unknown_dims;
  std::vector<std::vector<int64_t>> ub_info;
  std::unordered_map<std::string, int32_t> const_block_dims;
};

struct TilingInfoTransdataDSL {
  int64_t blk_dim{DEFAULT_VALUE};
  int64_t blk_idx{DEFAULT_VALUE};
  int64_t blk_factor{DEFAULT_VALUE};
  int64_t ub_0_idx{DEFAULT_VALUE};
  int64_t ub_0_factor{DEFAULT_VALUE};
  int64_t ub_1_idx{DEFAULT_VALUE};
  int64_t ub_1_factor{DEFAULT_VALUE};
  int64_t mte2_burst_len{DEFAULT_VALUE};
  int64_t mte3_burst_len{DEFAULT_VALUE};
  int64_t core{DEFAULT_VALUE};
  bool split_once{false};
  float percent{0};
};

struct TransdataDSLMTEInfo {
  int64_t virLen{DEFAULT_VALUE};
  int64_t mainLen{DEFAULT_VALUE};
  int64_t tailLen{DEFAULT_VALUE};
};

class TransdataBase {
 public:
  explicit TransdataBase(const std::string& _op_type, const ge::Operator& _op_paras,
                         const CompileInfoTransdataDSL& _compileInfo, utils::OpRunInfo& _run_info)
      : op_type(_op_type), op_paras(_op_paras), compileInfo(_compileInfo), run_info(_run_info) {
  }
  ~TransdataBase() {
  }
  bool DoTiling();

 private:
  bool CalcTiling();
  bool WriteTilingData();

  bool UpdateValue();
  bool IsConstRuntime();
  bool GetInputOutput();
  bool ChooseStrategy();
  bool BaseUBTiling();
  bool BaseBlockTiling();

  bool InferInput();
  bool InferOutput();
  bool DoFusing(int64_t* input, int64_t* output, size_t ori_length);

  bool UBTilingFilter(int64_t* input, int64_t* output, int64_t array_length);
  bool UBTilingForwardProcess(int64_t* input, int64_t* output, size_t array_length);
  bool UBTilingBackwardProcess(int64_t* input, int64_t* output, size_t array_length);
  void UBTilingUpdate(int64_t ptrA, int64_t ptrB, int64_t factorA, int64_t factorB, int64_t mte2, int64_t mte3,
                      float percent, int64_t core);

  void FindAxisInUB(int64_t* axis_in_ub, int64_t length);
  void ForwardBlockProcess(const int64_t* axis_in_ub);
  void BackwardBlockProcess(const int64_t* axis_in_ub);

 private:
  int64_t SetAlign(int64_t value, int64_t align_factor) const;
  int64_t Prod(const int64_t* input, int64_t ptr, int64_t length) const;
  int32_t CalcTilingKey();
  int64_t LimitMap(int64_t factor) const;
  bool CommonAlignLimit(int64_t dim_len, int64_t ub_size);
  void StorageAlign(int64_t* new_input, int64_t* new_out, int64_t* input);
  bool CheckValidSplit(const int64_t* input, const int64_t* output, int64_t ptrA, int64_t ptrB);
  void GetOutPutRealTail(int64_t ptr, int64_t factor, TransdataDSLMTEInfo& mte);

 private:
  const std::string& op_type;
  const ge::Operator& op_paras;
  const CompileInfoTransdataDSL& compileInfo;
  utils::OpRunInfo& run_info;

  int64_t input_shape[MAX_DIM_NUM] = {0};
  int64_t output_shape[MAX_DIM_NUM] = {0};
  int64_t reshape[MAX_DIM_NUM] = {0};
  int64_t reshape_mapping_output[MAX_DIM_NUM] = {0};
  int64_t possible_ub_tiling[MAX_DIM_NUM * MAX_DIM_NUM] = {0};

  size_t size_input{0};
  size_t size_output{0};
  size_t size_reshape{0};
  size_t computeType{0};
  size_t shapeType{0};

  int64_t c0_idx{DEFAULT_VALUE};
  int64_t c1_idx{DEFAULT_VALUE};
  int64_t UBSize{DEFAULT_VALUE};
  int64_t ub_tiling_num{DEFAULT_VALUE};

  bool is_last_transpose{false};
  bool isDataMove{true};

  TilingInfoTransdataDSL tilingInfo;
  TransdataDSLMTEInfo mte2;
  TransdataDSLMTEInfo mte3;
};

class TransdataTilingHandler : public AutoTilingHandler {
 public:
  TransdataTilingHandler(const std::string& o, const std::string& p, const nlohmann::json& c)
      : AutoTilingHandler(o, p), compileInfo(o, c) {
  }
  ~TransdataTilingHandler() = default;
  bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const override;
  bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info, const OpInfo& op_info) const override;
  bool ParsedSuccess() {
    return compileInfo.check_success;
  };

 private:
  const CompileInfoTransdataDSL compileInfo;
};

std::shared_ptr<AutoTilingHandler> CreateTransdataTilingHandler(const std::string& op_type, const std::string& pattern,
                                                                const nlohmann::json& parsed_compile_info);
}  // namespace optiling

#endif  // TRANSDATA_DSL_H