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
 * \file transdata_dsl_general.h
 * \brief dynamic transdata_dsl_general op tiling
 */

#ifndef TRANSDATA_DSL_GENERAL_H
#define TRANSDATA_DSL_GENERAL_H

#include "transdata_dsl.h"

namespace optiling {
constexpr float GENERAL_THRESHOLD = 0.95;

struct GeneralTilingInfo {
  int64_t blk_dim;
  int64_t blk_factor;
  int64_t ub_0_factor;
  int64_t ub_1_factor;
  int64_t mte2_burst_len;
  int64_t mte3_burst_len;
  int64_t core;
  bool split_once;
  float percent;
  size_t blk_idx;
  size_t ub_0_idx;
  size_t ub_1_idx;

  void Reset() {
    blk_dim = 0;
    blk_factor = 0;
    ub_0_factor = 0;
    ub_1_factor = 0;
    mte2_burst_len = 0;
    mte3_burst_len = 0;
    core = 0;
    split_once = false;
    percent = 0;
    blk_idx = 0;
    ub_0_idx = 0;
    ub_1_idx = 0;
  }

  void UBInfoSet(size_t ptrA, size_t ptrB, int64_t factorA, int64_t factorB, int64_t mte2_rate, int64_t mte3_rate,
                 float ub_percent, int64_t possible_core, bool once_split) {
    ub_0_idx = ptrA;
    ub_1_idx = ptrB;
    ub_0_factor = factorA;
    ub_1_factor = factorB;
    mte2_burst_len = mte2_rate;
    mte3_burst_len = mte3_rate;
    percent = ub_percent;
    core = possible_core;
    split_once = once_split;
  }

  GeneralTilingInfo() {
    Reset();
  }
};

struct GeneralSplit {
  size_t ptrA;
  size_t ptrB;

  void Reset() {
    ptrA = 0;
    ptrB = 0;
  }

  void Set(size_t a, size_t b) {
    ptrA = a;
    ptrB = b;
  }

  GeneralSplit() {
    Reset();
  }
};

class TransdataGeneral {
 public:
  explicit TransdataGeneral(const std::string& _op_type, const CompileInfoTransdataDSL& _compileInfo, utils::OpRunInfo& _run_info,
                            const Shape& _input, const Shape& _output, const Shape& _reshape)
      : op_type(_op_type),
        compileInfo(_compileInfo),
        run_info(_run_info),
        input(_input),
        output(_output),
        reshape(_reshape) {
  }
  ~TransdataGeneral() {
  }
  bool DoTiling();

 private:
  const std::string& op_type;
  const CompileInfoTransdataDSL& compileInfo;
  utils::OpRunInfo& run_info;
  const Shape& input;
  const Shape& output;
  const Shape& reshape;
  GeneralTilingInfo tilingInfo;

  bool is_last_transpose{false};
  bool is_data_move{true};
  bool split_once{false};
  float percent{0};

  size_t c1_index{0};
  size_t c0_index{0};
  size_t computeType{BASE_SCH};
  size_t shapeType{COMMON_ALIGN};
  // reshape mapping output
  size_t r_mapping_o[MAX_DIM] = {0};

  Shape tiling_input;
  Shape tiling_output;

  size_t array_size{0};
  GeneralSplit split_array[MAX_DIM];

  MTEInfo mte2;
  MTEInfo mte3;

  size_t ptrI{0};
  size_t ptrO{0};
  int64_t UBSize{0};
  int64_t factorI{0};
  int64_t factorO{0};
  int64_t core{0};
  int64_t num_in_ub{0};
  int64_t ele_byte{0};
  int64_t mte_rate{0};
  int64_t total_num{0};

 private:
  bool CalcTiling();
  bool WriteTilingData();

  bool Init();
  bool IsConstRuntime();
  bool Strategy();
  bool UBTiling();
  bool Filter();
  bool UBTilingProcess();
  bool BlockTiling();
  void BlkTilingProcess(const AxisType* type_array, const Shape& res);

 private:
  int64_t CommonAlignLimit(int64_t factor) const;
  int64_t CalcTilingKey();
  bool ChooseType(int64_t dim_len, int64_t ub_size) const;
  bool CheckValidSplit(size_t ptrA, size_t ptrB) const;
  bool OnceTiling();
  bool TwiceTiling();
  bool InitUBFactorMTE2(int64_t lower, int64_t higher);
  bool InitUBFactorMTE3(int64_t lower, int64_t higher);

  void SetStorageAlign(Shape& input_shape, Shape& output_shape, const Shape& ori_input) const;
  void CompareTiling();
  void AdjustUBFactor();
  void AdjustUBFactorMTE2(int64_t lower, int64_t higher);
  void AdjustUBFactorMTE3(int64_t lower, int64_t higher);
  void GetOutputRealTail(int64_t ptr, int64_t factor, MTEInfo& mte);
  void DiscriminationAxisType(AxisType* type_array, size_t length);
};
}  // namespace optiling

#endif  // TRANSDATA_DSL_GENERAL_H