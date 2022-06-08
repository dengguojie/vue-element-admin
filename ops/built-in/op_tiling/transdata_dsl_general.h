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
constexpr size_t SINGLE_CORE_MODE = 1;
constexpr size_t MULTI_CORE_MODE = 2;

struct GeneralTilingInfo {
  int64_t blk_dim;
  int64_t blk_factor;
  int64_t ub_0_factor;
  int64_t ub_1_factor;
  int64_t mte2_burst_len;
  int64_t mte3_burst_len;
  size_t blk_idx;
  size_t ub_0_idx;
  size_t ub_1_idx;
  size_t core_mode;
  bool split_once;
  float percent;

  void Reset() {
    blk_dim = 0;
    blk_factor = 0;
    ub_0_factor = 0;
    ub_1_factor = 0;
    mte2_burst_len = 0;
    mte3_burst_len = 0;
    blk_idx = 0;
    ub_0_idx = 0;
    ub_1_idx = 0;
    core_mode = SINGLE_CORE_MODE;
    split_once = false;
    percent = 0;
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
                            const Shape& _input, const Shape& _output, const Shape& _reshape, size_t _transpose_work)
      : op_type(_op_type),
        compileInfo(_compileInfo),
        run_info(_run_info),
        input(_input),
        output(_output),
        reshape(_reshape),
        transposeWork(_transpose_work) {
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
  size_t transposeWork;
  size_t computeType{BASE_SCH};
  size_t shapeType{COMMON_ALIGN};
  // avoidBCWork: avoid bank conflict is 1 else 0.
  size_t avoidBCWork{0};
  GeneralTilingInfo tilingInfo;

  bool is_last_transpose{false};
  bool is_data_move{true};
  bool split_once{false};
  bool avoid_bc{false};
  float percent{0};

  size_t c1_index{0};
  size_t c0_index{0};
  // reshape mapping output
  size_t r_mapping_o[MAX_DIM] = {0};

  Shape tiling_input;
  Shape tiling_output;
  Shape bound_input;
  Shape bound_output;

  size_t array_size{0};
  GeneralSplit split_array[MAX_DIM];

  MTEInfo mte2;
  MTEInfo mte3;

  size_t ptrI{0};
  size_t ptrO{0};
  // ptrX split axes that attend transpose.
  bool ptrISplitT{false};
  bool ptrOSplitT{false};
  int64_t UBSize{0};
  int64_t factorI{0};
  int64_t factorO{0};
  int64_t core{0};
  int64_t tiling_in_ub{0};
  int64_t bound_in_ub{0};
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
  bool MultiBlkTiling(const AxisType* type_array, const Shape& res);
  bool SingleBlkTiling(const AxisType* type_array, const Shape& res);

 private:
  bool CommonAlignLimit(int64_t ub_size, int64_t last_dim_align_value, int64_t num_bit) const;
  bool ChooseType(int64_t dim_len, int64_t ub_size) const;
  bool CheckValidSplit(size_t ptrA, size_t ptrB) const;
  bool OnceTiling();
  bool TwiceTiling();
  bool InitUBFactorMTE2(int64_t lower, int64_t higher);
  bool InitUBFactorMTE3(int64_t lower, int64_t higher);
  void UBPreProcess(int64_t loop, size_t a, size_t b);

  void UBInfoSet(size_t core_mode);
  void SetStorageAlign(const Shape& input_shape, size_t length);
  void AlignVNC(Shape& input_shape) const;
  void VectorOptimization();
  void VOROptimize(int64_t &a, int64_t &b, int64_t extent, int64_t stride) const;
  void UpdateCore();
  void CompareTiling(bool is_multi_core_mode);
  void AdjustUBFactor();
  void AvoidBankConflictIsWork();
  void AdjustUBFactorMTE2(int64_t lower, int64_t higher);
  void AdjustUBFactorMTE3(int64_t lower, int64_t higher);
  void GetOutputRealTail(int64_t ptr, int64_t factor, MTEInfo& mte);
  void DiscriminationAxisType(AxisType* type_array, size_t length);
  bool AvoidBankConflict();
  bool AvoidVNCHWCONV();
  bool AvoidVOR();
  bool AvoidBCOfVNCHWCONV(int64_t a, int64_t b, int64_t c0, size_t idxA, size_t idxB);
  bool AvoidBCOfVOR(int64_t a, int64_t b, int64_t c0, size_t idxA, size_t idxB);
  int64_t CalcTilingKey();
  int64_t CommonRefineFactor(int64_t ori_factor, size_t ptr);
  int64_t AxisValueInUB(size_t ptr) const;
};
}  // namespace optiling

#endif  // TRANSDATA_DSL_GENERAL_H