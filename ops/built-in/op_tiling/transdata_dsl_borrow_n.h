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
 * \file transdata_dsl_borrow_n.h
 * \brief dynamic transdata_dsl_borrow_n op tiling
 */

#ifndef TRANSDATA_DSL_BORROW_N_H
#define TRANSDATA_DSL_BORROW_N_H

#include "transdata_dsl.h"

namespace optiling {
struct TDBNTilingInfo {
  int64_t blk_dim;
  int64_t blk_factor;
  int64_t ub_0_factor;
  int64_t ub_1_factor;
  int64_t core;
  size_t blk_idx;
  size_t ub_0_idx;
  size_t ub_1_idx;

  void Reset() {
    blk_dim = 0;
    blk_factor = 0;
    ub_0_factor = 0;
    ub_1_factor = 0;
    core = 0;
    blk_idx = 0;
    ub_0_idx = 0;
    ub_1_idx = 0;
  }

  void UBInfoSet(size_t ptrA, size_t ptrB, int64_t factorA, int64_t factorB, int64_t possible_core) {
    ub_0_idx = ptrA;
    ub_1_idx = ptrB;
    ub_0_factor = factorA;
    ub_1_factor = factorB;
    core = possible_core;
  }

  TDBNTilingInfo() {
    Reset();
  }
};

struct TDBNSplit {
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

  TDBNSplit() {
    Reset();
  }
};

class TransdataBN {
 public:
  explicit TransdataBN(const std::string& _op_type, const CompileInfoTransdataDSL& _compileInfo, utils::OpRunInfo& _run_info,
                       Shape& _input, Shape& _output)
      : op_type(_op_type),
        compileInfo(_compileInfo),
        run_info(_run_info),
        input(_input),
        output(_output){
  }
  ~TransdataBN() {
  }
  bool DoTiling();

 private:
  const std::string& op_type;
  const CompileInfoTransdataDSL& compileInfo;
  utils::OpRunInfo& run_info;
  Shape& input;
  Shape& output;

  Shape tiling_input;
  Shape tiling_output;
  size_t c_index{0};
  size_t c1_index{0};
  size_t c0_index{0};
  std::vector<size_t> permute{std::vector<size_t>(MAX_DIM, 0)};
  size_t computeType{BORROW_N_SCH};
  size_t shapeType{STORAGE_ALIGN};
  size_t array_size{0};
  TDBNSplit split_array[MAX_DIM];

  bool has_dim_n{true};
  bool is_reinterpret{false};
  size_t ptrI{0};
  size_t ptrO{0};
  int64_t UBSize{0};
  int64_t factorI{0};
  int64_t factorO{0};
  int64_t core{0};
  int64_t num_in_ub{0};
  int64_t align_size{0};
  int64_t ele_byte{0};
  int64_t mte_rate{0};
  MTEInfo mte3;
  TDBNTilingInfo tilingInfo;

 private:
  bool CalcTiling();
  bool WriteTilingData();
  bool InitBackward();
  bool InitForward();
  bool IsConstRuntime();
  bool Strategy();
  bool UBTiling();
  bool Filter();
  bool UBTilingProcess();
  bool BlockTiling();
  void BlkTilingProcess(const AxisType* type_array, const Shape& res);

 private:
  bool InferTilingInput();
  bool OnceTiling();
  void CompareTiling();
  void AdjustUBFactorForward();
  void AdjustUBFactorBackward();
  void GetOutputRealTail(int64_t ptr, int64_t factor, MTEInfo& mte);
  void DiscriminationAxisType(AxisType* type_array, size_t length);
  int64_t CalcTilingKey();
};
}  // namespace optiling

#endif  // TRANSDATA_DSL_BORROW_N_H