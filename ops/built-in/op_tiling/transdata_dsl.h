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

#include "error_log.h"
#include "vector_tiling.h"
#include "vector_tiling_log.h"
#include "graph/utils/op_desc_utils.h"

namespace optiling {
constexpr size_t OFFSET_2 = 2;
constexpr size_t STRIDE_2 = 2;
constexpr size_t STRIDE_3 = 3;
constexpr size_t STRIDE_16 = 16;
constexpr size_t STRIDE_64 = 64;
constexpr size_t STRIDE_128 = 128;
constexpr size_t STRIDE_160 = 160;
constexpr size_t STRIDE_168 = 168;
constexpr size_t STRIDE_256 = 256;
constexpr size_t STRIDE_1024 = 1024;

constexpr int64_t MAX_DIM = 8;
constexpr int64_t DEFAULT = -1;

constexpr int64_t BLOCK = 32;
constexpr int64_t INT8_BYTE = 1;
constexpr int64_t FP16_BYTE = 2;
constexpr int64_t FP32_BYTE = 4;
constexpr int64_t FP32_TRANSPOSE_LIMIT = 128;
constexpr int64_t PACKET_SENDING_RATE = 256;

constexpr size_t BASE_SCH = 0;
constexpr size_t BORROW_N_SCH = 1;
constexpr size_t BORROW_H_SCH = 2;
constexpr size_t EXISTED_C1C0 = 2;
constexpr int64_t STORAGE_ALIGN = 0;
constexpr int64_t COMMON_ALIGN = 1;

constexpr int64_t CONST_KEY = 123;
constexpr int64_t FORWARD_KEY = 2;
constexpr int64_t BACKWARD_KEY = 3;

enum AxisType {
  UB_EXTERNAL = 0,
  UB_INTERNAL = 1,
  UB_FIRST_SPLIT = 2,
  UB_SECOND_SPLIT = 3,
};

struct Shape {
  int64_t shape[MAX_DIM];
  size_t size;

  // Insert
  void Insert(size_t index, int64_t value) {
    if (size >= MAX_DIM || index > size || index >= MAX_DIM) {
      return;
    }

    int64_t tmp = 0;
    for (size_t i = index; i < size; i++) {
      tmp = shape[i];
      shape[i] = value;
      value = tmp;
    }
    shape[size] = value;
    size++;
  }

  // Reset
  void Reset() {
    size = MAX_DIM;
    for (size_t i = 0; i < size; i++) {
      shape[i] = DEFAULT;
    }
  }

  // Set Attr
  void SetSize(size_t num) {
    size = num;
  }

  // Init
  Shape() {
    Reset();
  }
};

struct CompileInfoTransdataDSL {
  // construct func
  CompileInfoTransdataDSL() = default;
  CompileInfoTransdataDSL(const std::string& op_type, const nlohmann::json& parsed_json_obj);
  // check value
  bool Check() const;
  bool check_success{false};
  bool is_const_compile{false};
  bool is_const{false};
  bool is_forward{false};
  int64_t align_size{DEFAULT};
  int64_t core_num{DEFAULT};
  std::vector<size_t> src_fuse;
  std::vector<size_t> permute;
  std::vector<size_t> src_pad_mode;
  std::vector<size_t> src_pad_var;
  std::vector<std::vector<int64_t>> ub_info;
  std::vector<size_t> bn_x1x0;
  std::vector<size_t> bn_c1c0;
  std::vector<size_t> bn_permute;
  std::vector<size_t> bh_x1x0;
  std::vector<size_t> bh_c1c0;
  std::vector<size_t> bh_permute;
  std::unordered_map<std::string, int32_t> const_block_dims;

 private:
  void ParseCommonInfo(const nlohmann::json& parsed_json_obj);
  void ParseBaseGraphInfo(const nlohmann::json& parsed_json_obj);
  void ParseBNGraphInfo(const nlohmann::json& parsed_json_obj);
  void ParseBHGraphInfo(const nlohmann::json& parsed_json_obj);
};

struct MTEInfo {
  int64_t virLen;
  int64_t mainLen;
  int64_t tailLen;

  // Reset
  void Reset() {
    virLen = DEFAULT;
    mainLen = DEFAULT;
    tailLen = DEFAULT;
  }

  // Init
  MTEInfo() {
    Reset();
  }
};

class TransdataClassify {
 public:
  explicit TransdataClassify(const ge::Operator& _op_paras, const CompileInfoTransdataDSL& _compileInfo)
      : op_paras(_op_paras), compileInfo(_compileInfo) {
  }
  ~TransdataClassify() {
  }
  void GetInputOutput(Shape& input, Shape& output, Shape& reshape);
  size_t ChooseStrategy(const Shape& input, const Shape& output);
  size_t TransposeWork(const Shape& input, const Shape& output) const;

 private:
  const ge::Operator& op_paras;
  const CompileInfoTransdataDSL& compileInfo;
  bool is_last_transpose{true};
  bool is_data_move{true};
  size_t index_c{0};
  int64_t UBSize{-1};
  int64_t borrow_factor{-1};
  void DoFusing(Shape& input, Shape& output, Shape& reshape);
  size_t BorrowStrategy(const Shape& input);
  size_t BHBNStrategy(const Shape& input);
  bool IsLegalBurstLen(const Shape& input);
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

namespace transdata_dsl {
#define CEILING(x, y) (((x) + (y) - 1) / (y))
#define REFINE(x, y) CEILING(x, CEILING(x, y))

int64_t SetAlign(int64_t value, int64_t factor);
int64_t Prod(const int64_t* input, size_t ptr, size_t length);
int64_t CeilDiv(int64_t value, int64_t factor);

template <typename T>
int64_t VectorIndex(const std::vector<T>& input, T value) {
  for (int64_t i = 0; i < static_cast<int64_t>(input.size()); i++) {
    if (input[i] == value) {
      return i;
    }
  }
  return DEFAULT;
}
}  // namespace transdata_dsl
}  // namespace optiling

#endif  // TRANSDATA_DSL_H