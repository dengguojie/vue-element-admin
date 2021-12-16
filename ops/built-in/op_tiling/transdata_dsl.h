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
constexpr size_t STRIDE_16 = 16;
constexpr size_t STRIDE_2 = 2;
constexpr int64_t MAX_DIM = 8;
constexpr int64_t DEFAULT = -1;

constexpr int64_t PACKET_SENDING_RATE = 256;
constexpr int64_t BLOCK = 32;
constexpr int64_t INT8_BYTE = 1;
constexpr int64_t FP16_BYTE = 2;
constexpr int64_t FP32_BYTE = 4;
constexpr int64_t FP32_TRANSPOSE_LIMIT = 128;

constexpr int64_t BaseSch = 0;
constexpr int64_t BorrowNSch = 1;
constexpr int64_t BorrowHSch = 2;
constexpr int64_t StorageAlign = 0;
constexpr int64_t CommonAlign = 1;
constexpr int64_t EXISTED_C1C0 = 2;

constexpr int64_t CONST_KEY = 123;
constexpr int64_t FORWARD_KEY = 2;
constexpr int64_t BACKWARD_KEY = 3;

struct Shape {
  int64_t shape[MAX_DIM];
  size_t size;

  void Reset() {
    size = MAX_DIM;
    for (size_t i = 0; i < size; i++) {
      shape[i] = DEFAULT;
    }
  }

  void SetSize(size_t num) {
    size = num;
  }

  Shape() {
    Reset();
  }
};

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
  int64_t align_size{DEFAULT};
  int64_t pad_align_size{DEFAULT};
  int64_t core_num{DEFAULT};
  std::vector<size_t> src_pad;
  std::vector<size_t> src_fuse;
  std::vector<size_t> permute;
  std::vector<size_t> unknown_dims;
  std::vector<std::vector<int64_t>> ub_info;
  std::unordered_map<std::string, int32_t> const_block_dims;
};

struct MTEInfo {
  int64_t virLen;
  int64_t mainLen;
  int64_t tailLen;

  void Reset() {
    virLen = DEFAULT;
    mainLen = DEFAULT;
    tailLen = DEFAULT;
  }

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
  int64_t ChooseStrategy(Shape& input, Shape& output) const;

 private:
  const ge::Operator& op_paras;
  const CompileInfoTransdataDSL& compileInfo;
  void DoFusing(Shape& input, Shape& output, Shape& reshape);
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
int64_t SetAlign(int64_t value, int64_t align_factor);

template <typename T>
int64_t VectorIndex(const std::vector<T>& input, T value) {
  for (int64_t i = 0; i < static_cast<int64_t>(input.size()); i++) {
    if (input[i] == value) {
      return i;
    }
  }
  return DEFAULT;
}

int64_t Prod(const int64_t* input, size_t ptr, size_t length);

int64_t CeilDiv(int64_t value, int64_t factor);
}  // namespace transdata_dsl
}  // namespace optiling

#endif  // TRANSDATA_DSL_H