/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file auto_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_AUTO_TILING_CONTEXT_H
#define OPS_BUILT_IN_OP_TILING_AUTO_TILING_CONTEXT_H

#include <iterator>

#include "graph/utils/op_desc_utils.h"
#include "exe_graph/runtime/tiling_context.h"
#include "external/graph/operator.h"
#include "register/op_tiling_info.h"
#include "vector_op_info.h"
#include "vector_tiling_rt2.h"

namespace optiling {
class OpShape {
public:
  OpShape() = default;
  explicit OpShape(const gert::Shape* _rt_shape) : rt_shape(_rt_shape) {};
  explicit OpShape(const ge::GeShape* _ge_shape) : ge_shape(_ge_shape) {};
  ~OpShape() = default;

  size_t GetDimNum() const;
  int64_t GetDim(size_t idx) const;
  int64_t GetShapeSize() const;
  bool Empty() const;

private:
  const gert::Shape* rt_shape{nullptr};
  const ge::GeShape* ge_shape{nullptr};
};

class AutoTilingOp {
public:
  explicit AutoTilingOp(const char* _op_type,
                        const ge::Operator* _op_paras,
                        const AutoTilingCompileInfo* _compile_info,
                        utils::OpRunInfo* _run_info)
        : op_type(_op_type),
          op_paras(_op_paras),
          compile_info(_compile_info),
          run_info(_run_info) {};
  ~AutoTilingOp() = default;

public:
  bool GetInputDataType(size_t idx, ge::DataType& dtype);
  bool GetInputDataType(const OpInfoImpl* op_info, ge::DataType& dtype);
  bool GetOutputDataType(size_t idx, ge::DataType& dtype);
  size_t GetInputNums();
  size_t GetInputNums(const OpInfoImpl* op_info);
  size_t GetOutputNums();
  OpShape GetInputShape(size_t idx);
  OpShape GetOutputShape(size_t idx);
  const char* GetOpType();
  const AutoTilingCompileInfo* GetCompileInfo();
  bool SetBlockDim(uint32_t block_dims);
  bool SetTilingKey(uint64_t tiling_key);
  bool SetNeedAtomic(bool flag);
  void SetCompileInfo(const AutoTilingCompileInfo* _compile_info);
  bool GetAttr(const char* name, size_t index, int64_t& value);
  bool GetAttr(const char* name, size_t index, std::vector<int64_t>& values);
  bool GetConstInput(const char* name, size_t index, int64_t& value);
  bool GetConstInput(const char* name, size_t index, std::vector<int64_t>& values);
  const ge::Operator* GetOpParas();
  utils::OpRunInfo* GetRunInfo();

  template<typename ForwardIterator>
  bool AddWorkspace(ForwardIterator first, size_t n) {
    for (size_t i = 0; i < n; i++) {
      run_info->AddWorkspace(static_cast<int64_t>(*first));
      first++;
    }
    return true;
  }

  template <typename T>
  bool Append(const T& data) {
    run_info->AddTilingData(data);
    return true;
  }

private:
  const char* op_type {nullptr};
  const ge::Operator* op_paras {nullptr};
  const AutoTilingCompileInfo* compile_info {nullptr};
  utils::OpRunInfo* run_info {nullptr};
};

class AutoTilingContext {
public:
  explicit AutoTilingContext(gert::TilingContext* _context) : context(_context) {
    tiling_data = context->GetRawTilingData();
  }
  explicit AutoTilingContext(gert::TilingContext* _context, const AutoTilingCompileInfo* _compile_info)
      : context(_context), compile_info(_compile_info) {
    tiling_data = context->GetRawTilingData();
  }
  ~AutoTilingContext() = default;

public:
  bool GetInputDataType(size_t idx, ge::DataType& dtype);
  bool GetInputDataType(const OpInfoImpl* op_info, ge::DataType& dtype);
  bool GetOutputDataType(size_t idx, ge::DataType& dtype);
  size_t GetInputNums();
  size_t GetInputNums(const OpInfoImpl* op_info);
  size_t GetOutputNums();
  OpShape GetInputShape(size_t idx);
  OpShape GetOutputShape(size_t idx);
  const char* GetOpType();
  const AutoTilingCompileInfo* GetCompileInfo();
  bool SetBlockDim(uint32_t block_dims);
  bool SetTilingKey(uint64_t tiling_key);
  bool SetNeedAtomic(bool flag);
  void SetCompileInfo(const AutoTilingCompileInfo* _compile_info);
  bool GetAttr(const char* name, size_t index, int64_t& value);
  bool GetAttr(const char* name, size_t index, std::vector<int64_t>& values);
  bool GetConstInput(const char* name, size_t index, int64_t& value);
  bool GetConstInput(const char* name, size_t index, std::vector<int64_t>& values);
  const ge::Operator* GetOpParas();
  utils::OpRunInfo* GetRunInfo();

  template<typename ForwardIterator>
  bool AddWorkspace(ForwardIterator first, size_t n) {
    auto workspace_data = context->GetWorkspaceSizes(n);
    if (workspace_data == nullptr) {
      return false;
    }
    for (size_t i = 0; i < n; i++) {
      workspace_data[i] = *first;
      first++;
    }
    return true;
  }

  template <typename T>
  bool Append(const T& data) {
    if (tiling_data->Append(data) == ge::GRAPH_FAILED) {
      return false;
    }
    return true;
  }

private:
  gert::TilingContext *context{nullptr};
  gert::TilingData* tiling_data;
  const AutoTilingCompileInfo* compile_info{nullptr};
};
} // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_AUTO_TILING_CONTEXT_H