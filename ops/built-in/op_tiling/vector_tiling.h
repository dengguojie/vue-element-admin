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
 * \file vector_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_VECTOR_TILING_H_
#define OPS_BUILT_IN_OP_TILING_VECTOR_TILING_H_

#include <string>
#include <map>
#include <chrono>
#include <memory>
#include <cstdlib>
#include <nlohmann/json.hpp>

#include "register/op_tiling_registry.h"
#include "op_log.h"
#include "error_log.h"
#include "graph/debug/ge_log.h"
#include "vector_tiling_log.h"

#include "op_tiling.h"

namespace optiling {

class OpInfo {
public:
    explicit OpInfo(const std::vector<std::vector<int64_t>>& _op_input_shapes,
                    const ge::DataType& _op_in_type,
                    const std::vector<std::vector<int32_t>>& _op_reduce_axes = dummy_variable) :
            op_input_shapes (_op_input_shapes),
            op_in_type (_op_in_type),
            op_reduce_axes (_op_reduce_axes){
    }

    const std::vector<std::vector<int64_t>>& GetInputShape() const {
      return op_input_shapes;
    }

    const std::vector<std::vector<int32_t>>& GetReduceAxes() const {
      return op_reduce_axes;
    }

    const ge::DataType& GetInType() const {
      return op_in_type;
    }

private:
    const std::vector<std::vector<int64_t>>& op_input_shapes;
    const ge::DataType& op_in_type;
    const std::vector<std::vector<int32_t>>& op_reduce_axes;
    static const std::vector<std::vector<int32_t>> dummy_variable;
};

class AutoTilingHandler {
  public:
  AutoTilingHandler(const std::string& o, const std::string& p) : op_type(o), pattern(p) {}
  virtual bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const = 0;
  virtual bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info, const OpInfo& op_info) const = 0;
  virtual ~AutoTilingHandler() = default;

  protected:
  const std::string op_type;
  const std::string pattern;
};

/*
 * @brief: tiling function of reduce operator
 * @param [in] op_type: op_type of the reduce operator
 * @param [in] op_paras: inputs/outputs/attrs of the reduce operator
 * @param [in] json_info: compile time generated info of the reduce operator
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool ReduceTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& json_info,
                  OpRunInfo& run_info);
bool ReduceTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& json_info,
                  utils::OpRunInfo& run_info);
bool ReduceTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& json_info,
                  utils::OpRunInfo& run_info, const OpInfo& op_info);
/*
 * @brief: tiling function of element-wise operator
 * @param [in] op_type: op_type of the element-wise operator
 * @param [in] op_paras: inputs/outputs/attrs of the element-wise operator
 * @param [in] compile_info: compile time generated info of the element-wise operator
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool EletwiseTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& compile_info,
                    OpRunInfo& run_info);
bool EletwiseTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& compile_info,
                    utils::OpRunInfo& run_info);
bool EletwiseTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& compile_info,
                    utils::OpRunInfo& run_info, const OpInfo& op_info);

std::shared_ptr<AutoTilingHandler> CreateAutoTilingHandler(const std::string& op_type, const std::string& pattern,
                                                               const nlohmann::json& parsed_compile_info);
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_VECTOR_TILING_H_
