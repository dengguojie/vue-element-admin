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
#include <string>
#include <cstdlib>
#include <nlohmann/json.hpp>

#include "register/op_tiling_registry.h"
#include "op_log.h"
#include "error_log.h"
#include "graph/debug/ge_log.h"
#include "vector_tiling_log.h"

#include "op_tiling.h"

namespace optiling {

struct AutoTilingCompileInfo {
  AutoTilingCompileInfo(const std::string& o,
                        const std::string& p,
                        const std::shared_ptr<void> c) : op_type(o), pattern(p), compile_info(c) {}
  std::string op_type;
  std::string pattern;
  std::shared_ptr<void> compile_info;
};

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

/*
 * @brief: tiling function of reduce operator
 * @param [in] op_type: op_type of the reduce operator
 * @param [in] op_paras: inputs/outputs/attrs of the reduce operator
 * @param [in] op_info: compile time generated info of the reduce operator
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool ReduceTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                  OpRunInfo& run_info);
bool ReduceTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                  utils::OpRunInfo& run_info);
bool ReduceTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                  utils::OpRunInfo& run_info, const OpInfo& opInfo);
/*
 * @brief: tiling function of element-wise operator
 * @param [in] op_type: op_type of the element-wise operator
 * @param [in] op_paras: inputs/outputs/attrs of the element-wise operator
 * @param [in] op_info: compile time generated info of the element-wise operator
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool EletwiseTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                    OpRunInfo& run_info);
bool EletwiseTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                    utils::OpRunInfo& run_info);
bool EletwiseTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                    utils::OpRunInfo& run_info, const OpInfo& opInfo);
/*
 * @brief: tiling function of norm operator
 * @param [in] op_type: op_type of the norm operator
 * @param [in] op_paras: inputs/outputs/attrs of the norm operator
 * @param [in] op_compile_info: compile time generated info of the norm operator
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool NormTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                utils::OpRunInfo& run_info);

/*
 * @brief: tiling function of transpose operator
 * @param [in] op_type: op_type of the transpose operator
 * @param [in] op_paras: inputs/outputs/atts of the transpose operator
 * @param [in] op_info: compile time generated info of the transpose operator
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool TransposeDsl(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& compile_info,
                  utils::OpRunInfo& run_info);

}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_VECTOR_TILING_H_
