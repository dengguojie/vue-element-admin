/*
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef VECTOR_TILING_HPP
#define VECTOR_TILING_HPP

#include <string>
#include <nlohmann/json.hpp>
#include "graph/debug/ge_log.h"
#include "register/op_tiling.h"

namespace optiling {
    /*
     * @brief: tiling function of reduce operator
     * @param [in] op_type: op_type of the reduce operator
     * @param [in] op_paras: inputs/outputs/atts of the reduce operator
     * @param [in] op_info: compile time generated info of the reduce operator
     * @param [out] run_info: result data
     * @return bool: success or not
     */
    bool ReduceTiling(const std::string &op_type, const TeOpParas &op_paras,
                      const nlohmann::json &op_info, OpRunInfo &run_info);

    /*
     * @brief: tiling function of elementwise operator
     * @param [in] op_type: op_type of the elementwise operator
     * @param [in] op_paras: inputs/outputs/atts of the elementwise operator
     * @param [in] op_info: compile time generated info of the elementwise operator
     * @param [out] run_info: result data
     * @return bool: success or not
     */
    bool EletwiseTiling(const std::string &op_type, const TeOpParas &op_paras,
                        const nlohmann::json &op_info, OpRunInfo &run_info);
}

#endif //VECTOR_TILING_HPP
