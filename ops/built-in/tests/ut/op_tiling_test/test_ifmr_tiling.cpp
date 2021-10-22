/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file test_ifmr_tiling.cpp
 * \brief dynamic tiling test of IFMR
 */
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class IFMRTilingTest : public testing::Test {
protected:
static void SetUpTestCase() {
    std::cout << "IouTilingTest SetUp" << std::endl;
}

static void TearDownTestCase() {
    std::cout << "IouTilingTest TearDown" << std::endl;
}
};

static string to_string(const std::stringstream &tiling_data) {
    auto data = tiling_data.str();
    string result;
    int32_t tmp = 0;
    for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
        memcpy(&tmp, data.c_str() + i, sizeof(tmp));
        result += std::to_string(tmp);
        result += " ";
    }

    return result;
}

TEST_F(IFMRTilingTest, ifmr_tiling_test_1) {
    using namespace optiling;
    optiling::OpRunInfo op_run_info;
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("IFMR");
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
    TeOpTensorArg tensorInputs, tensorOutputsArg;
    TeOpParas opParas;

    vector<vector<int64_t>> input_shapes = {
        {16, 3, 224, 224},
        {1},
        {1},
        {1024}
    };
    vector<vector<int64_t>> output_shapes = {
        {1},
        {1}
    };

    vector<string> input_types = {"float16", "float16", "float16", "int32"};
    vector<string> output_types = {"float32", "float32"};

    for (size_t i = 0; i < input_shapes.size(); i++) {
        tensorInputs.tensor.clear();
        TeOpTensor tensorInput;
        tensorInput.shape = input_shapes[i];
        tensorInput.dtype = input_types[i];
        tensorInputs.tensor.push_back(tensorInput);
        tensorInputs.arg_type = TA_SINGLE;
        opParas.inputs.push_back(tensorInputs);
    }

    for (size_t i = 0; i < output_shapes.size(); i++) {
        TeOpTensor tensorOutput;
        tensorOutput.shape = output_shapes[i];
        tensorOutput.dtype = output_types[i];
        tensorOutputsArg.tensor.push_back(tensorOutput);
        tensorOutputsArg.arg_type = TA_SINGLE;
        opParas.outputs.push_back(tensorOutputsArg);
    }

    opParas.op_type = "IFMR";
    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\": 253952, \"block_size\": 8}}";
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "ifmr_tiling_test_1";

    // do tilling, get runInfo
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "2408448 ");
}
