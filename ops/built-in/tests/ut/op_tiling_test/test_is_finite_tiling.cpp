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
 * \file test_is_finite_tiling.cpp
 * \brief dynamic tiling test of is_finite
 */
#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "selection_ops.h"
#include "math_ops.h"
#include "array_ops.h"
#include "register/op_tiling_registry.h"

using namespace std;
using namespace ge;

class IsFiniteTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "IsFiniteTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "IsFiniteTiling TearDown" << std::endl;
    }
};

static std::string to_string(const std::stringstream& tiling_data) {
    auto data = tiling_data.str();
    std::string result;
    int64_t tmp = 0;
    for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
        memcpy(&tmp, data.c_str() + i, sizeof(tmp));
        result += std::to_string(tmp);
        result += " ";
    }

    return result;
}

TEST_F(IsFiniteTiling, IsFiniteTiling1) {
std::string op_name = "IsFinite";
auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

std::string compileInfo =
        "{\"vars\": {\"ub_size\": 24320, \"core_num\": 48, \"input_data_byte\": 4}}";

std::vector<int64_t> input{16, 32};
std::vector<int64_t> output{16, 32};

TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_BOOL);
auto data = op::Data("data");
data.update_input_desc_x(tensor_input);
data.update_output_desc_y(tensor_output);

auto opParas = op::IsFinite("IsFinite");
opParas.set_input_x(data);
opParas.UpdateInputDesc("x", tensor_input);

optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo.c_str());
optiling::utils::OpRunInfo runInfo;
ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
EXPECT_EQ(
        to_string(runInfo.GetAllTilingData()),
"1 512 0 0 0 512 0 512 ");
}

TEST_F(IsFiniteTiling, IsFiniteTiling2) {
std::string op_name = "IsFinite";
auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

std::string compileInfo =
        "{\"vars\": {\"ub_size\": 24320, \"core_num\": 48, \"input_data_byte\": 4}}";

std::vector<int64_t> input{10, 10, 1000};
std::vector<int64_t> output{10, 10, 1000};

TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_BOOL);
auto data = op::Data("data");
data.update_input_desc_x(tensor_input);
data.update_output_desc_y(tensor_output);

auto opParas = op::IsFinite("IsFinite");
opParas.set_input_x(data);
opParas.UpdateInputDesc("x", tensor_input);

optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo.c_str());
optiling::utils::OpRunInfo runInfo;
ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
EXPECT_EQ(
        to_string(runInfo.GetAllTilingData()),
"48 100000 2112 0 2112 736 0 736 ");
}

TEST_F(IsFiniteTiling, IsFiniteTiling3) {
std::string op_name = "IsFinite";
auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

std::string compileInfo =
        "{\"vars\": {\"ub_size\": 24320, \"core_num\": 48, \"input_data_byte\": 4}}";

std::vector<int64_t> input{10, 4, 34, 19, 76};
std::vector<int64_t> output{10, 4, 34, 19, 76};

TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_BOOL);
auto data = op::Data("data");
data.update_input_desc_x(tensor_input);
data.update_output_desc_y(tensor_output);

auto opParas = op::IsFinite("IsFinite");
opParas.set_input_x(data);
opParas.UpdateInputDesc("x", tensor_input);

optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo.c_str());
optiling::utils::OpRunInfo runInfo;
ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
EXPECT_EQ(
        to_string(runInfo.GetAllTilingData()),
"48 1963840 40928 3 4448 40224 3 3744 ");
}

TEST_F(IsFiniteTiling, IsFiniteTiling4) {
std::string op_name = "IsFinite";
auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

std::string compileInfo =
        "{\"vars\": {\"ub_size\": 48640, \"core_num\": 48, \"input_data_byte\": 2}}";

std::vector<int64_t> input{4, 17, 370};
std::vector<int64_t> output{4, 17, 370};

TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT16);
TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_BOOL);
auto data = op::Data("data");
data.update_input_desc_x(tensor_input);
data.update_output_desc_y(tensor_output);

auto opParas = op::IsFinite("IsFinite");
opParas.set_input_x(data);
opParas.UpdateInputDesc("x", tensor_input);

optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo.c_str());
optiling::utils::OpRunInfo runInfo;
ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
EXPECT_EQ(
        to_string(runInfo.GetAllTilingData()),
"47 25160 544 0 544 136 0 136 ");
}

TEST_F(IsFiniteTiling, IsFiniteTiling5) {
std::string op_name = "IsFinite";
auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

std::string compileInfo =
        "{\"vars\": {\"ub_size\": 48640, \"core_num\": 48, \"input_data_byte\": 2}}";

std::vector<int64_t> input{4, 17, 54};
std::vector<int64_t> output{4, 17, 54};

TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT16);
TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_BOOL);
auto data = op::Data("data");
data.update_input_desc_x(tensor_input);
data.update_output_desc_y(tensor_output);

auto opParas = op::IsFinite("IsFinite");
opParas.set_input_x(data);
opParas.UpdateInputDesc("x", tensor_input);

optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo.c_str());
optiling::utils::OpRunInfo runInfo;
ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
EXPECT_EQ(
        to_string(runInfo.GetAllTilingData()),
"1 3672 0 0 0 3672 0 3672 ");
}

TEST_F(IsFiniteTiling, IsFiniteTiling6) {
std::string op_name = "IsFinite";
auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

std::string compileInfo =
        "{\"vars\": {\"ub_size\": 48640, \"core_num\": 48, \"input_data_byte\": 2}}";

std::vector<int64_t> input{41, 17, 54, 78};
std::vector<int64_t> output{41, 17, 54, 78};

TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT16);
TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_BOOL);
auto data = op::Data("data");
data.update_input_desc_x(tensor_input);
data.update_output_desc_y(tensor_output);

auto opParas = op::IsFinite("IsFinite");
opParas.set_input_x(data);
opParas.UpdateInputDesc("x", tensor_input);

optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo.c_str());
optiling::utils::OpRunInfo runInfo;
ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
EXPECT_EQ(
        to_string(runInfo.GetAllTilingData()),
"48 2935764 61184 2 12544 60116 2 11476 ");
}