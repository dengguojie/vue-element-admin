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
 * \file test_ascendantiquant_tiling.cpp
 * \brief dynamic tiling test of ascend_anti_quant
 */
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "selection_ops.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"
#include "quantize_ops.h"
#define private public
#include "register/op_tiling_registry.h"

using namespace std;
using namespace ge;

class AscendAntiQuantTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AscendAntiQuantTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AscendAntiQuantTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream& tiling_data) {
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

TEST_F(AscendAntiQuantTiling, AscendAntiQuant_tiling_0) {
  std::string op_name = "AscendAntiQuant";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = R"({"common_info": [16256, 32]})";

  std::vector<int64_t> input{16, 6, 79, 69, 32};
  std::vector<int64_t> output{16, 12, 79, 69, 16};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_NC1HWC0, DT_INT8);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_NC1HWC0, DT_FLOAT16);
  auto data = op::Data("data");
  data.update_input_desc_x(tensor_input);
  data.update_output_desc_y(tensor_output);

  auto opParas = op::AscendAntiQuant("AscendAntiQuant");
  opParas.set_input_x(data);
  vector<Operator> inputs{data};
  vector<Operator> outputs{opParas};
  opParas.UpdateInputDesc("x", tensor_input);

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(1, 1);
}