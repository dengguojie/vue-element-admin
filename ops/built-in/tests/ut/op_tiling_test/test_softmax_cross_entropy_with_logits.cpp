#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <vector>
#define private public
#include "array_ops.h"
#include "common/utils/ut_op_util.h"
#include "nn_norm_ops.h"
#include "op_tiling/op_tiling_util.h"
#include "register/op_tiling_registry.h"
#include "test_common.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class SoftmaxCrossEntropyWithLogitsTiling : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "SoftmaxCrossEntropyWithLogitsTiling SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "SoftmaxCrossEntropyWithLogitsTiling TearDown" << std::endl; }
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


TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_1) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
    "_align_base_key":10000,
    "_available_size":10528,
    "_core_num":32,
    "_custom_vars":{},
    "_normal_vars":{
        "1":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10001":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10002":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10004":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10006":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10008":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10009":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10010":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "101":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10101":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10102":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10104":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10106":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10108":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10109":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10110":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "102":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "104":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "106":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "108":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "109":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "110":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "2":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "4":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "6":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "8":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "9":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ]
    },
    "_dimension_align_ward":[
        2,
        2
    ],
    "_is_const":false,
    "_is_template":true,
    "_max_dim_len":2,
    "_max_type":"float32",
    "_min_type":"float32",
    "_pattern":"SoftmaxCrossEntropyWithLogits",
    "common_info":{
        "ub_size":262144,
        "core_num":32
    }
  })";

  std::vector<std::vector<int64_t>> inputs{{14, 2}, {14, 1}};

  std::vector<std::vector<int64_t>> outputs{{
                                                14,
                                            },
                                            {14, 2}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}


TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_2) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
    "_align_base_key":10000,
    "_available_size":10528,
    "_core_num":32,
    "_custom_vars":{},
    "_normal_vars":{
        "1":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10001":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10002":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10004":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10006":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10008":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10009":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10010":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "101":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10101":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10102":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10104":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10106":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10108":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10109":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10110":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "102":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "104":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "106":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "108":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "109":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "110":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "2":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "4":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "6":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "8":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "9":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ]
    },
    "_dimension_align_ward":[
        2,
        2
    ],
    "_is_const":false,
    "_is_template":true,
    "_max_dim_len":2,
    "_max_type":"float32",
    "_min_type":"float32",
    "_pattern":"SoftmaxCrossEntropyWithLogits",
    "common_info":{
        "ub_size":262144,
        "core_num":32
    }
  })";

  std::vector<std::vector<int64_t>> inputs{{320, 3200}, {320, 3200}};

  std::vector<std::vector<int64_t>> outputs{{
                                                320,
                                            },
                                            {320, 3200}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}


TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_3) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
    "_align_base_key":10000,
    "_available_size":10528,
    "_core_num":32,
    "_custom_vars":{},
    "_normal_vars":{
        "1":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10001":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10002":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10004":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10006":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10008":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10009":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10010":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "101":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10101":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10102":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10104":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10106":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10108":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10109":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10110":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "102":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "104":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "106":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "108":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "109":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "110":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "2":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "4":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "6":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "8":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "9":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ]
    },
    "_dimension_align_ward":[
        2,
        2
    ],
    "_is_const":false,
    "_is_template":true,
    "_max_dim_len":2,
    "_max_type":"float32",
    "_min_type":"float32",
    "_pattern":"SoftmaxCrossEntropyWithLogits",
    "common_info":{
        "ub_size":262144,
        "core_num":32
    }
  })";

  std::vector<std::vector<int64_t>> inputs{{320, 32000}, {320, 32000}};

  std::vector<std::vector<int64_t>> outputs{{
                                                320,
                                            },
                                            {320, 32000}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}


TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_4) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
    "_align_base_key":10000,
    "_available_size":10528,
    "_core_num":32,
    "_custom_vars":{},
    "_normal_vars":{
        "1":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10001":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10002":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10004":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10006":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10008":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10009":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10010":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "101":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10101":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10102":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10104":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10106":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10108":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10109":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "10110":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "102":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "104":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "106":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "108":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "109":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "110":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "2":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "4":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "6":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "8":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ],
        "9":[
          "_dim_0_0",
          "_dim_1_0",
          "_dim_0_1",
          "_dim_1_1",
          "_block_nparts",
          "_ub_factor"
        ]
    },
    "_dimension_align_ward":[
        2,
        2
    ],
    "_is_const":false,
    "_is_template":true,
    "_max_dim_len":2,
    "_max_type":"float32",
    "_min_type":"float32",
    "_pattern":"SoftmaxCrossEntropyWithLogits",
    "common_info":{
        "ub_size":262144,
        "core_num":32
    }
  })";

  std::vector<std::vector<int64_t>> inputs{{32, 3200}, {32, 3200}};

  std::vector<std::vector<int64_t>> outputs{{
                                                32,
                                            },
                                            {32, 3200}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}

