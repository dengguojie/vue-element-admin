#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class BroadcastToTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BroadcastToTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BroadcastToTiling TearDown" << std::endl;
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

TEST_F(BroadcastToTiling, BroadcastTo_tiling_test_1) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("BroadcastTo");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
      {1, 1, 5},
      {3},
  };

  vector<string> dtypes = {"float16", "int32"};
  for (size_t i = 0; i < input_shapes.size(); i++) {
    tensorInputs.tensor.clear();
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);
  }

  std::vector<int64_t> shape_shape;
  shape_shape.push_back(3);
  ge::Shape ge_shape(shape_shape);
  ge::Tensor const_tensor(ge::TensorDesc(ge_shape, ge::Format::FORMAT_ND, ge::DataType::DT_INT32));
  int32_t buf[3];
  buf[0] = 3;
  buf[1] = 1;
  buf[2] = 5;
  opParas.const_inputs["shape"] = std::make_tuple((const unsigned char *)buf, sizeof(buf), const_tensor);

  TeOpTensor tensorOutput;
  tensorOutput.shape = {3, 1, 5};
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "BroadcastTo";
  std::string compileInfo = R"( {"_pattern": "Broadcast", "push_status": 0,"_flag_info": [false, false, true, false, false, false, false], "_base_info": {"000": [32, 2, 43680, 21840]}, "_elewise_vars": {"0": [10000, 10100], "1": [10000, 10100, 20000, 30000], "2": [10000, 10100, 20000, 30001], "3": [10000, 10100, 20000, 30002], "5": [10000, 10100, 20001, 30001], "6": [10000, 10100, 20001, 30002], "9": [10000, 10100, 20002, 30002]}, "_vars": {"0": ["_dim_0_0", "_dim_1_0"], "1": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_2"], "5": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_1_0", "_block_factor_2", "_ub_factor_2"]}})";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456a";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "1 1 ");
}

