#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class TileWithAxisTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TileWithAxisTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TileWithAxisTiling TearDown" << std::endl;
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

TEST_F(TileWithAxisTiling, TileWithAxisTiling_test_1) {

  using namespace optiling;

  std::string op_name = "TileWithAxis";

  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = R"({"ori_axis_value": -1, "attr_axis": 0, "attr_tiles": 2, "_fusion_index": [[0], [1], [2], [3], [4], [5]], "_pattern": "Broadcast", "_flag_info": [false, false, true, false, false, false, false], "_base_info": {"000": [32, 4, 32760, 16376]}, "_elewise_vars": {"0": [10100], "1": [10100, 20000, 30000], "2": [10100, 20000, 30001], "3": [10100, 20000, 30002], "4": [10100, 20000, 30003], "5": [10100, 20000, 30004], "6": [10100, 20000, 30005], "8": [10100, 20001, 30001], "9": [10100, 20001, 30002], "10": [10100, 20001, 30003], "11": [10100, 20001, 30004], "12": [10100, 20001, 30005], "15": [10100, 20002, 30002], "16": [10100, 20002, 30003], "17": [10100, 20002, 30004], "18": [10100, 20002, 30005], "22": [10100, 20003, 30003], "23": [10100, 20003, 30004], "24": [10100, 20003, 30005], "29": [10100, 20004, 30004], "30": [10100, 20004, 30005], "36": [10100, 20005, 30005]}, "push_status": 1})";

  std::vector<int64_t> input{5, 2, 16, 8, 3};
  std::vector<int64_t> output{5, 2, 16, 8, 3};
  std::string in_dtype = "float32";

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = in_dtype;

  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = in_dtype;

  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_input);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpTensorArg tensor_arg_out;
  tensor_arg_out.tensor.push_back(tensor_output);
  tensor_arg_out.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_arg);
  opParas.outputs.push_back(tensor_arg_out);
  opParas.op_type = op_name;

  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "TILEWITHAXIS__COUNTER__1";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));

  EXPECT_EQ(to_string(runInfo.tiling_data), "5 16 16 ");

  std::cout << "TileWithAxisTiling_test_1 " << 12 << std::endl;
}
