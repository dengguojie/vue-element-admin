#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "op_tiling/vector_tiling.h"
#include "op_tiling/softmax_tiling.h"
#include "graph/utils/op_desc_utils.h"
#include "op_tiling/tiling_handler.h"

using namespace std;
using namespace ge;
using namespace optiling;

class SoftmaxTiling : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "SoftmaxTiling SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "SoftmaxTiling TearDown" << std::endl; }
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

TEST_F(SoftmaxTiling, SoftmaxTiling1) {
  std::string compileInfo = R"({ "ori_axis": [3], "common_info": [262144, 32, 1, 16]})";
  nlohmann::json op_info = nlohmann::json::parse(compileInfo.c_str());

  std::vector<int64_t> input{1024,32,64,64};
  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NHWC";
  tensor_input.ori_format = "NHWC";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TensorArgType::TA_SINGLE;

  TeOpParas op_paras;
  op_paras.inputs.push_back(tensor_input_arg);
  op_paras.outputs.push_back(tensor_input_arg);
  op_paras.op_type = "SoftmaxV2";

  optiling::OpRunInfo runInfo;
  optiling::Softmax softmax("softmax", op_paras, op_info, runInfo);
  ASSERT_TRUE(softmax.DoTiling());
  EXPECT_EQ(true, true);
}

TEST_F(SoftmaxTiling, SoftmaxTiling2) {
  std::string compileInfo = R"({ "ori_axis": [0], "common_info": [262144, 32, 1, 16]})";
  nlohmann::json op_info = nlohmann::json::parse(compileInfo.c_str());

  std::vector<int64_t> input{1024};
  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NHWC";
  tensor_input.ori_format = "NHWC";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TensorArgType::TA_SINGLE;

  TeOpParas op_paras;
  op_paras.inputs.push_back(tensor_input_arg);
  op_paras.outputs.push_back(tensor_input_arg);
  op_paras.op_type = "SoftmaxV2";

  optiling::OpRunInfo runInfo;
  optiling::Softmax softmax("softmax", op_paras, op_info, runInfo);
  ASSERT_TRUE(softmax.DoTiling());
  EXPECT_EQ(true, true);
}

TEST_F(SoftmaxTiling, SoftmaxTiling3) {
  std::string compileInfo = R"({ "ori_axis": [0], "common_info": [262144, 32, 1, 16]})";
  nlohmann::json op_info = nlohmann::json::parse(compileInfo.c_str());

  std::vector<int64_t> input{1024,32,64,64};
  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NHWC";
  tensor_input.ori_format = "NHWC";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TensorArgType::TA_SINGLE;

  TeOpParas op_paras;
  op_paras.inputs.push_back(tensor_input_arg);
  op_paras.outputs.push_back(tensor_input_arg);
  op_paras.op_type = "SoftmaxV2";

  optiling::OpRunInfo runInfo;
  optiling::Softmax softmax("softmax", op_paras, op_info, runInfo);
  ASSERT_TRUE(softmax.DoTiling());
  EXPECT_EQ(true, true);
}

TEST_F(SoftmaxTiling, SoftmaxTiling4) {
  std::string compileInfo = R"({ "ori_axis": [1], "common_info": [262144, 32, 1, 16]})";
  nlohmann::json op_info = nlohmann::json::parse(compileInfo.c_str());

  std::vector<int64_t> input{1024,32,64,64};
  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NHWC";
  tensor_input.ori_format = "NHWC";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TensorArgType::TA_SINGLE;

  TeOpParas op_paras;
  op_paras.inputs.push_back(tensor_input_arg);
  op_paras.outputs.push_back(tensor_input_arg);
  op_paras.op_type = "SoftmaxV2";

  optiling::OpRunInfo runInfo;
  optiling::Softmax softmax("softmax", op_paras, op_info, runInfo);
  ASSERT_TRUE(softmax.DoTiling());
  EXPECT_EQ(true, true);
}

TEST_F(SoftmaxTiling, SoftmaxTiling5) {
  std::string compileInfo = R"({ "ori_axis": [1], "common_info": [262144, 32, 1, 16]})";
  nlohmann::json op_info = nlohmann::json::parse(compileInfo.c_str());

  std::vector<int64_t> input{16,32,64,64};
  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NHWC";
  tensor_input.ori_format = "NHWC";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TensorArgType::TA_SINGLE;

  TeOpParas op_paras;
  op_paras.inputs.push_back(tensor_input_arg);
  op_paras.outputs.push_back(tensor_input_arg);
  op_paras.op_type = "SoftmaxV2";

  optiling::OpRunInfo runInfo;
  optiling::Softmax softmax("softmax", op_paras, op_info, runInfo);
  ASSERT_TRUE(softmax.DoTiling());
  EXPECT_EQ(true, true);
}