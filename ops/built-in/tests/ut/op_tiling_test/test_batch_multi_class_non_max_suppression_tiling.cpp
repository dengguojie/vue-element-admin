#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class BatchMultiClassNonMaxSuppressionTiling : public testing::Test {
  protected:
    static void SetUpTestCase() {
      std::cout << "BatchMultiClassNonMaxSuppressionTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "BatchMultiClassNonMaxSuppressionTiling TearDown" << std::endl;
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

TEST_F(BatchMultiClassNonMaxSuppressionTiling, batch_multi_class_non_max_suppression_tiling_0) {
  using namespace optiling;
  std::string op_name = "BatchMultiClassNonMaxSuppression";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"aicore_num\": 32, \"proposal_topk_k\": 2}}";
  std::vector<int64_t> input0{1, 1024, 1, 4}; // boxes_shape
  std::vector<int64_t> input1{1, 1024, 1};    // scores_shape
  std::vector<int64_t> output{1, 4, 100};

  // boxes
  TeOpTensor tensor_input0;
  tensor_input0.shape = input0;
  tensor_input0.dtype = "float16";
  tensor_input0.format = "ND";
  tensor_input0.ori_format = "ND";
  TeOpTensorArg tensor_input_arg0;
  tensor_input_arg0.tensor.push_back(tensor_input0);
  tensor_input_arg0.arg_type = TA_SINGLE;

 // scores
  TeOpTensor tensor_input1;
  tensor_input1.shape = input1;
  tensor_input1.dtype = "float16";
  tensor_input1.format = "ND";
  tensor_input1.ori_format = "ND";
  TeOpTensorArg tensor_input_arg1;
  tensor_input_arg1.tensor.push_back(tensor_input1);
  tensor_input_arg1.arg_type = TA_SINGLE;

  // nmsed_boxes
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";
  tensor_output.format = "ND";
  tensor_output.ori_format = "ND";
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas op_paras;
  op_paras.inputs.push_back(tensor_input_arg0);
  op_paras.inputs.push_back(tensor_input_arg1);
  op_paras.outputs.push_back(tensor_output_arg);
  op_paras.op_type = op_name;

  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "2021_04_02_15_26";

  OpRunInfo runInfo;

  ASSERT_TRUE(iter->second(op_paras, op_compile_info, runInfo));
  //1-cal_mode 2-core_used 3-bath_per_core 4-bath_last_core 
  //5-batch 6-classes 7-boxes_num 8-topk_loop_time 9-topk_loop_tail
  // EXPECT_EQ(to_string(runInfo.tiling_data), "1 2 3 4 5 6 7 8 9 ");
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 1 1 1 1 1024 1 0 1 ");
}

TEST_F(BatchMultiClassNonMaxSuppressionTiling, batch_multi_class_non_max_suppression_tiling_1) {
  using namespace optiling;
  std::string op_name = "BatchMultiClassNonMaxSuppression";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"aicore_num\": 32, \"proposal_topk_k\": 2}}";
  std::vector<int64_t> input0{2, 16, 1, 4}; // boxes_shape
  std::vector<int64_t> input1{2, 16, 1};    // scores_shape
  std::vector<int64_t> output{2,4, 4};

  // boxes
  TeOpTensor tensor_input0;
  tensor_input0.shape = input0;
  tensor_input0.dtype = "float16";
  tensor_input0.format = "ND";
  tensor_input0.ori_format = "ND";
  TeOpTensorArg tensor_input_arg0;
  tensor_input_arg0.tensor.push_back(tensor_input0);
  tensor_input_arg0.arg_type = TA_SINGLE;

  // scores
  TeOpTensor tensor_input1;
  tensor_input1.shape = input1;
  tensor_input1.dtype = "float16";
  tensor_input1.format = "ND";
  tensor_input1.ori_format = "ND";
  TeOpTensorArg tensor_input_arg1;
  tensor_input_arg1.tensor.push_back(tensor_input1);
  tensor_input_arg1.arg_type = TA_SINGLE;

  // nmsed_boxes
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";
  tensor_output.format = "ND";
  tensor_output.ori_format = "ND";
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas op_paras;
  op_paras.inputs.push_back(tensor_input_arg0);
  op_paras.inputs.push_back(tensor_input_arg1);
  op_paras.outputs.push_back(tensor_output_arg);
  op_paras.op_type = op_name;

  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "2021_04_02_17_26";

  OpRunInfo runInfo;

  ASSERT_TRUE(iter->second(op_paras, op_compile_info, runInfo));
  //1-cal_mode 2-core_used 3-bath_per_core 4-bath_last_core 
  //5-batch 6-classes 7-boxes_num 8-topk_loop_time 9-topk_loop_tail
  // EXPECT_EQ(to_string(runInfo.tiling_data), "1 2 3 4 5 6 7 8 9 ");
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 2 1 1 2 16 1 0 1 ");
}

