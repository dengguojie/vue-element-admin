#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class UnsortedSegmentSumTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "UnsortedSegmentSumTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "UnsortedSegmentSumTiling TearDown" << std::endl;
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
  std::cout << "to_string" << std::endl;
  std::cout << result << std::endl;
  return result;
}

TEST_F(UnsortedSegmentSumTiling, segmentsum_tiling_0) {
  using namespace optiling;
  std::string op_name = "SegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{2,3132864};
  std::vector<int64_t> inputB{2};
  std::vector<int32_t> segment_ids{0,0};
  std::vector<int64_t> output{1,3132864};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  opParas.const_inputs["segment_ids"] =
            std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)segment_ids.data(), segment_ids.size() * 4, ge::Tensor());
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "aa";
  OpRunInfo runInfo;

  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "17 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3132864 96 4096 32768 2488 19904 2 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ");
}
TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_0) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{2,3132864};
  std::vector<int64_t> inputB{2};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments{1,};
  std::vector<int64_t> output{1,3132864};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  opParas.const_inputs["num_segments"] =
            std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)num_segments.data(), num_segments.size() * 4, ge::Tensor());
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "aa";
  OpRunInfo runInfo;

  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));

  EXPECT_EQ(to_string(runInfo.tiling_data), "17 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3132864 96 4096 32768 2488 19904 2 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}
TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_1) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{1024,80};
  std::vector<int64_t> inputB{1024};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments{300,};
  std::vector<int64_t> output{300,80};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
  
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  opParas.const_inputs["num_segments"] =
            std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)num_segments.data(), num_segments.size() * 4, ge::Tensor());
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "aa";
  OpRunInfo runInfo;

  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));

  EXPECT_EQ(to_string(runInfo.tiling_data), "1 32 2560 1 320 320 2560 2560 32 32 1 320 320 2560 2560 32 32 2560 1 320 320 2560 2560 32 32 1 320 320 2560 2560 32 32 80 1 10 80 10 80 1024 32 1 4 4 32 32 32 1 4 4 32 32 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}
TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_2) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{1024,80};
  std::vector<int64_t> inputB{1024};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments{300,};
  std::vector<int64_t> output{300,80};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
  
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  opParas.const_inputs["num_segments"] =
            std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)num_segments.data(), num_segments.size() * 4, ge::Tensor());
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "aa";
  OpRunInfo runInfo;
  
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  
  EXPECT_EQ(to_string(runInfo.tiling_data), "9 32 9 21 1024 1 1024 128 1024 128 80 1 5 80 1 5 80 1 0 0 5 5 419 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}
TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_3) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{2,3132864};
  std::vector<int64_t> inputB{2};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments{1,};
  std::vector<int64_t> output{1,3132864};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  opParas.const_inputs["num_segments"] =
            std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)num_segments.data(), num_segments.size() * 4, ge::Tensor());
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "aa";
  OpRunInfo runInfo;

  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  
  EXPECT_EQ(to_string(runInfo.tiling_data), "15 32 97888 98336 2 1 2 1 2 1 3132864 3 2048 32768 1 2022 32 253 0 0 0 0 0 0 0 0 0 0 0 2048 2 2 1 1 4 1 ");
}
TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_4) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{46,44};
  std::vector<int64_t> inputB{46};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments{100,};
  std::vector<int64_t> output{100,44};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  opParas.const_inputs["num_segments"] =
            std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)num_segments.data(), num_segments.size() * 4, ge::Tensor());
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "aa";
  OpRunInfo runInfo;

  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "4 32 44 1 6 6 44 44 1 1 1 6 6 44 44 1 1 660 1 83 83 660 660 15 15 1 83 83 660 660 15 15 44 1 5 40 1 4 46 1 1 1 1 1 1 15 1 2 2 15 15 1 1 1 1 2 1 2 1 40 48 4 0 1 1 1 1 2 1 2 1 ");
}
TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_5) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{0,44};
  std::vector<int64_t> inputB{0};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments{100,};
  std::vector<int64_t> output{100,44};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  opParas.const_inputs["num_segments"] =
            std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)num_segments.data(), num_segments.size() * 4, ge::Tensor());
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "aa";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}
TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_6) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{1024,15};
  std::vector<int64_t> inputB{1024};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments{600,};
  std::vector<int64_t> output{300,80};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
  
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  opParas.const_inputs["num_segments"] =
            std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)num_segments.data(), num_segments.size() * 4, ge::Tensor());
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "aa";
  OpRunInfo runInfo;
  
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  
  EXPECT_EQ(to_string(runInfo.tiling_data), "13 32 16 104 1024 1 1024 128 1024 128 15 1 15 15 1 15 15 1 0 8 1 1 2180 1 1 16 16 104 104 97 97 0 0 0 0 0 ");
}
TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_7) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{2,3132864};
  std::vector<int64_t> inputB{2};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments{2,};
  std::vector<int64_t> output{1,3132864};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  opParas.const_inputs["num_segments"] =
            std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)num_segments.data(), num_segments.size() * 4, ge::Tensor());
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "aa";
  OpRunInfo runInfo;

  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  
  EXPECT_EQ(to_string(runInfo.tiling_data), "11 2 1 1 2 1 2 1 2 1 3132864 196 1000 16000 125 804 12864 101 0 0 1000 804 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}
TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_8) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{1024,15};
  std::vector<int64_t> inputB{1024};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments{100000,};
  std::vector<int64_t> output{300,80};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
  
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  opParas.const_inputs["num_segments"] =
            std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)num_segments.data(), num_segments.size() * 4, ge::Tensor());
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "aa";
  OpRunInfo runInfo;
  
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  
  EXPECT_EQ(to_string(runInfo.tiling_data), "13 32 3120 3280 1024 1 1024 128 1024 128 15 1 2043 15 1 881 15 1 12 12 1 1 2180 2 2 2180 940 2180 1100 2043 1031 0 0 0 0 0 ");
}
TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_9) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{1024,1};
  std::vector<int64_t> inputB{1024};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments{300,};
  std::vector<int64_t> output{300,1};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
  
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  opParas.const_inputs["num_segments"] =
            std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)num_segments.data(), num_segments.size() * 4, ge::Tensor());
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "aa";
  OpRunInfo runInfo;

  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));

  EXPECT_EQ(to_string(runInfo.tiling_data), "8 16 64 0 8 0 64 0 64 0 0 8 0 64 0 64 0 64 0 8 0 64 0 64 0 0 8 0 64 0 64 0 1 1 1 0 0 0 1024 64 1 8 8 64 64 64 1 8 8 64 64 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}
TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_10) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{1024,20000};
  std::vector<int64_t> inputB{1024};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments{300,};
  std::vector<int64_t> output{300,80};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
  
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  opParas.const_inputs["num_segments"] =
            std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)num_segments.data(), num_segments.size() * 4, ge::Tensor());
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "aa";
  OpRunInfo runInfo;

  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));

  EXPECT_EQ(to_string(runInfo.tiling_data), "5 32 640000 32 0 0 16384 3616 0 0 32 0 0 16384 3616 0 0 640000 32 0 0 16384 3616 0 0 32 0 0 16384 3616 0 0 20000 2 2048 16384 452 3616 1024 32 1 4 4 32 32 32 1 4 4 32 32 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}
TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_11) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{1024,19994};
  std::vector<int64_t> inputB{1024};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments{300,};
  std::vector<int64_t> output{300,80};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
  
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  opParas.const_inputs["num_segments"] =
            std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)num_segments.data(), num_segments.size() * 4, ge::Tensor());
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "aa";
  OpRunInfo runInfo;

  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));

  EXPECT_EQ(to_string(runInfo.tiling_data), "6 32 639808 32 0 0 10920 9074 0 0 32 0 0 10920 9074 0 0 639808 32 0 0 10920 9074 0 0 32 0 0 10920 9074 0 0 19994 2 1365 10920 1134 9074 1024 32 1 4 4 32 32 32 1 4 4 32 32 0 0 0 0 0 0 0 0 9072 9080 2 1135 0 0 0 0 0 0 0 0 ");
}