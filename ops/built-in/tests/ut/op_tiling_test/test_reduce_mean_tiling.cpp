#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class ReduceMeanTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
      std::cout << "ReduceMeanTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "ReduceMeanTiling TearDown" << std::endl;
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

TEST_F(ReduceMeanTiling, ReduceMeanTiling1) {
  using namespace optiling;
  std::string op_name = "ReduceMeanD";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());


  std::string compileInfo = R"({ "_ori_axis": [0], "_pattern": "CommReduce","push_status": 0,"_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5], "_ub_info": [16256], "_ub_info_rf": [16256], "_vars": {"-1000500": ["_dim_1_0", "_block_factor", "_ub_factor"]}})";

  std::vector<int64_t> input{1};
  std::vector<int64_t> output{1};
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
  op_compile_info.key = "REDUCEMEAN__COUNTER__1";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 1);
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 1 1 ");
}

TEST_F(ReduceMeanTiling, ReduceMeanTiling2) {
  using namespace optiling;
  std::string op_name = "ReduceMeanD";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());


  std::string compileInfo = R"({ "_ori_axis": [1], "_pattern": "CommReduce", "push_status": 0, "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5, 4, 9], "_ub_info": [16256, 16000, 16256], "_ub_info_rf": [16256, 16000, 16256], "reduce_mean_cof_dtype": "float32"})";

  std::vector<int64_t> input{7, 2};
  std::vector<int64_t> output{7};
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
  op_compile_info.key = "REDUCEMEAN__COUNTER__2";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 1);
  EXPECT_EQ(to_string(runInfo.tiling_data), "7 2 8 8 1056964608 ");
}

TEST_F(ReduceMeanTiling, ReduceMeanTiling3) {
  using namespace optiling;
  std::string op_name = "ReduceMeanD";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());


  std::string compileInfo = R"({ "_ori_axis": [1], "_pattern": "CommReduce", "push_status": 0, "_common_info": [2, 1, 16, 0, 1], "_pattern_info": [5, 4, 9], "_ub_info": [31488, 31104, 31488], "_ub_info_rf": [31488, 31104, 31488], "reduce_mean_cof_dtype": "float16"})";

  std::vector<int64_t> input{7, 2};
  std::vector<int64_t> output{7};
  std::string in_dtype = "float16";

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
  op_compile_info.key = "REDUCEMEAN__COUNTER__3";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 1);
  EXPECT_EQ(to_string(runInfo.tiling_data), "7 2 16 16 14336 ");
}

TEST_F(ReduceMeanTiling, ReduceTiling4) {
  using namespace optiling;
  std::string op_name = "ReduceMean";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());


  std::string compileInfo = R"({"_idx_before_reduce": 0, "_pattern": "CommReduce", "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [-1], "_ub_info": [32512], "_ub_info_rf": [32512], "_reduce_shape_known": true, "_const_shape_post": true, "_compile_pattern": -1, "_block_dims": {"-1": 32}, "_atomic_flags": {"-1": false}})";

  std::vector<int64_t> input{1, 10};
  std::vector<int64_t> output{1, 10};
  std::vector<int64_t> input_axis{0};
  std::vector<int32_t> axis{0};
  std::string in_dtype = "float32";

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = in_dtype;
  TeOpTensor tensor_input_axis;
  tensor_input_axis.shape = input_axis;
  tensor_input_axis.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = in_dtype;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_input);
  tensor_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_input_axis_arg;
  tensor_input_axis_arg.tensor.push_back(tensor_input_axis);
  tensor_input_axis_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg_out;
  tensor_arg_out.tensor.push_back(tensor_output);
  tensor_arg_out.arg_type = TA_SINGLE;
  TeOpParas opParas;
  opParas.inputs.push_back(tensor_arg);
  ge::TensorDesc tensorDesc1;
  tensorDesc1.SetDataType(ge::DT_INT32);
  auto ge_tensor = ge::Tensor(tensorDesc1);
  opParas.const_inputs["axes"] =
    std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)axis.data(), axis.size() * 4, ge_tensor);
  opParas.inputs.push_back(tensor_input_axis_arg);
  opParas.outputs.push_back(tensor_arg_out);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "REDUCEMEAN__COUNTER__4";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}
TEST_F(ReduceMeanTiling, ReduceTiling6) {
  using namespace optiling;
  std::cout << "ReduceMeanTiling ReduceTiling6" << std::endl;
  std::string op_name = "ReduceMean";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());


  std::string compileInfo = R"({"_idx_before_reduce": 0, "_pattern": "CommReduce", "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [-1], "_ub_info": [32512], "_ub_info_rf": [32512], "_reduce_shape_known": true, "_const_shape_post": true, "_compile_pattern": -1, "_block_dims": {"-1": 32}, "_atomic_flags": {"-1": false},"reduce_mean_cof_dtype":"float32"})";

  std::vector<int64_t> input{0, 10,3};
  std::vector<int64_t> output{0, 10,3};
  std::vector<int64_t> input_axis{1};
  std::vector<int64_t> axis{1};
  std::string in_dtype = "float32";

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = in_dtype;
  TeOpTensor tensor_input_axis;
  tensor_input_axis.shape = input_axis;
  tensor_input_axis.dtype = "int64";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = in_dtype;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_input);
  tensor_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_input_axis_arg;
  tensor_input_axis_arg.tensor.push_back(tensor_input_axis);
  tensor_input_axis_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg_out;
  tensor_arg_out.tensor.push_back(tensor_output);
  tensor_arg_out.arg_type = TA_SINGLE;
  TeOpParas opParas;
  opParas.inputs.push_back(tensor_arg);
  ge::TensorDesc tensorDesc1;
  tensorDesc1.SetDataType(ge::DT_INT64);
  auto ge_tensor = ge::Tensor(tensorDesc1);
  opParas.const_inputs["axes"] =
    std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)axis.data(), axis.size() * 8, ge_tensor);
  opParas.inputs.push_back(tensor_input_axis_arg);
  opParas.outputs.push_back(tensor_arg_out);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "REDUCEMEAN__COUNTER__6";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}
TEST_F(ReduceMeanTiling, ReduceTiling7) {
  using namespace optiling;
  std::cout << "ReduceMeanTiling ReduceTiling7" << std::endl;
  std::string op_name = "ReduceMean";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());


  std::string compileInfo = R"({"_idx_before_reduce": 0, "_pattern": "CommReduce", "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [-1], "_ub_info": [32512], "_ub_info_rf": [32512], "_reduce_shape_known": true, "_const_shape_post": true, "_compile_pattern": -1, "_block_dims": {"-1": 32}, "_atomic_flags": {"-1": false},"reduce_mean_cof_dtype":"float16"})";

  std::vector<int64_t> input{0, 10,3};
  std::vector<int64_t> output{0, 10,3};
  std::vector<int64_t> input_axis{1};
  std::vector<int64_t> axis{1};
  std::string in_dtype = "float16";

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = in_dtype;
  TeOpTensor tensor_input_axis;
  tensor_input_axis.shape = input_axis;
  tensor_input_axis.dtype = "int64";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = in_dtype;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_input);
  tensor_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_input_axis_arg;
  tensor_input_axis_arg.tensor.push_back(tensor_input_axis);
  tensor_input_axis_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg_out;
  tensor_arg_out.tensor.push_back(tensor_output);
  tensor_arg_out.arg_type = TA_SINGLE;
  TeOpParas opParas;
  opParas.inputs.push_back(tensor_arg);
  ge::TensorDesc tensorDesc1;
  tensorDesc1.SetDataType(ge::DT_INT64);
  auto ge_tensor = ge::Tensor(tensorDesc1);
  opParas.const_inputs["axes"] =
    std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)axis.data(), axis.size() * 8, ge_tensor);
  opParas.inputs.push_back(tensor_input_axis_arg);
  opParas.outputs.push_back(tensor_arg_out);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "REDUCEMEAN__COUNTER__7";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

