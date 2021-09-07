#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "register/op_tiling_registry.h"

using namespace std;

class LayerNormTiling : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "LayerNormTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "LayerNormTiling TearDown" << std::endl;
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

TEST_F(LayerNormTiling, LayerNorm_tiling_test_1) {
  using namespace optiling;
  std::string op_name = "LayerNorm";
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter !=
              optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  std::string compileInfo = R"({
                        "input_format": "NCHW",
                        "core_num": 32,
                        "begin_norm_axis":-1,
                        "begin_params_axis":-1,                        
                        "is_tik_support":true,
                        "tik_mode": "dynamic",                        
                        "ub_max_byte": 253952})";

  std::vector<std::vector<int64_t>> inputs{{11, 12, 512}, {512}, {512}};

  std::vector<std::vector<int64_t>> outputs{
      {11, 12, 512}, {11, 12, 1}, {11, 12, 1}};

  std::vector<std::string> input_types{"float16", "float16", "float16"};
  std::vector<std::string> output_types{"float16", "float16", "float16"};
  std::string data_format = "NCHW";

  TeOpParas opParas;
  for (size_t i = 0; i < inputs.size(); i++) {
    TeOpTensor tensor_input;
    TeOpTensorArg tensor_arg;
    tensor_input.shape = inputs[i];
    tensor_input.dtype = input_types[i];
    tensor_input.format = data_format;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensor_arg);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    TeOpTensor tensor_output;
    TeOpTensorArg tensor_arg;
    tensor_output.shape = outputs[i];
    tensor_output.dtype = output_types[i];
    tensor_output.format = data_format;
    tensor_arg.tensor.push_back(tensor_output);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensor_arg);
  }
  opParas.op_type = op_name;

  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "LayerNorm_tiling_test_1";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 32);
  EXPECT_EQ(to_string(runInfo.tiling_data), "0 132 512 27 5 2 989855744 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_2) {
  using namespace optiling;
  std::string op_name = "LayerNorm";
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter !=
              optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  std::string compileInfo = R"({
                        "begin_norm_axis":1,
                        "begin_params_axis":-1,
                        "is_tik_support":false,
                        "tik_mode": "dynamic",
                        "ub_max_byte": 253952,
                        "_attr_vars": {"1480005": [], "1540005": [], "2180005": [], "2240005": [], "390005": [],
                          "1480000": [], "1540000": [], "2180000": [], "2240000": [], "390000": [], "1480001": [], "1540001": [], "2180001": [], "2240001": [], "390001": [], "1480002": [], "1540002": [], "2180002": [], "2240002": [], "390002": []},
                        "_custom_vars": {
                        "1480005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1540005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2180005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2240005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "390005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1480000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1540000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2180000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2240000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "390000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1480001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1540001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2180001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2240001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "390001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1480002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1540002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2180002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2240002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "390002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "_normal_vars": {"1480005": [], "1540005": [], "2180005": [], "2240005": [], "390005": [],
                          "1480000": [], "1540000": [], "2180000": [], "2240000": [], "390000": [], "1480001": [], "1540001": [], "2180001": [], "2240001": [], "390001": [], "1480002": [], "1540002": [], "2180002": [], "2240002": [], "390002": []},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "1480005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1540005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2180005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2240005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "390005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1480000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "1540000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2180000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2240000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "390000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1480001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "1540001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2180001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2240001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "390001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1480002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "1540002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2180002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2240002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "390002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "common_info": [32, 1, 8, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "mode": "original",
                        "pattern_info": [39],
                        "reduce_axis": [1,2],
                        "input_format": "NCHW",
                        "reduce_mean_cof_dtype":"float32",
                        "is_support_vexp":true,
                        "ub_info":[16384]})";

  std::vector<std::vector<int64_t>> inputs{{1024, 30, 512}, {512}, {512}};

  std::vector<std::vector<int64_t>> outputs{
      {1024, 30, 512}, {1024, 1, 1}, {1024, 1, 1}};

  std::vector<std::string> input_types{"float32", "float32", "float32"};
  std::vector<std::string> output_types{"float32", "float32", "float32"};
  std::string data_format = "NCHW";

  TeOpParas opParas;
  for (size_t i = 0; i < inputs.size(); i++) {
    TeOpTensor tensor_input;
    TeOpTensorArg tensor_arg;
    tensor_input.shape = inputs[i];
    tensor_input.dtype = input_types[i];
    tensor_input.format = data_format;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensor_arg);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    TeOpTensor tensor_output;
    TeOpTensorArg tensor_arg;
    tensor_output.shape = outputs[i];
    tensor_output.dtype = output_types[i];
    tensor_output.format = data_format;
    tensor_arg.tensor.push_back(tensor_output);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensor_arg);
  }
  opParas.op_type = op_name;

  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "LayerNorm_tiling_test_2";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 32);
  EXPECT_EQ(runInfo.tiling_key, 1480001);
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "1024 30 512 948471945 32 1 20 23 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_3) {
  using namespace optiling;
  std::string op_name = "LayerNorm";
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter !=
              optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  std::string compileInfo = R"({
                        "input_format": "NCHW",
                        "core_num": 32,
                        "begin_norm_axis":1,
                        "begin_params_axis":1,
                        "is_tik_support":true,
                        "tik_mode": "dynamic",  
                        "ub_max_byte": 253952})";

  std::vector<std::vector<int64_t>> inputs{
      {34, 309, 512}, {309, 512}, {309, 512}};

  std::vector<std::vector<int64_t>> outputs{
      {34, 309, 512}, {34, 1, 1}, {34, 1, 1}};

  std::vector<std::string> input_types{"float16", "float16", "float16"};
  std::vector<std::string> output_types{"float16", "float16", "float16"};
  std::string data_format = "NCHW";

  TeOpParas opParas;
  for (size_t i = 0; i < inputs.size(); i++) {
    TeOpTensor tensor_input;
    TeOpTensorArg tensor_arg;
    tensor_input.shape = inputs[i];
    tensor_input.dtype = input_types[i];
    tensor_input.format = data_format;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensor_arg);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    TeOpTensor tensor_output;
    TeOpTensorArg tensor_arg;
    tensor_output.shape = outputs[i];
    tensor_output.dtype = output_types[i];
    tensor_output.format = data_format;
    tensor_arg.tensor.push_back(tensor_output);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensor_arg);
  }
  opParas.op_type = op_name;

  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "LayerNorm_tiling_test_3";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 32);
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 34 158208 17 2 2 919869235 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_4) {
  using namespace optiling;
  std::string op_name = "LayerNorm";
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter !=
              optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  std::string compileInfo = R"({
                        "begin_norm_axis":0,
                        "begin_params_axis":-1,                       
                        "is_tik_support":false,
                        "tik_mode": "dynamic",                        
                        "ub_max_byte": 253952,
                        "_attr_vars": {"1260005": [],"1320005": [],"1380005": [],"1960005": [],"2020005": [],"2080005": [],"2660005": [],"2720005": [],"2780005": [],"1260000": [],"1320000": [],"1380000": [],"1960000": [],"2020000": [],"2080000": [],"2660000": [],"2720000": [],"2780000": [], "1260001": [],"1320001": [],"1380001": [],"1960001": [],"2020001": [],"2080001": [],"2660001": [],"2720001": [],"2780001": [], "1260002": [],"1320002": [],"1380002": [],"1960002": [],"2020002": [],"2080002": [],"2660002": [],"2720002": [],"2780002": []},
                        "_custom_vars": {
                        "1260005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1320005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1380005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1960005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2020005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2080005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2660005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2720005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2780005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1260000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1320000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1380000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1960000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2020000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2080000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2660000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2720000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2780000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1260001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1320001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1380001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1960001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2020001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2080001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2660001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2720001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2780001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1260002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1320002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1380002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1960002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2020002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2080002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2660002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2720002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2780002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "_normal_vars": {"1260005": [],"1320005": [],"1380005": [],"1960005": [],"2020005": [],"2080005": [],"2660005": [],"2720005": [],"2780005": [],"1260000": [],"1320000": [],"1380000": [],"1960000": [],"2020000": [],"2080000": [],"2660000": [],"2720000": [],"2780000": [], "1260001": [],"1320001": [],"1380001": [],"1960001": [],"2020001": [],"2080001": [],"2660001": [],"2720001": [],"2780001": [], "1260002": [],"1320002": [],"1380002": [],"1960002": [],"2020002": [],"2080002": [],"2660002": [],"2720002": [],"2780002": []},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "1260005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1320005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1380005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1960005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2020005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2080005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2660005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2720005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2780005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1260000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1320000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1380000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1960000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2020000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2080000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2660000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2720000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2780000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1260001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1320001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1380001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1960001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2020001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2080001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2660001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2720001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2780001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1260002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1320002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1380002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1960002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2020002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2080002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2660002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2720002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "2780002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "common_info": [32, 1, 8, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "mode": "original",
                        "pattern_info": [63],
                        "reduce_axis": [0,1,2],
                        "reduce_mean_cof_dtype":"float32",
                        "is_support_vexp":true,
                        "input_format": "NCHW",
                        "ub_info":[16384]})";

  std::vector<std::vector<int64_t>> inputs{{34, 309, 512}, {512}, {512}};

  std::vector<std::vector<int64_t>> outputs{
      {34, 309, 512}, {1, 1, 1}, {1, 1, 1}};

  std::vector<std::string> input_types{"float32", "float32", "float32"};
  std::vector<std::string> output_types{"float32", "float32", "float32"};
  std::string data_format = "NCHW";

  TeOpParas opParas;
  for (size_t i = 0; i < inputs.size(); i++) {
    TeOpTensor tensor_input;
    TeOpTensorArg tensor_arg;
    tensor_input.shape = inputs[i];
    tensor_input.dtype = input_types[i];
    tensor_input.format = data_format;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensor_arg);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    TeOpTensor tensor_output;
    TeOpTensorArg tensor_arg;
    tensor_output.shape = outputs[i];
    tensor_output.dtype = output_types[i];
    tensor_output.format = data_format;
    tensor_arg.tensor.push_back(tensor_output);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensor_arg);
  }
  opParas.op_type = op_name;

  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "LayerNorm_tiling_test_4";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 1);
  EXPECT_EQ(runInfo.tiling_key, 2020001);
  EXPECT_EQ(to_string(runInfo.tiling_data), "34 309 512 877108573 34 1 20 34 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_5) {
  using namespace optiling;
  std::string op_name = "LayerNorm";
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter !=
              optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  std::string compileInfo = R"({
                        "begin_norm_axis":1,
                        "begin_params_axis":-1,                        
                        "is_tik_support":false,
                        "tik_mode": "dynamic",                        
                        "ub_max_byte": 253952,
                        "_attr_vars": {"1480005": [], "1540005": [], "2180005": [], "2240005": [], "390005": [],"1480000": [], "1540000": [], "2180000": [], "2240000": [], "390000": [], "1480001": [], "1540001": [], "2180001": [], "2240001": [], "390001": [], "1480002": [], "1540002": [], "2180002": [], "2240002": [], "390002": []},
                        "_custom_vars": {
                        "1480005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "1540005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2180005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2240005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "390005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1480000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "1540000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2180000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2240000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "390000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1480001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "1540001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2180001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2240001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "390001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1480002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "1540002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2180002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2240002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "390002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "_normal_vars": {"1480005": [], "1540005": [], "2180005": [], "2240005": [], "390005": [],"1480000": [], "1540000": [], "2180000": [], "2240000": [], "390000": [], "1480001": [], "1540001": [], "2180001": [], "2240001": [], "390001": [], "1480002": [], "1540002": [], "2180002": [], "2240002": [], "390002": []},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "1480005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "1540005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2180005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2240005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "390005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1480000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "1540000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2180000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2240000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "390000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1480001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "1540001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2180001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2240001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "390001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1480002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "1540002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2180002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2240002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "390002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "common_info": [32, 1, 8, 0],
                        "common_info": [32, 1, 8, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "mode": "original",
                        "pattern_info": [39],
                        "reduce_axis": [1,2],
                        "reduce_mean_cof_dtype":"float32",
                        "is_support_vexp":true,
                        "input_format": "NCHW",
                        "ub_info":[16384]})";

  std::vector<std::vector<int64_t>> inputs{{34, 309, 512}, {512}, {512}};

  std::vector<std::vector<int64_t>> outputs{
      {34, 309, 512}, {34, 1, 1}, {34, 1, 1}};

  std::vector<std::string> input_types{"float32", "float32", "float32"};
  std::vector<std::string> output_types{"float32", "float32", "float32"};
  std::string data_format = "NCHW";

  TeOpParas opParas;
  for (size_t i = 0; i < inputs.size(); i++) {
    TeOpTensor tensor_input;
    TeOpTensorArg tensor_arg;
    tensor_input.shape = inputs[i];
    tensor_input.dtype = input_types[i];
    tensor_input.format = data_format;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensor_arg);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    TeOpTensor tensor_output;
    TeOpTensorArg tensor_arg;
    tensor_output.shape = outputs[i];
    tensor_output.dtype = output_types[i];
    tensor_output.format = data_format;
    tensor_arg.tensor.push_back(tensor_output);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensor_arg);
  }
  opParas.op_type = op_name;

  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "LayerNorm_tiling_test_5";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 5);
  EXPECT_EQ(runInfo.tiling_key, 1480001);
  EXPECT_EQ(to_string(runInfo.tiling_data), "34 309 512 919869235 8 1 20 8 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_6) {
  using namespace optiling;
  std::string op_name = "LayerNorm";
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter !=
              optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  std::string compileInfo = R"({
                        "begin_norm_axis": -1,
                        "begin_params_axis":-1,                        
                        "is_tik_support":false,
                        "tik_mode": "dynamic",                        
                        "ub_max_byte": 253952,
                        "_attr_vars": {"1940005": [], "270005": [], "670005": [], "671005": [], "1940000": [], "270000": [], "670000": [], "671000": [], "1940001": [], "270001": [], "670001": [], "671001": [], "1940002": [], "270002": [], "670002": [], "671002": []},
                        "_custom_vars": {
                        "1940005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "_normal_vars": {"1940005": [], "270005": [], "670005": [], "671005": [], "1940000": [], "270000": [], "670000": [], "671000": [], "1940001": [], "270001": [], "670001": [], "671001": [], "1940002": [], "270002": [], "670002": [], "671002": []},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "1940005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "common_info": [32, 1, 16, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "mode": "original",
                        "pattern_info": [27],
                        "reduce_axis": [2],
                        "reduce_mean_cof_dtype":"float32",
                        "is_support_vexp":true,
                        "input_format": "NCHW",
                        "ub_info":[16384]})";

  std::vector<std::vector<int64_t>> inputs{{20, 304, 512}, {512}, {512}};

  std::vector<std::vector<int64_t>> outputs{
      {20, 304, 512}, {20, 304, 1}, {20, 304, 1}};

  std::vector<std::string> input_types{"float16", "float16", "float16"};
  std::vector<std::string> output_types{"float16", "float16", "float16"};
  std::string data_format = "NCHW";

  TeOpParas opParas;
  for (size_t i = 0; i < inputs.size(); i++) {
    TeOpTensor tensor_input;
    TeOpTensorArg tensor_arg;
    tensor_input.shape = inputs[i];
    tensor_input.dtype = input_types[i];
    tensor_input.format = data_format;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensor_arg);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    TeOpTensor tensor_output;
    TeOpTensorArg tensor_arg;
    tensor_output.shape = outputs[i];
    tensor_output.dtype = output_types[i];
    tensor_output.format = data_format;
    tensor_arg.tensor.push_back(tensor_output);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensor_arg);
  }
  opParas.op_type = op_name;

  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "LayerNorm_tiling_test_6";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 32);
  EXPECT_EQ(runInfo.tiling_key, 671001);
  EXPECT_EQ(to_string(runInfo.tiling_data), "20 304 512 989855744 2 16 19 0 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_7) {
  using namespace optiling;
  std::string op_name = "LayerNorm";
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter !=
              optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  std::string compileInfo = R"({
                        "begin_norm_axis":-1,
                        "begin_params_axis":-1,                        
                        "is_tik_support":false,
                        "tik_mode": "dynamic",                        
                        "ub_max_byte": 253952,
                        "_attr_vars": {"1940005": [], "270005": [], "670005": [], "671005": [], "1940000": [], "270000": [], "670000": [], "671000": [], "1940001": [], "270001": [], "670001": [], "671001": [], "1940002": [], "270002": [], "670002": [], "671002": []},
                        "_custom_vars": {
                        "1940005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "_normal_vars": {"1940005": [], "270005": [], "670005": [], "671005": [], "1940000": [], "270000": [], "670000": [], "671000": [], "1940001": [], "270001": [], "670001": [], "671001": [], "1940002": [], "270002": [], "670002": [], "671002": []},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "1940005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "common_info": [32, 1, 16, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "mode": "original",
                        "pattern_info": [27],
                        "reduce_axis": [2],
                        "reduce_mean_cof_dtype":"float32",
                        "is_support_vexp":true,
                        "input_format": "NCHW",
                        "ub_info":[16384]})";

  std::vector<std::vector<int64_t>> inputs{{49, 304, 512}, {512}, {512}};

  std::vector<std::vector<int64_t>> outputs{
      {49, 304, 512}, {49, 304, 1}, {49, 304, 1}};

  std::vector<std::string> input_types{"float16", "float16", "float16"};
  std::vector<std::string> output_types{"float16", "float16", "float16"};
  std::string data_format = "NCHW";

  TeOpParas opParas;
  for (size_t i = 0; i < inputs.size(); i++) {
    TeOpTensor tensor_input;
    TeOpTensorArg tensor_arg;
    tensor_input.shape = inputs[i];
    tensor_input.dtype = input_types[i];
    tensor_input.format = data_format;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensor_arg);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    TeOpTensor tensor_output;
    TeOpTensorArg tensor_arg;
    tensor_output.shape = outputs[i];
    tensor_output.dtype = output_types[i];
    tensor_output.format = data_format;
    tensor_arg.tensor.push_back(tensor_output);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensor_arg);
  }
  opParas.op_type = op_name;

  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "LayerNorm_tiling_test_7";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 784);
  EXPECT_EQ(runInfo.tiling_key, 671001);
  EXPECT_EQ(to_string(runInfo.tiling_data), "49 304 512 989855744 49 16 19 0 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_8) {
  using namespace optiling;
  std::string op_name = "LayerNorm";
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter !=
              optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  std::string compileInfo = R"({
                        "begin_norm_axis": -1,
                        "begin_params_axis":-1,                        
                        "is_tik_support":false,
                        "tik_mode": "dynamic",                        
                        "ub_max_byte": 253952,
                        "_attr_vars": {"1940005": [], "270005": [], "670005": [], "671005": [], "1940000": [], "270000": [], "670000": [], "671000": [], "1940001": [], "270001": [], "670001": [], "671001": [], "1940002": [], "270002": [], "670002": [], "671002": []},
                        "_custom_vars": {
                        "1940005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "_normal_vars": {"1940005": [], "270005": [], "670005": [], "671005": [], "1940000": [], "270000": [], "670000": [], "671000": [], "1940001": [], "270001": [], "670001": [], "671001": [], "1940002": [], "270002": [], "670002": [], "671002": []},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "1940005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "common_info": [32, 1, 8, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "mode": "original",
                        "pattern_info": [27],
                        "reduce_axis": [2],
                        "reduce_mean_cof_dtype":"float32",
                        "is_support_vexp":true,
                        "input_format": "NCHW",
                        "ub_info":[16384]})";

  std::vector<std::vector<int64_t>> inputs{{34, 309, 512}, {512}, {512}};

  std::vector<std::vector<int64_t>> outputs{
      {34, 304, 512}, {34, 304, 1}, {34, 304, 1}};

  std::vector<std::string> input_types{"float32", "float32", "float32"};
  std::vector<std::string> output_types{"float32", "float32", "float32"};
  std::string data_format = "NCHW";

  TeOpParas opParas;
  for (size_t i = 0; i < inputs.size(); i++) {
    TeOpTensor tensor_input;
    TeOpTensorArg tensor_arg;
    tensor_input.shape = inputs[i];
    tensor_input.dtype = input_types[i];
    tensor_input.format = data_format;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensor_arg);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    TeOpTensor tensor_output;
    TeOpTensorArg tensor_arg;
    tensor_output.shape = outputs[i];
    tensor_output.dtype = output_types[i];
    tensor_output.format = data_format;
    tensor_arg.tensor.push_back(tensor_output);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensor_arg);
  }
  opParas.op_type = op_name;

  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "LayerNorm_tiling_test_8";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 32);
  EXPECT_EQ(runInfo.tiling_key, 671001);
  EXPECT_EQ(to_string(runInfo.tiling_data), "34 309 512 989855744 2 16 20 0 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_9) {
  using namespace optiling;
  std::string op_name = "LayerNorm";
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter !=
              optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  std::string compileInfo = R"({
                        "begin_norm_axis":-1,
                        "begin_params_axis":-1,                        
                        "is_tik_support":false,
                        "tik_mode": "dynamic",                        
                        "ub_max_byte": 253952,
                        "_attr_vars": {"1940005": [], "270005": [], "670005": [], "671005": [], "1940000": [], "270000": [], "670000": [], "671000": [], "1940001": [], "270001": [], "670001": [], "671001": [], "1940002": [], "270002": [], "670002": [], "671002": []},
                        "_custom_vars": {
                        "1940005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "_normal_vars": {"1940005": [], "270005": [], "670005": [], "671005": [], "1940000": [], "270000": [], "670000": [], "671000": [], "1940001": [], "270001": [], "670001": [], "671001": [], "1940002": [], "270002": [], "670002": [], "671002": []},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "1940005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "common_info": [32, 1, 16, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "mode": "original",
                        "pattern_info": [27],
                        "reduce_axis": [2],
                        "reduce_mean_cof_dtype":"float32",
                        "is_support_vexp":true,
                        "input_format": "NCHW",
                        "ub_info":[16384]})";

  std::vector<std::vector<int64_t>> inputs{{352, 4, 512}, {512}, {512}};

  std::vector<std::vector<int64_t>> outputs{
      {352, 4, 512}, {352, 4, 1}, {352, 4, 1}};

  std::vector<std::string> input_types{"float16", "float16", "float16"};
  std::vector<std::string> output_types{"float16", "float16", "float16"};
  std::string data_format = "NCHW";

  TeOpParas opParas;
  for (size_t i = 0; i < inputs.size(); i++) {
    TeOpTensor tensor_input;
    TeOpTensorArg tensor_arg;
    tensor_input.shape = inputs[i];
    tensor_input.dtype = input_types[i];
    tensor_input.format = data_format;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensor_arg);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    TeOpTensor tensor_output;
    TeOpTensorArg tensor_arg;
    tensor_output.shape = outputs[i];
    tensor_output.dtype = output_types[i];
    tensor_output.format = data_format;
    tensor_arg.tensor.push_back(tensor_output);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensor_arg);
  }
  opParas.op_type = op_name;

  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "LayerNorm_tiling_test_9";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 1);
  EXPECT_EQ(runInfo.tiling_key, 270001);
  EXPECT_EQ(to_string(runInfo.tiling_data), "352 4 512 989855744 352 1 5 0 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_10) {
  using namespace optiling;
  std::string op_name = "LayerNorm";
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter !=
              optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  std::string compileInfo = R"({
                        "begin_norm_axis":-1,
                        "begin_params_axis":-1,                        
                        "is_tik_support":false,
                        "tik_mode": "dynamic",                       
                        "ub_max_byte": 253952,
                        "_attr_vars": {"1940005": [], "270005": [], "670005": [], "671005": [],"1940000": [], "270000": [], "670000": [], "671000": [], "1940001": [], "270001": [], "670001": [], "671001": [], "1940002": [], "270002": [], "670002": [], "671002": []},
                        "_custom_vars": {
                        "1940005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "_normal_vars": {"1940005": [], "270005": [], "670005": [], "671005": [],"1940000": [], "270000": [], "670000": [], "671000": [], "1940001": [], "270001": [], "670001": [], "671001": [], "1940002": [], "270002": [], "670002": [], "671002": []},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "1940005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671005": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671001": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "1940002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "671002": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "common_info": [32, 1, 16, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "mode": "original",
                        "pattern_info": [27],
                        "reduce_axis": [2],
                        "reduce_mean_cof_dtype":"float32",
                        "is_support_vexp":true,
                        "input_format": "NCHW",
                        "ub_info":[16384]})";

  std::vector<std::vector<int64_t>> inputs{{32, 121, 768}, {768}, {768}};

  std::vector<std::vector<int64_t>> outputs{
      {32, 121, 768}, {32, 121, 1}, {32, 121, 768}};

  std::vector<std::string> input_types{"float16", "float16", "float16"};
  std::vector<std::string> output_types{"float16", "float16", "float16"};
  std::string data_format = "NCHW";

  TeOpParas opParas;
  for (size_t i = 0; i < inputs.size(); i++) {
    TeOpTensor tensor_input;
    TeOpTensorArg tensor_arg;
    tensor_input.shape = inputs[i];
    tensor_input.dtype = input_types[i];
    tensor_input.format = data_format;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensor_arg);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    TeOpTensor tensor_output;
    TeOpTensorArg tensor_arg;
    tensor_output.shape = outputs[i];
    tensor_output.dtype = output_types[i];
    tensor_output.format = data_format;
    tensor_arg.tensor.push_back(tensor_output);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensor_arg);
  }
  opParas.op_type = op_name;

  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "LayerNorm_tiling_test_10";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 32);
  EXPECT_EQ(runInfo.tiling_key, 1940001);
  EXPECT_EQ(to_string(runInfo.tiling_data), "32 121 768 984263339 1 1 768 1 ");
}

// static shape case
TEST_F(LayerNormTiling, LayerNorm_tiling_test_11) {
  using namespace optiling;
  std::string op_name = "LayerNorm";
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter !=
              optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  std::string compileInfo = R"({
                        "begin_norm_axis":-1,
                        "begin_params_axis":-1,                        
                        "is_tik_support":false,
                        "tik_mode": "dynamic",                        
                        "ub_max_byte": 253952,
                        "_attr_vars": {"671001": []},
                        "_custom_vars": {
                        "671001": ["dim0_0", "dim0_1", "dim0_2"]},
                        "_normal_vars": {"671001": []},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "671001": ["dim0_0", "dim0_1", "dim0_2"]},
                        "common_info": [32, 1, 16, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "mode": "const",
                        "pattern_info": [27],
                        "reduce_axis": [2],
                        "input_format": "NCHW",
                        "is_support_vexp":true,
                        "ub_info":[16384]})";

  std::vector<std::vector<int64_t>> inputs{{20, 304, 512}, {512}, {512}};

  std::vector<std::vector<int64_t>> outputs{
      {20, 304, 512}, {20, 304, 1}, {20, 304, 1}};

  std::vector<std::string> input_types{"float16", "float16", "float16"};
  std::vector<std::string> output_types{"float16", "float16", "float16"};
  std::string data_format = "NCHW";

  TeOpParas opParas;
  for (size_t i = 0; i < inputs.size(); i++) {
    TeOpTensor tensor_input;
    TeOpTensorArg tensor_arg;
    tensor_input.shape = inputs[i];
    tensor_input.dtype = input_types[i];
    tensor_input.format = data_format;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensor_arg);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    TeOpTensor tensor_output;
    TeOpTensorArg tensor_arg;
    tensor_output.shape = outputs[i];
    tensor_output.dtype = output_types[i];
    tensor_output.format = data_format;
    tensor_arg.tensor.push_back(tensor_output);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensor_arg);
  }
  opParas.op_type = op_name;

  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "LayerNorm_tiling_test_11";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 32);
  EXPECT_EQ(runInfo.tiling_key, 671001);
  EXPECT_EQ(to_string(runInfo.tiling_data), "20 304 512 0 1 1 0 2 16 19 0 1 ");
}

// NZ case
TEST_F(LayerNormTiling, LayerNorm_tiling_test_12) {
  using namespace optiling;
  std::string op_name = "LayerNorm";
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter !=
              optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  std::string compileInfo = R"({
                        "begin_norm_axis":-1,
                        "begin_params_axis":-1,                        
                        "is_tik_support":false,
                        "tik_mode": "dynamic",                        
                        "ub_max_byte": 253952,
                        "_attr_vars": {"898000": [], "9040000": [], "11080000": [], "11140000": [], "5390000": [], "5791000": [], "5792000": []},
                        "_custom_vars": {
                        "898000": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "9040000": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "11080000": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "11140000": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "5390000": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "5791000": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "5792000": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "_normal_vars": {"898000": [], "9040000": [], "11080000": [], "11140000": [], "5390000": [], "5791000": [], "5792000": []},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "898000": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "9040000": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "11080000": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "11140000": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"],
                        "5390000": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "5791000": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "5792000": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "common_info": [32, 1, 16, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "mode": "original",
                        "pattern_info": [99],
                        "reduce_axis": [0, 3],
                        "reduce_mean_cof_dtype":"float32",
                        "is_support_vexp":true,
                        "input_format": "FRACTAL_NZ",
                        "ub_info":[16384]})";

  std::vector<std::vector<int64_t>> inputs{{32, 32, 16, 16}, {512}, {512}};

  std::vector<std::vector<int64_t>> outputs{
      {32, 32, 16, 16}, {512, 1}, {512, 1}};

  std::vector<std::string> input_types{"float16", "float16", "float16"};
  std::vector<std::string> output_types{"float16", "float16", "float16"};
  std::string data_format = "FRACTAL_NZ";

  TeOpParas opParas;
  for (size_t i = 0; i < inputs.size(); i++) {
    TeOpTensor tensor_input;
    TeOpTensorArg tensor_arg;
    tensor_input.shape = inputs[i];
    tensor_input.dtype = input_types[i];
    tensor_input.format = data_format;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensor_arg);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    TeOpTensor tensor_output;
    TeOpTensorArg tensor_arg;
    tensor_output.shape = outputs[i];
    tensor_output.dtype = output_types[i];
    tensor_output.format = data_format;
    tensor_arg.tensor.push_back(tensor_output);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensor_arg);
  }
  opParas.op_type = op_name;

  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "LayerNorm_tiling_test_12";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 32);
  EXPECT_EQ(runInfo.tiling_key, 5390000);
  EXPECT_EQ(to_string(runInfo.tiling_data), "32 32 16 16 989855744 1 1 1 0 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_13) {
  using namespace optiling;
  // tik case
  std::string op_name = "LayerNorm";
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter !=
              optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  std::string compileInfo = R"({
                        "input_format": "NCHW",
                        "core_num": 32,
                        "begin_norm_axis":-1,
                        "begin_params_axis":-1,                       
                        "is_tik_support":true,
                        "tik_mode": "const",                       
                        "ub_max_byte": 253952})";

  std::vector<std::vector<int64_t>> inputs{{11, 12, 512}, {512}, {512}};

  std::vector<std::vector<int64_t>> outputs{
      {11, 12, 512}, {11, 12, 1}, {11, 12, 1}};

  std::vector<std::string> input_types{"float16", "float16", "float16"};
  std::vector<std::string> output_types{"float16", "float16", "float16"};
  std::string data_format = "NCHW";

  TeOpParas opParas;
  for (size_t i = 0; i < inputs.size(); i++) {
    TeOpTensor tensor_input;
    TeOpTensorArg tensor_arg;
    tensor_input.shape = inputs[i];
    tensor_input.dtype = input_types[i];
    tensor_input.format = data_format;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensor_arg);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    TeOpTensor tensor_output;
    TeOpTensorArg tensor_arg;
    tensor_output.shape = outputs[i];
    tensor_output.dtype = output_types[i];
    tensor_output.format = data_format;
    tensor_arg.tensor.push_back(tensor_output);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensor_arg);
  }
  opParas.op_type = op_name;

  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "LayerNorm_tiling_test_13";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 27);
  EXPECT_EQ(to_string(runInfo.tiling_data), "0 132 512 27 5 2 989855744 ");
}