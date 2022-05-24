#include <stdlib.h>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "common/utils/ut_op_util.h"
#include "common_unittest.h"
#include "array_ops.h"
#include "rnn.h"
#include "dynamic_rnn_v3.h"
using namespace std;
using namespace ut_util;
using namespace ge;

class DynamicRnnV3Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DynamicRnnV3Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DynamicRnnV3Tiling TearDown" << std::endl;
  }
};

/*
 * be careful of the to_string fuction
 * the type of tiling_data in other ops is int64 while int32 here
 */
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

TEST_F(DynamicRnnV3Tiling, dynamic_rnn_v3_tiling_0) {
  using namespace optiling;
  std::string op_name = "DynamicRNNV3";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("DynamicRNNV3");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"tune_shape_list\": [[32,128,0],[64,64,1]]}}";

  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
      {32,32,8,16,16},
      {40,32,16,16},
      {512},
  };

  vector<std::string> dtypes = {"float16", "float16", "float16"};
  for (size_t i = 0; i < input_shapes.size(); i++) {
    tensorInputs.tensor.clear();
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TensorArgType::TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);
  }

  vector<vector<int64_t>> output_shapes = {
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
  };

  vector<std::string> out_dtypes = {"float16", "float16", "float16", "float16",
                           "float16", "float16", "float16", "float16"};
  for (size_t i = 0; i < output_shapes.size(); i++) {
    tensorOutputsArg.tensor.clear();
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = out_dtypes[i];
    tensorOutputsArg.tensor.push_back(tensorOutput);
    tensorOutputsArg.arg_type = TensorArgType::TA_SINGLE;
    opParas.outputs.push_back(tensorOutputsArg);
  }

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456a001";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "32 8 0 ");
}

TEST_F(DynamicRnnV3Tiling, dynamic_rnn_v3_tiling_1) {
  using namespace optiling;
  std::string op_name = "DynamicRNNV3";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("DynamicRNNV3");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"tune_shape_list\": [[-1, -1, 0], [99, 99, 1]]}}";

  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
      {32,32,8,16,16},
      {40,32,16,16},
      {512},
  };

  vector<std::string> dtypes = {"float16", "float16", "float16"};
  for (size_t i = 0; i < input_shapes.size(); i++) {
    tensorInputs.tensor.clear();
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TensorArgType::TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);
  }

  vector<vector<int64_t>> output_shapes = {
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
  };

  vector<std::string> out_dtypes = {"float16", "float16", "float16", "float16",
                           "float16", "float16", "float16", "float16"};
  for (size_t i = 0; i < output_shapes.size(); i++) {
    tensorOutputsArg.tensor.clear();
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = out_dtypes[i];
    tensorOutputsArg.tensor.push_back(tensorOutput);
    tensorOutputsArg.arg_type = TensorArgType::TA_SINGLE;
    opParas.outputs.push_back(tensorOutputsArg);
  }

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456a002";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "32 8 0 ");
}

TEST_F(DynamicRnnV3Tiling, dynamic_rnn_v3_tiling_2) {
  using namespace optiling;
  std::string op_name = "DynamicRNNV3";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("DynamicRNNV3");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"tune_shape_list\": [[32,8,0]]}}";

  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
      {8,32},
      {8,32},
      {512},
  };

  vector<std::string> dtypes = {"float16", "float16", "float16"};
  for (size_t i = 0; i < input_shapes.size(); i++) {
    tensorInputs.tensor.clear();
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TensorArgType::TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);
  }

  vector<vector<int64_t>> output_shapes = {
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
  };

  vector<std::string> out_dtypes = {"float16", "float16", "float16", "float16",
                           "float16", "float16", "float16", "float16"};
  for (size_t i = 0; i < output_shapes.size(); i++) {
    tensorOutputsArg.tensor.clear();
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = out_dtypes[i];
    tensorOutputsArg.tensor.push_back(tensorOutput);
    tensorOutputsArg.arg_type = TensorArgType::TA_SINGLE;
    opParas.outputs.push_back(tensorOutputsArg);
  }

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456a003";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}

TEST_F(DynamicRnnV3Tiling, dynamic_rnn_v3_tiling_3) {
  using namespace optiling;
  std::string op_name = "DynamicRNNV3";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("DynamicRNNV3");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"tune_shape_list\": [[32,444,0]]}}";

  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
    {32,32,8,16,16},
    {40,32,16,16},
    {512},
  };

  vector<std::string> dtypes = {"float16", "float16", "float16"};
  for (size_t i = 0; i < input_shapes.size(); i++) {
    tensorInputs.tensor.clear();
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TensorArgType::TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);
  }

  vector<vector<int64_t>> output_shapes = {
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
  };

  vector<std::string> out_dtypes = {"float16", "float16", "float16", "float16",
                           "float16", "float16", "float16", "float16"};
  for (size_t i = 0; i < output_shapes.size(); i++) {
    tensorOutputsArg.tensor.clear();
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = out_dtypes[i];
    tensorOutputsArg.tensor.push_back(tensorOutput);
    tensorOutputsArg.arg_type = TensorArgType::TA_SINGLE;
    opParas.outputs.push_back(tensorOutputsArg);
  }

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456a004";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}

TEST_F(DynamicRnnV3Tiling, dynamic_rnn_v3_tiling_4) {
  using namespace optiling;
  std::string op_name = "DynamicRNNV3";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("DynamicRNNV3");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"tune_shape_list\": [[32,6]]}}";

  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
    {32,32,8,16,16},
  };

  vector<std::string> dtypes = {"float16", "float16", "float16"};
  for (size_t i = 0; i < input_shapes.size(); i++) {
    tensorInputs.tensor.clear();
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TensorArgType::TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);
  }

  vector<vector<int64_t>> output_shapes = {
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
  };

  vector<std::string> out_dtypes = {"float16", "float16", "float16", "float16",
                           "float16", "float16", "float16", "float16"};
  for (size_t i = 0; i < output_shapes.size(); i++) {
    tensorOutputsArg.tensor.clear();
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = out_dtypes[i];
    tensorOutputsArg.tensor.push_back(tensorOutput);
    tensorOutputsArg.arg_type = TensorArgType::TA_SINGLE;
    opParas.outputs.push_back(tensorOutputsArg);
  }

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456a005";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}

TEST_F(DynamicRnnV3Tiling, dynamic_rnn_v3_tiling_5) {
  using namespace optiling;
  std::string op_name = "DynamicRNNV3";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("DynamicRNNV3");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"tune_shape_list\": [[32, 8, 0]]}}";

  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
      {-1,32,-1,16,16},
      {40,32,16,16},
      {512},
  };

  vector<std::string> dtypes = {"float16", "float16", "float16"};
  for (size_t i = 0; i < input_shapes.size(); i++) {
    tensorInputs.tensor.clear();
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TensorArgType::TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);
  }

  vector<vector<int64_t>> output_shapes = {
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
  };

  vector<std::string> out_dtypes = {"float16", "float16", "float16", "float16",
                           "float16", "float16", "float16", "float16"};
  for (size_t i = 0; i < output_shapes.size(); i++) {
    tensorOutputsArg.tensor.clear();
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = out_dtypes[i];
    tensorOutputsArg.tensor.push_back(tensorOutput);
    tensorOutputsArg.arg_type = TensorArgType::TA_SINGLE;
    opParas.outputs.push_back(tensorOutputsArg);
  }

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456a006";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}

TEST_F(DynamicRnnV3Tiling, dynamic_rnn_v3_tiling_6) {
  using namespace optiling;
  std::string op_name = "DynamicRNNV3";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("DynamicRNNV3");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"tune_shape_list\": [[32,444]]}}";

  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
    {32,32,8,16,16},
    {40,32,16,16},
    {512},
  };

  vector<std::string> dtypes = {"float16", "float16", "float16"};
  for (size_t i = 0; i < input_shapes.size(); i++) {
    tensorInputs.tensor.clear();
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TensorArgType::TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);
  }

  vector<vector<int64_t>> output_shapes = {
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
  };

  vector<std::string> out_dtypes = {"float16", "float16", "float16", "float16",
                           "float16", "float16", "float16", "float16"};
  for (size_t i = 0; i < output_shapes.size(); i++) {
    tensorOutputsArg.tensor.clear();
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = out_dtypes[i];
    tensorOutputsArg.tensor.push_back(tensorOutput);
    tensorOutputsArg.arg_type = TensorArgType::TA_SINGLE;
    opParas.outputs.push_back(tensorOutputsArg);
  }

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456a007";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}

#include "test_common.h"
TEST_F(DynamicRnnV3Tiling, dynamic_rnn_v3_tiling_7) {
  std::string op_name = "DynamicRNNV3";
  std::string compileInfo = "{\"vars\": {\"tune_shape_list\": [[-1, -1, 0], [99, 99, 1]]}}";

  vector<vector<int64_t>> v3_input_shapes = {
      {32,32,8,16,16},
      {40,32,16,16},
      {512},
  };

  vector<int64_t> v3_output_shape = {32,8,8,16,16};

  auto opParas = op::DynamicRNNV3(op_name);

  TENSOR_INPUT_WITH_SHAPE(opParas, x, v3_input_shapes[0], DT_FLOAT16, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, w, v3_input_shapes[1], DT_FLOAT16, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, b, v3_input_shapes[2], DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output_h, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output_c, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, i, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, j, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, f, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, o, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, tanhc, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});

  optiling::DynamicRNNV3CompileInfo info;
  int32_t tiling_len = sizeof(optiling::DynamicRnnV3TilingData);
  TILING_PARSE_JSON_TO_COMPILEINFO("DynamicRNNV3", compileInfo, info);

  ATTACH_OPERATOR_TO_HOLDER(holder, opParas, tiling_len, info);
  HOLDER_DO_TILING(holder, "DynamicRNNV3", ge::GRAPH_SUCCESS);
  TILING_DATA_VERIFY_BYTYPE(holder, int32_t, "32 8 0 ");
}

TEST_F(DynamicRnnV3Tiling, dynamic_rnn_v3_tiling_8) {
  std::string op_name = "DynamicRNNV3";
  std::string compileInfo = "{\"vars\": {\"tune_shape_list\": [[32, 120, 0], [99, 99, 1]]}}";

  vector<vector<int64_t>> v3_input_shapes = {
      {32,32,8,16,16},
      {40,32,16,16},
      {512},
  };

  vector<int64_t> v3_output_shape = {32,8,8,16,16};

  auto opParas = op::DynamicRNNV3(op_name);

  TENSOR_INPUT_WITH_SHAPE(opParas, x, v3_input_shapes[0], DT_FLOAT16, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, w, v3_input_shapes[1], DT_FLOAT16, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, b, v3_input_shapes[2], DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output_h, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output_c, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, i, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, j, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, f, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, o, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, tanhc, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});

  optiling::DynamicRNNV3CompileInfo info;
  int32_t tiling_len = sizeof(optiling::DynamicRnnV3TilingData);
  TILING_PARSE_JSON_TO_COMPILEINFO("DynamicRNNV3", compileInfo, info);

  ATTACH_OPERATOR_TO_HOLDER(holder, opParas, tiling_len, info);
  HOLDER_DO_TILING(holder, "DynamicRNNV3", ge::GRAPH_SUCCESS);
  TILING_DATA_VERIFY_BYTYPE(holder, int32_t, "32 8 0 ");
}

TEST_F(DynamicRnnV3Tiling, dynamic_rnn_v3_tiling_9) {
  std::string op_name = "DynamicRNNV3";
  std::string compileInfo = "{\"vars\": {\"tune_shape_list\": [[98, 99, 0], [99, 99, 1]]}}";

  vector<vector<int64_t>> v3_input_shapes = {
      {32,32,8,16,16},
      {40,32,16,16},
      {512},
  };

  vector<int64_t> v3_output_shape = {32,8,8,16,16};

  auto opParas = op::DynamicRNNV3(op_name);

  TENSOR_INPUT_WITH_SHAPE(opParas, x, v3_input_shapes[0], DT_FLOAT16, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, w, v3_input_shapes[1], DT_FLOAT16, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, b, v3_input_shapes[2], DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output_h, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output_c, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, i, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, j, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, f, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, o, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, tanhc, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});

  optiling::DynamicRNNV3CompileInfo info;
  int32_t tiling_len = sizeof(optiling::DynamicRnnV3TilingData);
  TILING_PARSE_JSON_TO_COMPILEINFO("DynamicRNNV3", compileInfo, info);

  ATTACH_OPERATOR_TO_HOLDER(holder, opParas, tiling_len, info);
  HOLDER_DO_TILING(holder, "DynamicRNNV3", ge::GRAPH_FAILED);
}

TEST_F(DynamicRnnV3Tiling, dynamic_rnn_v3_tiling_10) {
  std::string op_name = "DynamicRNNV3";
  std::string compileInfo = "{\"vars\": {\"tune_shape_list\": [[98, 99], [99, 99, 1]]}}";

  vector<vector<int64_t>> v3_input_shapes = {
      {32,32,8,16,16},
      {40,32,16,16},
      {512},
  };

  vector<int64_t> v3_output_shape = {32,8,8,16,16};

  auto opParas = op::DynamicRNNV3(op_name);

  TENSOR_INPUT_WITH_SHAPE(opParas, x, v3_input_shapes[0], DT_FLOAT16, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, w, v3_input_shapes[1], DT_FLOAT16, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, b, v3_input_shapes[2], DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output_h, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output_c, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, i, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, j, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, f, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, o, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, tanhc, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});

  optiling::DynamicRNNV3CompileInfo info;
  int32_t tiling_len = sizeof(optiling::DynamicRnnV3TilingData);
  TILING_PARSE_JSON_TO_COMPILEINFO("DynamicRNNV3", compileInfo, info);

  ATTACH_OPERATOR_TO_HOLDER(holder, opParas, tiling_len, info);
  HOLDER_DO_TILING(holder, "DynamicRNNV3", ge::GRAPH_FAILED);
}

TEST_F(DynamicRnnV3Tiling, dynamic_rnn_v3_tiling_11) {
  std::string op_name = "DynamicRNNV3";
  std::string compileInfo = "{\"vars\": {\"tune_shape_list\": [[98, 99], [99, 99, 1]]}}";

  vector<vector<int64_t>> v3_input_shapes = {
      {32,32,8,16,16},
      {40,32,16,16},
      {512},
  };

  vector<int64_t> v3_output_shape = {32,8,8,16,16};

  auto opParas = op::DynamicRNNV3(op_name);

  TENSOR_INPUT_WITH_SHAPE(opParas, x, v3_input_shapes[0], DT_FLOAT16, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, w, v3_input_shapes[1], DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output_h, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output_c, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, i, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, j, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, f, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, o, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, tanhc, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});

  optiling::DynamicRNNV3CompileInfo info;
  int32_t tiling_len = sizeof(optiling::DynamicRnnV3TilingData);
  TILING_PARSE_JSON_TO_COMPILEINFO("DynamicRNNV3", compileInfo, info);

  ATTACH_OPERATOR_TO_HOLDER(holder, opParas, tiling_len, info);
  HOLDER_DO_TILING(holder, "DynamicRNNV3", ge::GRAPH_FAILED);
}

TEST_F(DynamicRnnV3Tiling, dynamic_rnn_v3_tiling_12) {
  std::string op_name = "DynamicRNNV3";
  std::string compileInfo = "{\"vars\": {\"tune_shape_list\": [[-1, -1, 0], [99, 99, 1]]}}";

  vector<vector<int64_t>> v3_input_shapes = {
      {32,32,8,16,16},
      {40,32,16,16},
      {512},
  };

  vector<int64_t> v3_output_shape = {32,8,8,16,16};

  auto opParas = op::DynamicRNNV3(op_name);

  TENSOR_INPUT_WITH_SHAPE(opParas, x, {}, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, w, v3_input_shapes[1], DT_FLOAT16, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, b, v3_input_shapes[2], DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output_h, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output_c, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, i, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, j, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, f, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, o, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, tanhc, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});

  optiling::DynamicRNNV3CompileInfo info;
  int32_t tiling_len = sizeof(optiling::DynamicRnnV3TilingData);
  TILING_PARSE_JSON_TO_COMPILEINFO("DynamicRNNV3", compileInfo, info);

  ATTACH_OPERATOR_TO_HOLDER(holder, opParas, tiling_len, info);
  HOLDER_DO_TILING(holder, "DynamicRNNV3", ge::GRAPH_FAILED);
}

TEST_F(DynamicRnnV3Tiling, dynamic_rnn_v3_tiling_13) {
  std::string op_name = "DynamicRNNV3";
  std::string compileInfo = "{\"vars\": {\"tune_shape_list\": [[-1, -1, 0], [99, 99, 1]]}}";

  vector<vector<int64_t>> v3_input_shapes = {
      {32,32},
      {40,32,16,16},
      {512},
  };

  vector<int64_t> v3_output_shape = {32,8,8,16,16};

  auto opParas = op::DynamicRNNV3(op_name);

  TENSOR_INPUT_WITH_SHAPE(opParas, x, v3_input_shapes[0], DT_FLOAT16, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, w, v3_input_shapes[1], DT_FLOAT16, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, b, v3_input_shapes[2], DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output_h, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output_c, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, i, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, j, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, f, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, o, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, tanhc, v3_output_shape, DT_FLOAT16, FORMAT_ND, {});

  optiling::DynamicRNNV3CompileInfo info;
  int32_t tiling_len = sizeof(optiling::DynamicRnnV3TilingData);
  TILING_PARSE_JSON_TO_COMPILEINFO("DynamicRNNV3", compileInfo, info);

  ATTACH_OPERATOR_TO_HOLDER(holder, opParas, tiling_len, info);
  HOLDER_DO_TILING(holder, "DynamicRNNV3", ge::GRAPH_FAILED);
}
