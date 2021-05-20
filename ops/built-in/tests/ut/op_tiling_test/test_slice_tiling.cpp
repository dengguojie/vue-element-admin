#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include <register/op_tiling.h>

using namespace std;

class slice_tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "slice_tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "slice_tiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream &tiling_data) {
  auto data = tiling_data.str();
  string result;
  int64_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }

  return result;
}

TEST_F(slice_tiling, slice_tiling_no_mask) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("Slice");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {4},
      {4},
  };

  vector<string> dtypes = {"float16", "int32", "int32"};
  for (size_t i = 0; i < input_shapes.size(); i++) {
    tensorInputs.tensor.clear();
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);
  }

  TeOpTensor tensorOutput;
  tensorOutput.shape = input_shapes[0];
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  vector<int32_t> offset = {1, 1, 1, 1};
  vector<int32_t> size = {2, 2, 2, 2};
  vector<int32_t> strides = {1, 1, 1, 1};
  opParas.const_inputs["offsets"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) offset.data(),
                                          offset.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["size"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) size.data(),
                                          size.size() * sizeof(int32_t), ge::Tensor());

  std::string compileInfo = R"({"vars": {"block_dim": 32}})";

  OpRunInfo runInfo;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  auto ret = iter->second(opParas, op_compile_info, runInfo);
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "0 4 4 4 4 4 2 2 2 2 1 1 1 1 3 3 3 3 1 1 1 1 ");
}

TEST_F(slice_tiling, slice_tiling_with_mask1) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("Slice");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {10,10},
      {2},
      {2},
  };

  vector<string> dtypes = {"float16", "int32", "int32"};
  for (size_t i = 0; i < input_shapes.size(); i++) {
    tensorInputs.tensor.clear();
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);
  }

  TeOpTensor tensorOutput;
  tensorOutput.shape = input_shapes[0];
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  vector<int32_t> offset = {0, 0};
  vector<int32_t> size = {3, 3};
  vector<int32_t> strides = {1, 1};
  opParas.const_inputs["offsets"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) offset.data(),
                                                      offset.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["size"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) size.data(),
                                                      size.size() * sizeof(int32_t), ge::Tensor());
  std::string compileInfo = R"({"vars": {"block_dim": 32}})";

  OpRunInfo runInfo;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  auto ret = iter->second(opParas, op_compile_info, runInfo);
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "0 2 10 10 3 3 0 0 3 3 1 1 ");
}

TEST_F(slice_tiling, slice_tiling_no_const_value) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("Slice");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {10,10},
      {2},
      {2},
  };

  vector<string> dtypes = {"float16", "int32", "int32"};
  for (size_t i = 0; i < input_shapes.size(); i++) {
    tensorInputs.tensor.clear();
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);
  }

  TeOpTensor tensorOutput;
  tensorOutput.shape = input_shapes[0];
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  vector<int32_t> offset = {0, 0};
  vector<int32_t> size = {3, 3};
  vector<int32_t> strides = {1, 1};
  std::string compileInfo = R"({"vars": {"block_dim": 32}})";

  OpRunInfo runInfo;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  auto ret = iter->second(opParas, op_compile_info, runInfo);
  ASSERT_FALSE(ret);
}

TEST_F(slice_tiling, slice_tiling_invalid_begin_length) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("Slice");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {10,10},
      {2},
      {2},
  };

  vector<string> dtypes = {"float16", "int32", "int32"};
  for (size_t i = 0; i < input_shapes.size(); i++) {
    tensorInputs.tensor.clear();
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);
  }

  TeOpTensor tensorOutput;
  tensorOutput.shape = input_shapes[0];
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  vector<int32_t> offset = {0, 0, 0};
  vector<int32_t> size = {3, 3};
  vector<int32_t> strides = {1, 1};
  opParas.const_inputs["offsets"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) offset.data(),
                                                      offset.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["size"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) size.data(),
                                                      size.size() * sizeof(int32_t), ge::Tensor());
  std::string compileInfo = R"({"vars": {"block_dim": 32}})";

  OpRunInfo runInfo;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  auto ret = iter->second(opParas, op_compile_info, runInfo);
  ASSERT_FALSE(ret);
}

TEST_F(slice_tiling, slice_tiling_invalid_begin_value) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("Slice");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {10,10},
      {2},
      {2},
  };

  vector<string> dtypes = {"float16", "int32", "int32"};
  for (size_t i = 0; i < input_shapes.size(); i++) {
    tensorInputs.tensor.clear();
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);
  }

  TeOpTensor tensorOutput;
  tensorOutput.shape = input_shapes[0];
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  vector<int32_t> offset = {-1, -1};
  vector<int32_t> size = {3, 3};
  vector<int32_t> strides = {1, 1};
  opParas.const_inputs["offsets"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) offset.data(),
                                                      offset.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["size"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) size.data(),
                                                      size.size() * sizeof(int32_t), ge::Tensor());
  std::string compileInfo = R"({"vars": {"block_dim": 32}})";

  OpRunInfo runInfo;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  auto ret = iter->second(opParas, op_compile_info, runInfo);
  ASSERT_FALSE(ret);
}

TEST_F(slice_tiling, slice_tiling_end_value_negative_one) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("Slice");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {10,10},
      {2},
      {2},
  };

  vector<string> dtypes = {"float16", "int32", "int32"};
  for (size_t i = 0; i < input_shapes.size(); i++) {
    tensorInputs.tensor.clear();
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);
  }

  TeOpTensor tensorOutput;
  tensorOutput.shape = input_shapes[0];
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  vector<int32_t> offset = {0, 0};
  vector<int32_t> size = {-1, -1};
  vector<int32_t> strides = {1, 1};
  opParas.const_inputs["offsets"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) offset.data(),
                                                      offset.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["size"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) size.data(),
                                                      size.size() * sizeof(int32_t), ge::Tensor());
  std::string compileInfo = R"({"vars": {"block_dim": 32}})";

  OpRunInfo runInfo;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  auto ret = iter->second(opParas, op_compile_info, runInfo);
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "0 1 100 100 0 100 1 ");
}

TEST_F(slice_tiling, slice_tiling_empty_input) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("Slice");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {10,10},
      {2},
      {2},
  };

  vector<string> dtypes = {"float16", "int32", "int32"};
  TeOpTensor tensorOutput;
  tensorOutput.shape = input_shapes[0];
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  vector<int32_t> offset = {0, 0};
  vector<int32_t> size = {-1, -1};
  vector<int32_t> strides = {1, 1};
  opParas.const_inputs["offsets"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) offset.data(),
                                                      offset.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["size"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) size.data(),
                                                      size.size() * sizeof(int32_t), ge::Tensor());
  std::string compileInfo = R"({"vars": {"block_dim": 32}})";

  OpRunInfo runInfo;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  auto ret = iter->second(opParas, op_compile_info, runInfo);
  ASSERT_FALSE(ret);
}
