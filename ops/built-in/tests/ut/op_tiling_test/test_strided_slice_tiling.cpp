#include <stdlib.h>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include <register/op_tiling.h>

using namespace std;

class stried_slice_tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "stried_slice_tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "stried_slice_tiling TearDown" << std::endl;
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

TEST_F(stried_slice_tiling, stried_slice_tiling_no_mask) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {4},
      {4},
      {4},
  };

  vector<string> dtypes = {"float16", "int32", "int32", "int32"};
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
  vector<int32_t> begin = {1, 1, 1, 1};
  vector<int32_t> end = {3, 3, 3, 3};
  vector<int32_t> strides = {1, 1, 1, 1};
  opParas.const_inputs["begin"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) begin.data(),
                                                      begin.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["end"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) end.data(),
                                                      end.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["strides"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) strides.data(),
                                                      strides.size() * sizeof(int32_t), ge::Tensor());
  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 262144}})";

  OpRunInfo runInfo;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  auto ret = iter->second(opParas, op_compile_info, runInfo);
  std::cout << to_string(runInfo.tiling_data) << std::endl;
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "1 4 4 4 4 4 2 2 2 2 1 1 1 1 3 3 3 3 1 1 1 1 ");
}

TEST_F(stried_slice_tiling, stried_slice_tiling_with_mask1) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {10,10,3,1,2},
      {2},
      {2},
      {2},
  };

  vector<string> dtypes = {"float16", "int32", "int32", "int32"};
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
  vector<int32_t> begin = {0,0};
  vector<int32_t> end = {3, 3};
  vector<int32_t> strides = {1, 1};
  opParas.const_inputs["begin"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) begin.data(),
                                          begin.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["end"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) end.data(),
                                          end.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["strides"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) strides.data(),
                                          strides.size() * sizeof(int32_t), ge::Tensor());
  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 1, "new_axis_mask": 0, "shrink_axis_mask": 2, "ub_size": 262144}})";

  OpRunInfo runInfo;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  auto ret = iter->second(opParas, op_compile_info, runInfo);
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "1 2 300 2 300 1 0 0 300 1 1 1 ");
}

TEST_F(stried_slice_tiling, stried_slice_tiling_int64_const) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {10,10,3,1,2},
      {2},
      {2},
      {2},
  };

  vector<string> dtypes = {"float16", "int64", "int64", "int64"};
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
  vector<int64_t> begin = {0,0};
  vector<int64_t> end = {3, 3};
  vector<int64_t> strides = {1, 1};
  opParas.const_inputs["begin"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) begin.data(),
                                                      begin.size() * sizeof(int64_t), ge::Tensor());
  opParas.const_inputs["end"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) end.data(),
                                                      end.size() * sizeof(int64_t), ge::Tensor());
  opParas.const_inputs["strides"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) strides.data(),
                                                      strides.size() * sizeof(int64_t), ge::Tensor());
  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 1, "new_axis_mask": 0, "shrink_axis_mask": 2, "ub_size": 262144}})";

  OpRunInfo runInfo;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  auto ret = iter->second(opParas, op_compile_info, runInfo);
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "1 2 300 2 300 1 0 0 300 1 1 1 ");
}

TEST_F(stried_slice_tiling, stried_slice_no_mask) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {10,10,3,1,2},
      {2},
      {2},
      {2},
  };

  vector<string> dtypes = {"float16", "int32", "int32", "int32"};
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
  vector<int32_t> begin = {0,0};
  vector<int32_t> end = {3, 3};
  vector<int32_t> strides = {1, 1};
  opParas.const_inputs["begin"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) begin.data(),
                                                      begin.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["end"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) end.data(),
                                                      end.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["strides"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) strides.data(),
                                                      strides.size() * sizeof(int32_t), ge::Tensor());
  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "end_mask": 0, "ellipsis_mask": 1, "new_axis_mask": 0, "shrink_axis_mask": 2}})";

  OpRunInfo runInfo;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  auto ret = iter->second(opParas, op_compile_info, runInfo);
  EXPECT_FALSE(ret);

  op_compile_info.str =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "ellipsis_mask": 1, "new_axis_mask": 0, "shrink_axis_mask": 2}})";
  op_compile_info.key += "1";
  ret = iter->second(opParas, op_compile_info, runInfo);
  EXPECT_FALSE(ret);

  op_compile_info.str =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 2}})";
  op_compile_info.key += "1";
  ret = iter->second(opParas, op_compile_info, runInfo);
  EXPECT_FALSE(ret);

  op_compile_info.str =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 1, "shrink_axis_mask": 2}})";
  op_compile_info.key += "1";
  ret = iter->second(opParas, op_compile_info, runInfo);
  EXPECT_FALSE(ret);

  op_compile_info.str =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 1, "new_axis_mask": 0}})";
  op_compile_info.key += "1";
  ret = iter->second(opParas, op_compile_info, runInfo);
  EXPECT_FALSE(ret);

  op_compile_info.str =
      R"({"vars": {"begin_mask": 0, "end_mask": 0, "ellipsis_mask": 1, "new_axis_mask": 0, "shrink_axis_mask": 2}})";
  op_compile_info.key += "1";
  ret = iter->second(opParas, op_compile_info, runInfo);
  EXPECT_FALSE(ret);
}

TEST_F(stried_slice_tiling, stried_slice_tiling_no_inputs) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {10,10,3,1,2},
      {2},
      {2},
      {2},
  };

  vector<string> dtypes = {"float16", "int32", "int32", "int32"};

  TeOpTensor tensorOutput;
  tensorOutput.shape = input_shapes[0];
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  vector<int32_t> begin = {0,0};
  vector<int32_t> end = {3, 3};
  vector<int32_t> strides = {1, 1};
  opParas.const_inputs["begin"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) begin.data(),
                                                      begin.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["end"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) end.data(),
                                                      end.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["strides"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) strides.data(),
                                                      strides.size() * sizeof(int32_t), ge::Tensor());
  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 1, "new_axis_mask": 0, "shrink_axis_mask": 2}})";

  OpRunInfo runInfo;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  auto ret = iter->second(opParas, op_compile_info, runInfo);
  ASSERT_FALSE(ret);
}

TEST_F(stried_slice_tiling, stried_slice_tiling_too_large_dims) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {10,10,3,1,2,2,2,3,3},
      {2},
      {2},
      {2},
  };

  vector<string> dtypes = {"float16", "int32", "int32", "int32"};
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
  vector<int32_t> begin = {0,0};
  vector<int32_t> end = {3, 3};
  vector<int32_t> strides = {1, 1};
  opParas.const_inputs["begin"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) begin.data(),
                                                      begin.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["end"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) end.data(),
                                                      end.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["strides"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) strides.data(),
                                                      strides.size() * sizeof(int32_t), ge::Tensor());
  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 1, "new_axis_mask": 0, "shrink_axis_mask": 2}})";

  OpRunInfo runInfo;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  auto ret = iter->second(opParas, op_compile_info, runInfo);
  ASSERT_FALSE(ret);
}

TEST_F(stried_slice_tiling, stried_slice_tiling_get_const_value_failed) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {4},
      {4},
      {4},
  };

  vector<string> dtypes = {"float16", "float32", "float32", "float32"};
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
  vector<float> begin = {1, 1, 1, 1};
  vector<float> end = {3, 3, 3, 3};
  vector<float> strides = {1, 1, 1, 1};
  opParas.const_inputs["begin"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) begin.data(),
                                                      begin.size() * sizeof(float), ge::Tensor());
  opParas.const_inputs["end"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) end.data(),
                                                      end.size() * sizeof(float), ge::Tensor());
  opParas.const_inputs["strides"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) strides.data(),
                                                      strides.size() * sizeof(float), ge::Tensor());
  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0}})";

  OpRunInfo runInfo;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  auto ret = iter->second(opParas, op_compile_info, runInfo);
  ASSERT_FALSE(ret);
}

TEST_F(stried_slice_tiling, stried_slice_tiling_invalid_stride) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {4},
      {4},
      {4},
  };

  vector<string> dtypes = {"float16", "int32", "int32", "int32"};
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
  vector<int32_t> begin = {1, 1, 1, 1};
  vector<int32_t> end = {3, 3, 3, 3};
  vector<int32_t> strides = {1, 1, 1, 0};
  opParas.const_inputs["begin"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) begin.data(),
                                                      begin.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["end"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) end.data(),
                                                      end.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["strides"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) strides.data(),
                                                      strides.size() * sizeof(int32_t), ge::Tensor());
  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0}})";

  OpRunInfo runInfo;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  auto ret = iter->second(opParas, op_compile_info, runInfo);
  ASSERT_FALSE(ret);
}

TEST_F(stried_slice_tiling, stried_slice_tiling_unsupported_stride) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {4},
      {4},
      {4},
  };

  vector<string> dtypes = {"float16", "int32", "int32", "int32"};
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
  vector<int32_t> begin = {1, 1, 1, 1};
  vector<int32_t> end = {3, 3, 3, 3};
  vector<int32_t> strides = {1, 1, 2, 1};
  opParas.const_inputs["begin"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) begin.data(),
                                                      begin.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["end"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) end.data(),
                                                      end.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["strides"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) strides.data(),
                                                      strides.size() * sizeof(int32_t), ge::Tensor());
  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0}})";

  OpRunInfo runInfo;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  auto ret = iter->second(opParas, op_compile_info, runInfo);
  ASSERT_FALSE(ret);
}

TEST_F(stried_slice_tiling, stried_slice_tiling_fused_dims) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {4},
      {4},
      {4},
  };

  vector<string> dtypes = {"float16", "int32", "int32", "int32"};
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
  vector<int32_t> begin = {1, 1, 1, 0};
  vector<int32_t> end = {3, 3, 3, 4};
  vector<int32_t> strides = {1, 1, 1, 1};
  opParas.const_inputs["begin"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) begin.data(),
                                                      begin.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["end"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) end.data(),
                                                      end.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["strides"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) strides.data(),
                                                      strides.size() * sizeof(int32_t), ge::Tensor());
  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 262144}})";

  OpRunInfo runInfo;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  setenv("ASCEND_GLOBAL_LOG_LEVEL", "0", 1);
  extern void InitLogLevelByEnv();
  InitLogLevelByEnv();
  auto ret = iter->second(opParas, op_compile_info, runInfo);
  unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
  InitLogLevelByEnv();
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "1 3 4 4 16 2 2 8 1 1 4 3 3 12 1 1 1 ");
}
