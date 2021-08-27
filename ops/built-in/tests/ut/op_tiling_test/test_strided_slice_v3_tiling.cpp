#include <stdlib.h>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include <register/op_tiling.h>

using namespace std;

class stried_slice_v3_tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "stried_slice_v3_tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() { 
    std::cout << "stried_slice_v3_tiling TearDown" << std::endl;
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

TEST_F(stried_slice_v3_tiling, stried_slice_v3_tiling_no_mask) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("StridedSliceV3");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {4},
      {4},
      {4},
      {4},
  };

  vector<string> dtypes = {"float16", "int32", "int32", "int32","int32"};
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
  vector<int32_t> axes = {0, 1, 2, 3};
  opParas.const_inputs["begin"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) begin.data(),
                                                      begin.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["end"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) end.data(),
                                                      end.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["strides"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) strides.data(),
                                                      strides.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["axes" ] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) axes.data(),
                                                      axes.size() * sizeof(int32_t), ge::Tensor());
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

TEST_F(stried_slice_v3_tiling, stried_slice_v3_tiling_no_axes) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("StridedSliceV3");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {4},
      {4},
      {4},
      //no axes
  };

  vector<string> dtypes = {"float16", "int32", "int32", "int32","int32"};
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

TEST_F(stried_slice_v3_tiling, stried_slice_v3_tiling_pad_head) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("StridedSliceV3");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {2},
      {2},
      {2},
      {2},
  };

  vector<string> dtypes = {"float16", "int64", "int64", "int64","int64"};
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
  vector<int64_t> begin = {1, 1};
  vector<int64_t> end = {3, 3 };
  vector<int64_t> strides = {1, 1};
  vector<int64_t> axes = {0, 1};
  opParas.const_inputs["begin"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) begin.data(),
                                                      begin.size() * sizeof(int64_t), ge::Tensor());
  opParas.const_inputs["end"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) end.data(),
                                                      end.size() * sizeof(int64_t), ge::Tensor());
  opParas.const_inputs["strides"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) strides.data(),
                                                      strides.size() * sizeof(int64_t), ge::Tensor());
  opParas.const_inputs["axes" ] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) axes.data(),
                                                      axes.size() * sizeof(int64_t), ge::Tensor());
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
            //before fused is "1 4 4 4 4 4 2 2 4 4 1 1 0 0 3 3 4 4 1 1 1 1 "
            "2 2 4 64 2 32 1 16 3 48 1 1 ");
}

TEST_F(stried_slice_v3_tiling, stried_slice_v3_tiling_pad_tail) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("StridedSliceV3");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {2},
      {2},
      {2},
      {2},
  };

  vector<string> dtypes = {"float16", "int32", "int32", "int32","int32"};
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
  vector<int32_t> begin = {1, 1};
  vector<int32_t> end = {3, 3 };
  vector<int32_t> strides = {1, 1};
  vector<int32_t> axes = {2, 3};
  opParas.const_inputs["begin"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) begin.data(),
                                                      begin.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["end"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) end.data(),
                                                      end.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["strides"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) strides.data(),
                                                      strides.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["axes" ] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) axes.data(),
                                                      axes.size() * sizeof(int32_t), ge::Tensor());
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
            "1 3 16 4 4 16 2 2 0 1 1 16 3 3 1 1 1 ");
}

TEST_F(stried_slice_v3_tiling, stried_slice_v3_tiling_no_begin) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("StridedSliceV3");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {},
      {4},
      {4},
      {4},
  };

  vector<string> dtypes = {"float16", "int32", "int32", "int32","int32"};
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
  vector<int32_t> end = {3, 3, 3, 3};
  vector<int32_t> strides = {1, 1, 1, 1};
  vector<int32_t> axes = {0, 1, 2, 3};
  opParas.const_inputs["end"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) end.data(),
                                                      end.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["strides"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) strides.data(),
                                                      strides.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["axes" ] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) axes.data(),
                                                      axes.size() * sizeof(int32_t), ge::Tensor());
  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 262144}})";
 
  OpRunInfo runInfo;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  auto ret = iter->second(opParas, op_compile_info, runInfo);
  std::cout << to_string(runInfo.tiling_data) << std::endl;
  ASSERT_FALSE(ret);
}

TEST_F(stried_slice_v3_tiling, stried_slice_v3_tiling_diff_size) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("StridedSliceV3");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {1},
      {4},
      {4},
      {4},
  };

  vector<string> dtypes = {"float16", "int32", "int32", "int32","int32"};
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
  vector<int32_t> begin = {1, };
  vector<int32_t> end = {3, 3, 3, 3};
  vector<int32_t> strides = {1, 1, 1, 1};
  vector<int32_t> axes = {0, 1, 2, 3};
  opParas.const_inputs["begin"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) begin.data(),
                                                      begin.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["end"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) end.data(),
                                                      end.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["strides"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) strides.data(),
                                                      strides.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["axes" ] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) axes.data(),
                                                      axes.size() * sizeof(int32_t), ge::Tensor());
  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 262144}})";
 
  OpRunInfo runInfo;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  auto ret = iter->second(opParas, op_compile_info, runInfo);
  std::cout << to_string(runInfo.tiling_data) << std::endl;
  ASSERT_FALSE(ret);
}

TEST_F(stried_slice_v3_tiling, stried_slice_v3_tiling_no_mask_neg) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter =
      optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("StridedSliceV3");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {4},
      {4},
      {4},
      {4},
  };

  vector<string> dtypes = {"float16", "int32", "int32", "int32","int32"};
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
  vector<int32_t> begin = {1, 1, 1, -3};
  vector<int32_t> end = {3, 3, 3, -1};
  vector<int32_t> strides = {1, 1, 1, 1};
  vector<int32_t> axes = {0, 1, 2, -1};
  opParas.const_inputs["begin"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) begin.data(),
                                                      begin.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["end"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) end.data(),
                                                      end.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["strides"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) strides.data(),
                                                      strides.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["axes" ] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) axes.data(),
                                                      axes.size() * sizeof(int32_t), ge::Tensor());
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