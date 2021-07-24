#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class SplitVTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SplitVTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SplitVTiling TearDown" << std::endl;
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


TEST_F(SplitVTiling, SplitV_tiling1) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("SplitV");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {1820, 232},
      {1},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {1820, 232},
  };
  vector<string> dtypes = {"int8", "int32", "int32"};
  std::vector<int32_t> size_splits{1820};
  std::vector<int32_t> split_dim{0};

  for (size_t i = 0; i < input_shapes.size(); i++) {
    TeOpTensorArg tensorInputArg;
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputArg.tensor.push_back(tensorInput);
    tensorInputArg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputArg);
  }

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TeOpTensorArg tensorOutputsArg;
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = dtypes[0];
    tensorOutputsArg.tensor.push_back(tensorOutput);
    tensorOutputsArg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputsArg);
  }

  opParas.const_inputs["size_splits"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)size_splits.data(), size_splits.size() * 4, ge::Tensor());
  opParas.const_inputs["split_dim"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)split_dim.data(), split_dim.size() * 4, ge::Tensor());

  opParas.op_type = "SplitV";
  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":253952, \"num_split\":1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456cde";
  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "1 32 422240 1820 13195 13195 0 13195 13195 0 13195 13195 232 1 422240 0 0 0 0 0 0 0 0 0 ");
}

TEST_F(SplitVTiling, SplitV_tiling2) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("SplitV");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {232, 1820},
      {3},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {100, 1820},
      {96, 1820},
      {36, 18720},
  };
  vector<string> dtypes = {"int32", "int32", "int32"};
  std::vector<int32_t> size_splits{100, 96, 36};
  std::vector<int32_t> split_dim{0};

  for (size_t i = 0; i < input_shapes.size(); i++) {
    TeOpTensorArg tensorInputArg;
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputArg.tensor.push_back(tensorInput);
    tensorInputArg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputArg);
  }

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TeOpTensorArg tensorOutputsArg;
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = dtypes[0];
    tensorOutputsArg.tensor.push_back(tensorOutput);
    tensorOutputsArg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputsArg);
  }

  opParas.const_inputs["size_splits"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)size_splits.data(), size_splits.size() * 4, ge::Tensor());
  opParas.const_inputs["split_dim"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)split_dim.data(), split_dim.size() * 4, ge::Tensor());

  opParas.op_type = "SplitV";
  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":63488, \"num_split\":3}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456cef";
  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "2 32 422240 232 0 0 0 0 0 0 0 0 1820 1 422240 0 0 0 0 0 0 0 0 0 ");
}

TEST_F(SplitVTiling, SplitV_tiling3) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("SplitV");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {1820, 232},
      {5},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {1820, 80},
      {1820, 50},
      {1820, 1},
      {1820, 46},
      {1820, 55},
  };
  vector<string> dtypes = {"float16", "int32", "int32"};
  std::vector<int32_t> size_splits{80, 50, 1, 46, 55};
  std::vector<int32_t> split_dim{-1};

  for (size_t i = 0; i < input_shapes.size(); i++) {
    TeOpTensorArg tensorInputArg;
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputArg.tensor.push_back(tensorInput);
    tensorInputArg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputArg);
  }

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TeOpTensorArg tensorOutputsArg;
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = dtypes[0];
    tensorOutputsArg.tensor.push_back(tensorOutput);
    tensorOutputsArg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputsArg);
  }

  opParas.const_inputs["size_splits"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)size_splits.data(), size_splits.size() * 4, ge::Tensor());
  opParas.const_inputs["split_dim"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)split_dim.data(), split_dim.size() * 4, ge::Tensor());

  opParas.op_type = "SplitV";
  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":126976, \"num_split\":5}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456cfg";
  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "3 32 422240 232 0 0 0 0 0 0 0 0 1 1820 232 0 0 0 0 0 0 0 0 0 ");
}

TEST_F(SplitVTiling, SplitV_tiling4) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("SplitV");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {1, 48, 512},
      {48},
      {1},
  };
  vector<int64_t> output_shape = {1, 1, 512};
  vector<string> dtypes = {"float16", "int32", "int32"};
  std::vector<int32_t> size_splits{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
  std::vector<int32_t> split_dim{1};

  for (size_t i = 0; i < input_shapes.size(); i++) {
    TeOpTensorArg tensorInputArg;
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputArg.tensor.push_back(tensorInput);
    tensorInputArg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputArg);
  }

  for (size_t i = 0; i < 48; i++) {
    TeOpTensorArg tensorOutputsArg;
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shape;
    tensorOutput.dtype = dtypes[0];
    tensorOutputsArg.tensor.push_back(tensorOutput);
    tensorOutputsArg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputsArg);
  }

  opParas.const_inputs["size_splits"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)size_splits.data(), size_splits.size() * 4, ge::Tensor());
  opParas.const_inputs["split_dim"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)split_dim.data(), split_dim.size() * 4, ge::Tensor());

  opParas.op_type = "SplitV";
  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":126976, \"num_split\":48}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456cgh";
  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "8 32 24576 48 0 0 0 0 0 0 0 0 512 1 24576 0 0 0 0 0 0 0 0 0 ");
}

TEST_F(SplitVTiling, SplitV_tiling5) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("SplitV");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {18720, 3},
      {3},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {18720, 1},
      {18720, 1},
      {18720, 1},
  };
  vector<string> dtypes = {"float16", "int32", "int32"};
  std::vector<int32_t> size_splits{1,1,1};
  std::vector<int32_t> split_dim{-1};

  for (size_t i = 0; i < input_shapes.size(); i++) {
    TeOpTensorArg tensorInputArg;
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputArg.tensor.push_back(tensorInput);
    tensorInputArg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputArg);
  }

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TeOpTensorArg tensorOutputsArg;
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = dtypes[0];
    tensorOutputsArg.tensor.push_back(tensorOutput);
    tensorOutputsArg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputsArg);
  }

  opParas.const_inputs["size_splits"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)size_splits.data(), size_splits.size() * 4, ge::Tensor());
  opParas.const_inputs["split_dim"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)split_dim.data(), split_dim.size() * 4, ge::Tensor());

  opParas.op_type = "SplitV";
  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":126976, \"num_split\":3}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456chi";
  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "4 25 56160 3 0 0 0 0 0 0 0 0 1 18720 3 0 224 3 0 3 1 0 1 0 ");
}

TEST_F(SplitVTiling, SplitV_tiling6) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("SplitV");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {18725, 6},
      {3},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {18725, 1},
      {18725, 2},
      {18725, 3},
  };
  vector<string> dtypes = {"float16", "int32", "int32"};
  std::vector<int32_t> size_splits{1,2,3};
  std::vector<int32_t> split_dim{-1};

  for (size_t i = 0; i < input_shapes.size(); i++) {
    TeOpTensorArg tensorInputArg;
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputArg.tensor.push_back(tensorInput);
    tensorInputArg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputArg);
  }

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TeOpTensorArg tensorOutputsArg;
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = dtypes[0];
    tensorOutputsArg.tensor.push_back(tensorOutput);
    tensorOutputsArg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputsArg);
  }

  opParas.const_inputs["size_splits"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)size_splits.data(), size_splits.size() * 4, ge::Tensor());
  opParas.const_inputs["split_dim"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)split_dim.data(), split_dim.size() * 4, ge::Tensor());

  opParas.op_type = "SplitV";
  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":126976, \"num_split\":3}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456cij";
  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "5 32 112350 6 592 373 0 592 592 0 373 373 1 18725 6 0 0 0 0 0 0 0 0 0 ");
}

TEST_F(SplitVTiling, SplitV_tiling7) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("SplitV");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {48000, 256},
      {7},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {48000, 80},
      {48000, 80},
      {48000, 80},
      {48000, 1},
      {48000, 1},
      {48000, 1},
      {48000, 13},
  };
  vector<string> dtypes = {"float16", "int32", "int32"};
  std::vector<int32_t> size_splits{80,80,80,1,1,1,13};
  std::vector<int32_t> split_dim{-1};

  for (size_t i = 0; i < input_shapes.size(); i++) {
    TeOpTensorArg tensorInputArg;
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputArg.tensor.push_back(tensorInput);
    tensorInputArg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputArg);
  }

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TeOpTensorArg tensorOutputsArg;
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = dtypes[0];
    tensorOutputsArg.tensor.push_back(tensorOutput);
    tensorOutputsArg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputsArg);
  }

  opParas.const_inputs["size_splits"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)size_splits.data(), size_splits.size() * 4, ge::Tensor());
  opParas.const_inputs["split_dim"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)split_dim.data(), split_dim.size() * 4, ge::Tensor());

  opParas.op_type = "SplitV";
  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":126976, \"num_split\":7}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456cjk";
  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "6 32 12288000 256 1504 1376 11 96 128 10 96 128 1 48000 256 0 0 0 0 0 0 0 0 0 ");
}

TEST_F(SplitVTiling, SplitV_tiling8) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("SplitV");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {40000, 85},
      {4},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {40000, 2},
      {40000, 2},
      {40000, 1},
      {40000, 80},
  };
  vector<string> dtypes = {"float16", "int32", "int32"};
  std::vector<int32_t> size_splits{2,2,1,80};
  std::vector<int32_t> split_dim{-1};

  for (size_t i = 0; i < input_shapes.size(); i++) {
    TeOpTensorArg tensorInputArg;
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputArg.tensor.push_back(tensorInput);
    tensorInputArg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputArg);
  }

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TeOpTensorArg tensorOutputsArg;
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = dtypes[0];
    tensorOutputsArg.tensor.push_back(tensorOutput);
    tensorOutputsArg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputsArg);
  }

  opParas.const_inputs["size_splits"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)size_splits.data(), size_splits.size() * 4, ge::Tensor());
  opParas.const_inputs["split_dim"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)split_dim.data(), split_dim.size() * 4, ge::Tensor());

  opParas.op_type = "SplitV";
  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":126976, \"num_split\":4}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456ckl";
  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "7 32 3400000 85 1280 320 5 0 256 1 64 256 1 40000 85 0 0 0 0 0 0 0 0 0 ");
}

TEST_F(SplitVTiling, SplitV_tiling9) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("SplitV");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {40000, 85},
      {4},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {40000, 2},
      {40000, 2},
      {40000, 1},
      {40000, 80},
  };
  vector<string> dtypes = {"float16", "int32", "int32"};
  std::vector<int32_t> size_splits{32,32,16,1280};
  std::vector<int32_t> split_dim{-1};

  for (size_t i = 0; i < input_shapes.size(); i++) {
    TeOpTensorArg tensorInputArg;
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInput.format = "FRACTAL_NZ";
    tensorInputArg.tensor.push_back(tensorInput);
    tensorInputArg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputArg);
  }

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TeOpTensorArg tensorOutputsArg;
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.format = "FRACTAL_NZ";
    tensorOutput.dtype = dtypes[0];
    tensorOutputsArg.tensor.push_back(tensorOutput);
    tensorOutputsArg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputsArg);
  }

  opParas.const_inputs["size_splits"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)size_splits.data(), size_splits.size() * 4, ge::Tensor());
  opParas.const_inputs["split_dim"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)split_dim.data(), split_dim.size() * 4, ge::Tensor());

  opParas.op_type = "SplitV";
  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":126976, \"num_split\":4}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456ckl";
  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            to_string(runInfo.tiling_data));
}
