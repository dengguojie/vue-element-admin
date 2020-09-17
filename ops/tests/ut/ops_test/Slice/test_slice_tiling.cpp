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
      optiling::OpTilingInterf::RegisteredOpInterf().find("Slice");
  ASSERT_TRUE(iter != optiling::OpTilingInterf::RegisteredOpInterf().end());
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
      std::tuple<const uint8_t *, size_t>((const uint8_t *) offset.data(),
                                          offset.size() * sizeof(int32_t));
  opParas.const_inputs["size"] =
      std::tuple<const uint8_t *, size_t>((const uint8_t *) size.data(),
                                          size.size() * sizeof(int32_t));

  std::string compileInfo =
      "{\"vars\": {\"block_dim\": 32, \"begin_mask\": 0, \"end_mask\": 0, \"ellipsis_mask\": 0, \"new_axis_mask\": 0, \"shrink_axis_mask\": 0}}";

// do tilling, get runInfo
  nlohmann::json op_info = nlohmann::json::parse(compileInfo);
  OpRunInfo runInfo;
  iter->second("StridedSlice", opParas, op_info, runInfo);
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "4 4 4 4 2 2 2 2 1 1 1 1 3 3 3 3 1 1 1 1 ");
}

TEST_F(slice_tiling, slice_tiling_with_mask1) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter =
      optiling::OpTilingInterf::RegisteredOpInterf().find("Slice");
  ASSERT_TRUE(iter != optiling::OpTilingInterf::RegisteredOpInterf().end());
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
      std::tuple<const uint8_t *, size_t>((const uint8_t *) offset.data(),
                                          offset.size() * sizeof(int32_t));
  opParas.const_inputs["size"] =
      std::tuple<const uint8_t *, size_t>((const uint8_t *) size.data(),
                                          size.size() * sizeof(int32_t));
  std::string compileInfo =
      "{\"vars\": {\"block_dim\": 32, \"begin_mask\": 0, \"end_mask\": 0, \"ellipsis_mask\": 1, \"new_axis_mask\": 0, \"shrink_axis_mask\": 2}}";

// do tilling, get runInfo
  nlohmann::json op_info = nlohmann::json::parse(compileInfo);
  OpRunInfo runInfo;
  iter->second("StridedSlice", opParas, op_info, runInfo);
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "10 10 3 3 0 0 3 3 1 1 ");
}
