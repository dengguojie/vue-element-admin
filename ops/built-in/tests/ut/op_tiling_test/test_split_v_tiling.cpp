#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"
#include "split_combination_ops.h"
#include "array_ops.h"

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

static string to_string(const std::stringstream& tiling_data) {
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

using namespace ge;
#include "test_common.h"
/*
    .INPUT(x, TensorType::BasicType())
    .INPUT(size_splits, TensorType::IndexNumberType())
    .INPUT(split_dim, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(num_split, Int)
*/

TEST_F(SplitVTiling, SplitV_tiling1) {
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find("SplitV");
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  auto opParas = op::SplitV("SplitV");
  vector<vector<int64_t>> input_shapes = {
      {1820, 232},
      {1},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {1820, 232},
  };
  vector<ge::DataType> dtypes = {ge::DT_INT8, ge::DT_INT32, ge::DT_INT32};
  vector<int32_t> SizSplits{1820};
  vector<int32_t> SplitDim{0};

  TensorDesc tensorInputx;
  tensorInputx.SetShape(ge::Shape(input_shapes[0]));
  tensorInputx.SetDataType(dtypes[0]);
  TENSOR_INPUT(opParas, tensorInputx, x);

  TensorDesc tensorInputSizeSplits;
  tensorInputSizeSplits.SetShape(ge::Shape(input_shapes[1]));
  tensorInputSizeSplits.SetDataType(dtypes[1]);
  TENSOR_INPUT_CONST(opParas, tensorInputSizeSplits, size_splits, (const uint8_t*)SizSplits.data(),
                     SizSplits.size() * 4);

  TensorDesc tensorInputSplitDim;
  tensorInputSplitDim.SetShape(ge::Shape(input_shapes[2]));
  tensorInputSplitDim.SetDataType(dtypes[2]);
  TENSOR_INPUT_CONST(opParas, tensorInputSplitDim, split_dim, (const uint8_t*)SplitDim.data(), SplitDim.size() * 4);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TensorDesc tensorOutput;
    tensorOutput.SetShape(ge::Shape(output_shapes[i]));
    tensorOutput.SetDataType(dtypes[0]);
    TENSOR_OUTPUT(opParas, tensorOutput, y);
  }

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":253952, \"num_split\":1}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "1 32 422240 1820 13195 13195 0 13195 13195 0 13195 13195 232 1 422240 0 0 0 0 0 0 0 0 0 ");
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second(opParas, op_compile_info, runInfo);
  }
}

TEST_F(SplitVTiling, SplitV_tiling2) {
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find("SplitV");
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  auto opParas = op::SplitV("SplitV");
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
  vector<ge::DataType> dtypes = {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> size_splits{100, 96, 36};
  std::vector<int32_t> split_dim{0};

  TensorDesc tensorInputx;
  tensorInputx.SetShape(ge::Shape(input_shapes[0]));
  tensorInputx.SetDataType(dtypes[0]);
  TENSOR_INPUT(opParas, tensorInputx, x);

  TensorDesc tensorInputSizeSplits;
  tensorInputSizeSplits.SetShape(ge::Shape(input_shapes[1]));
  tensorInputSizeSplits.SetDataType(dtypes[1]);
  TENSOR_INPUT_CONST(opParas, tensorInputSizeSplits, size_splits, (const uint8_t*)size_splits.data(),
                     size_splits.size() * 4);

  TensorDesc tensorInputSplitDim;
  tensorInputSplitDim.SetShape(ge::Shape(input_shapes[2]));
  tensorInputSplitDim.SetDataType(dtypes[2]);
  TENSOR_INPUT_CONST(opParas, tensorInputSplitDim, split_dim, (const uint8_t*)split_dim.data(), split_dim.size() * 4);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TensorDesc tensorOutput;
    tensorOutput.SetShape(ge::Shape(output_shapes[i]));
    tensorOutput.SetDataType(dtypes[0]);
    TENSOR_OUTPUT(opParas, tensorOutput, y);
  }

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":63488, \"num_split\":3}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 32 422240 232 0 0 0 0 0 0 0 0 1820 1 422240 0 0 0 0 0 0 0 0 0 ");
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second(opParas, op_compile_info, runInfo);
  }
}

TEST_F(SplitVTiling, SplitV_tiling3) {
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find("SplitV");
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  auto opParas = op::SplitV("SplitV");
  vector<vector<int64_t>> input_shapes = {
      {1820, 232},
      {5},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {1820, 80}, {1820, 50}, {1820, 1}, {1820, 46}, {1820, 55},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> size_splits{80, 50, 1, 46, 55};
  std::vector<int32_t> split_dim{-1};

  TensorDesc tensorInputx;
  tensorInputx.SetShape(ge::Shape(input_shapes[0]));
  tensorInputx.SetDataType(dtypes[0]);
  TENSOR_INPUT(opParas, tensorInputx, x);

  TensorDesc tensorInputSizeSplits;
  tensorInputSizeSplits.SetShape(ge::Shape(input_shapes[1]));
  tensorInputSizeSplits.SetDataType(dtypes[1]);
  TENSOR_INPUT_CONST(opParas, tensorInputSizeSplits, size_splits, (const uint8_t*)size_splits.data(),
                     size_splits.size() * 4);

  TensorDesc tensorInputSplitDim;
  tensorInputSplitDim.SetShape(ge::Shape(input_shapes[2]));
  tensorInputSplitDim.SetDataType(dtypes[2]);
  TENSOR_INPUT_CONST(opParas, tensorInputSplitDim, split_dim, (const uint8_t*)split_dim.data(), split_dim.size() * 4);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TensorDesc tensorOutput;
    tensorOutput.SetShape(ge::Shape(output_shapes[i]));
    tensorOutput.SetDataType(dtypes[0]);
    TENSOR_OUTPUT(opParas, tensorOutput, y);
  }

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":126976, \"num_split\":5}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "3 32 422240 232 0 0 0 0 0 0 0 0 1 1820 232 0 0 0 0 0 0 0 0 0 ");
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second(opParas, op_compile_info, runInfo);
  }
}

TEST_F(SplitVTiling, SplitV_tiling4) {
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find("SplitV");
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  auto opParas = op::SplitV("SplitV");
  vector<vector<int64_t>> input_shapes = {
      {1, 48, 512},
      {48},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {1, 1, 512},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> size_splits{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int32_t> split_dim{1};

  TensorDesc tensorInputx;
  tensorInputx.SetShape(ge::Shape(input_shapes[0]));
  tensorInputx.SetDataType(dtypes[0]);
  TENSOR_INPUT(opParas, tensorInputx, x);

  TensorDesc tensorInputSizeSplits;
  tensorInputSizeSplits.SetShape(ge::Shape(input_shapes[1]));
  tensorInputSizeSplits.SetDataType(dtypes[1]);
  TENSOR_INPUT_CONST(opParas, tensorInputSizeSplits, size_splits, (const uint8_t*)size_splits.data(),
                     size_splits.size() * 4);

  TensorDesc tensorInputSplitDim;
  tensorInputSplitDim.SetShape(ge::Shape(input_shapes[2]));
  tensorInputSplitDim.SetDataType(dtypes[2]);
  TENSOR_INPUT_CONST(opParas, tensorInputSplitDim, split_dim, (const uint8_t*)split_dim.data(), split_dim.size() * 4);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TensorDesc tensorOutput;
    tensorOutput.SetShape(ge::Shape(output_shapes[i]));
    tensorOutput.SetDataType(dtypes[0]);
    TENSOR_OUTPUT(opParas, tensorOutput, y);
  }

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":126976, \"num_split\":48}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "8 32 24576 48 0 0 0 0 0 0 0 0 512 1 24576 0 0 0 0 0 0 0 0 0 ");
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second(opParas, op_compile_info, runInfo);
  }
}

TEST_F(SplitVTiling, SplitV_tiling5) {
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find("SplitV");
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  auto opParas = op::SplitV("SplitV");
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
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> size_splits{1, 1, 1};
  std::vector<int32_t> split_dim{-1};

  TensorDesc tensorInputx;
  tensorInputx.SetShape(ge::Shape(input_shapes[0]));
  tensorInputx.SetDataType(dtypes[0]);
  TENSOR_INPUT(opParas, tensorInputx, x);

  TensorDesc tensorInputSizeSplits;
  tensorInputSizeSplits.SetShape(ge::Shape(input_shapes[1]));
  tensorInputSizeSplits.SetDataType(dtypes[1]);
  TENSOR_INPUT_CONST(opParas, tensorInputSizeSplits, size_splits, (const uint8_t*)size_splits.data(),
                     size_splits.size() * 4);

  TensorDesc tensorInputSplitDim;
  tensorInputSplitDim.SetShape(ge::Shape(input_shapes[2]));
  tensorInputSplitDim.SetDataType(dtypes[2]);
  TENSOR_INPUT_CONST(opParas, tensorInputSplitDim, split_dim, (const uint8_t*)split_dim.data(), split_dim.size() * 4);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TensorDesc tensorOutput;
    tensorOutput.SetShape(ge::Shape(output_shapes[i]));
    tensorOutput.SetDataType(dtypes[0]);
    TENSOR_OUTPUT(opParas, tensorOutput, y);
  }

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":126976, \"num_split\":3}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4 25 56160 3 0 0 0 0 0 0 0 0 1 18720 3 0 224 3 0 3 1 0 1 0 ");
  int64_t tiling_test_num =0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second(opParas, op_compile_info, runInfo);
  }
}

TEST_F(SplitVTiling, SplitV_tiling6) {
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find("SplitV");
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  auto opParas = op::SplitV("SplitV");
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
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> size_splits{1, 2, 3};
  std::vector<int32_t> split_dim{-1};

  TensorDesc tensorInputx;
  tensorInputx.SetShape(ge::Shape(input_shapes[0]));
  tensorInputx.SetDataType(dtypes[0]);
  TENSOR_INPUT(opParas, tensorInputx, x);

  TensorDesc tensorInputSizeSplits;
  tensorInputSizeSplits.SetShape(ge::Shape(input_shapes[1]));
  tensorInputSizeSplits.SetDataType(dtypes[1]);
  TENSOR_INPUT_CONST(opParas, tensorInputSizeSplits, size_splits, (const uint8_t*)size_splits.data(),
                     size_splits.size() * 4);

  TensorDesc tensorInputSplitDim;
  tensorInputSplitDim.SetShape(ge::Shape(input_shapes[2]));
  tensorInputSplitDim.SetDataType(dtypes[2]);
  TENSOR_INPUT_CONST(opParas, tensorInputSplitDim, split_dim, (const uint8_t*)split_dim.data(), split_dim.size() * 4);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TensorDesc tensorOutput;
    tensorOutput.SetShape(ge::Shape(output_shapes[i]));
    tensorOutput.SetDataType(dtypes[0]);
    TENSOR_OUTPUT(opParas, tensorOutput, y);
  }

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":126976, \"num_split\":3}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "5 32 112350 6 592 373 0 592 592 0 373 373 1 18725 6 0 0 0 0 0 0 0 0 0 ");
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second(opParas, op_compile_info, runInfo);
  }
}

TEST_F(SplitVTiling, SplitV_tiling7) {
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find("SplitV");
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  auto opParas = op::SplitV("SplitV");
  vector<vector<int64_t>> input_shapes = {
      {48000, 256},
      {7},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {48000, 80}, {48000, 80}, {48000, 80}, {48000, 1}, {48000, 1}, {48000, 1}, {48000, 13},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> size_splits{80, 80, 80, 1, 1, 1, 13};
  std::vector<int32_t> split_dim{-1};

  TensorDesc tensorInputx;
  tensorInputx.SetShape(ge::Shape(input_shapes[0]));
  tensorInputx.SetDataType(dtypes[0]);
  TENSOR_INPUT(opParas, tensorInputx, x);

  TensorDesc tensorInputSizeSplits;
  tensorInputSizeSplits.SetShape(ge::Shape(input_shapes[1]));
  tensorInputSizeSplits.SetDataType(dtypes[1]);
  TENSOR_INPUT_CONST(opParas, tensorInputSizeSplits, size_splits, (const uint8_t*)size_splits.data(),
                     size_splits.size() * 4);

  TensorDesc tensorInputSplitDim;
  tensorInputSplitDim.SetShape(ge::Shape(input_shapes[2]));
  tensorInputSplitDim.SetDataType(dtypes[2]);
  TENSOR_INPUT_CONST(opParas, tensorInputSplitDim, split_dim, (const uint8_t*)split_dim.data(), split_dim.size() * 4);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TensorDesc tensorOutput;
    tensorOutput.SetShape(ge::Shape(output_shapes[i]));
    tensorOutput.SetDataType(dtypes[0]);
    TENSOR_OUTPUT(opParas, tensorOutput, y);
  }

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":126976, \"num_split\":7}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "6 32 12288000 256 1504 1376 11 96 128 10 96 128 1 48000 256 0 0 0 0 0 0 0 0 0 ");
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second(opParas, op_compile_info, runInfo);
  }
}

TEST_F(SplitVTiling, SplitV_tiling8) {
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find("SplitV");
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  auto opParas = op::SplitV("SplitV");
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
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> size_splits{2, 2, 1, 80};
  std::vector<int32_t> split_dim{-1};

  TensorDesc tensorInputx;
  tensorInputx.SetShape(ge::Shape(input_shapes[0]));
  tensorInputx.SetDataType(dtypes[0]);
  TENSOR_INPUT(opParas, tensorInputx, x);

  TensorDesc tensorInputSizeSplits;
  tensorInputSizeSplits.SetShape(ge::Shape(input_shapes[1]));
  tensorInputSizeSplits.SetDataType(dtypes[1]);
  TENSOR_INPUT_CONST(opParas, tensorInputSizeSplits, size_splits, (const uint8_t*)size_splits.data(),
                     size_splits.size() * 4);

  TensorDesc tensorInputSplitDim;
  tensorInputSplitDim.SetShape(ge::Shape(input_shapes[2]));
  tensorInputSplitDim.SetDataType(dtypes[2]);
  TENSOR_INPUT_CONST(opParas, tensorInputSplitDim, split_dim, (const uint8_t*)split_dim.data(), split_dim.size() * 4);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TensorDesc tensorOutput;
    tensorOutput.SetShape(ge::Shape(output_shapes[i]));
    tensorOutput.SetDataType(dtypes[0]);
    TENSOR_OUTPUT(opParas, tensorOutput, y);
  }

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":126976, \"num_split\":4}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "7 32 3400000 85 1280 320 5 0 256 1 64 256 1 40000 85 0 0 0 0 0 0 0 0 0 ");
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second(opParas, op_compile_info, runInfo);
  }
}

TEST_F(SplitVTiling, SplitV_tiling9) {
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find("SplitV");
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  auto opParas = op::SplitV("SplitV");
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
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> size_splits{32, 32, 16, 1280};
  std::vector<int32_t> split_dim{-1};

  TensorDesc tensorInputx;
  tensorInputx.SetShape(ge::Shape(input_shapes[0]));
  tensorInputx.SetDataType(dtypes[0]);
  tensorInputx.SetFormat(ge::FORMAT_FRACTAL_NZ);
  TENSOR_INPUT(opParas, tensorInputx, x);

  TensorDesc tensorInputSizeSplits;
  tensorInputSizeSplits.SetShape(ge::Shape(input_shapes[1]));
  tensorInputSizeSplits.SetDataType(dtypes[1]);
  tensorInputSizeSplits.SetFormat(ge::FORMAT_FRACTAL_NZ);
  TENSOR_INPUT_CONST(opParas, tensorInputSizeSplits, size_splits, (const uint8_t*)size_splits.data(),
                     size_splits.size() * 4);

  TensorDesc tensorInputSplitDim;
  tensorInputSplitDim.SetShape(ge::Shape(input_shapes[2]));
  tensorInputSplitDim.SetDataType(dtypes[2]);
  tensorInputSplitDim.SetFormat(ge::FORMAT_FRACTAL_NZ);
  TENSOR_INPUT_CONST(opParas, tensorInputSplitDim, split_dim, (const uint8_t*)split_dim.data(), split_dim.size() * 4);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TensorDesc tensorOutput;
    tensorOutput.SetShape(ge::Shape(output_shapes[i]));
    tensorOutput.SetDataType(dtypes[0]);
    tensorOutput.SetFormat(ge::FORMAT_FRACTAL_NZ);
    TENSOR_OUTPUT(opParas, tensorOutput, y);
  }

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":126976, \"num_split\":4}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), to_string(runInfo.GetAllTilingData()));
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second(opParas, op_compile_info, runInfo);
  }
}
