#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include <register/op_tiling.h>
#include "test_common.h"
#include "array_ops.h"
#include "selection_ops.h"

using namespace std;
using namespace ge;

class slice_tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "slice_tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "slice_tiling TearDown" << std::endl;
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

TEST_F(slice_tiling, slice_tiling_no_mask) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Slice");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  auto opParas = op::Slice("Slice");
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {4},
      {4},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);

  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);

  vector<int32_t> offset = {1, 1, 1, 1};
  vector<int32_t> size = {2, 2, 2, 2};
  vector<int32_t> strides = {1, 1, 1, 1};

  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, offsets, (const uint8_t*)offset.data(), offset.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, size, (const uint8_t*)size.data(), size.size() * sizeof(int32_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);

  std::string compileInfo = R"({"vars": {"block_dim": 32}})";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 4 4 4 4 4 2 2 2 2 1 1 1 1 3 3 3 3 1 1 1 1 ");
  int64_t num = 100;
  for (int64_t i = 0; i < num; i++) {
    iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  }
}

TEST_F(slice_tiling, slice_tiling_with_mask1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Slice");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::Slice("Slice");
  vector<vector<int64_t>> input_shapes = {
      {10, 10},
      {2},
      {2},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);

  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);

  vector<int32_t> offset = {0, 0};
  vector<int32_t> size = {3, 3};
  vector<int32_t> strides = {1, 1};

  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, offsets, (const uint8_t*)offset.data(), offset.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, size, (const uint8_t*)size.data(), size.size() * sizeof(int32_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);

  std::string compileInfo = R"({"vars": {"block_dim": 32}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 2 10 10 3 3 0 0 3 3 1 1 ");
}

TEST_F(slice_tiling, slice_tiling_no_const_value) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Slice");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::Slice("Slice");
  vector<vector<int64_t>> input_shapes = {
      {10, 10},
      {2},
      {2},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);

  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);

  vector<int32_t> offset = {0, 0};
  vector<int32_t> size = {3, 3};
  vector<int32_t> strides = {1, 1};

  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT(opParas, tensorInput1, offsets);
  TENSOR_INPUT(opParas, tensorInput2, size);
  TENSOR_OUTPUT(opParas, tensorOutput, y);

  std::string compileInfo = R"({"vars": {"block_dim": 32}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  ASSERT_FALSE(ret);
}

TEST_F(slice_tiling, slice_tiling_invalid_begin_length) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Slice");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::Slice("Slice");
  vector<vector<int64_t>> input_shapes = {
      {10, 10},
      {2},
      {2},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);

  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);

  vector<int32_t> offset = {0, 0, 0};
  vector<int32_t> size = {3, 3};
  vector<int32_t> strides = {1, 1};

  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, offsets, (const uint8_t*)offset.data(), offset.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, size, (const uint8_t*)size.data(), size.size() * sizeof(int32_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);

  std::string compileInfo = R"({"vars": {"block_dim": 32}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  ASSERT_FALSE(ret);
}

TEST_F(slice_tiling, slice_tiling_invalid_begin_value) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Slice");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::Slice("Slice");
  vector<vector<int64_t>> input_shapes = {
      {10, 10},
      {2},
      {2},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);

  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);

  vector<int32_t> offset = {-1, -1};
  vector<int32_t> size = {3, 3};
  vector<int32_t> strides = {1, 1};

  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, offsets, (const uint8_t*)offset.data(), offset.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, size, (const uint8_t*)size.data(), size.size() * sizeof(int32_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);

  std::string compileInfo = R"({"vars": {"block_dim": 32}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  ASSERT_FALSE(ret);
}

TEST_F(slice_tiling, slice_tiling_end_value_negative_one) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Slice");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::Slice("Slice");
  vector<vector<int64_t>> input_shapes = {
      {10, 10},
      {2},
      {2},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);

  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);

  vector<int32_t> offset = {0, 0};
  vector<int32_t> size = {-1, -1};
  vector<int32_t> strides = {1, 1};

  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, offsets, (const uint8_t*)offset.data(), offset.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, size, (const uint8_t*)size.data(), size.size() * sizeof(int32_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);
  std::string compileInfo = R"({"vars": {"block_dim": 32}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "7 1 100 100 0 100 1 ");
}

TEST_F(slice_tiling, slice_tiling_empty_input) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Slice");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::Slice("Slice");
  vector<vector<int64_t>> input_shapes = {
      {10, 10},
      {2},
      {2},
  };

  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);

  vector<int32_t> offset = {0, 0};
  vector<int32_t> size = {-1, -1};
  vector<int32_t> strides = {1, 1};

  TENSOR_OUTPUT(opParas, tensorOutput, y);
  std::string compileInfo = R"({"vars": {"block_dim": 32}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  ASSERT_FALSE(ret);
}
