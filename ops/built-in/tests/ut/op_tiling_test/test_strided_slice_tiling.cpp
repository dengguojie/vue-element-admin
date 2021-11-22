#include <stdlib.h>
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

class stried_slice_tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "stried_slice_tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "stried_slice_tiling TearDown" << std::endl;
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

TEST_F(stried_slice_tiling, stried_slice_tiling_no_mask) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::StridedSlice("StridedSlice");
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {4},
      {4},
      {4},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);
  TensorDesc tensorInput3(ge::Shape(input_shapes[3]), ge::FORMAT_ND, dtypes[3]);

  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);

  vector<int32_t> begin = {1, 1, 1, 1};
  vector<int32_t> end = {3, 3, 3, 3};
  vector<int32_t> strides = {1, 1, 1, 1};

  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, begin, (const uint8_t*)begin.data(), begin.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, end, (const uint8_t*)end.data(), end.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput3, strides, (const uint8_t*)strides.data(), strides.size() * sizeof(int32_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);
  opParas.SetAttr("begin_mask", 0);
  opParas.SetAttr("end_mask", 0);
  opParas.SetAttr("ellipsis_mask", 0);
  opParas.SetAttr("new_axis_mask", 0);
  opParas.SetAttr("shrink_axis_mask", 0);

  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 262144}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  std::cout << to_string(runInfo.GetAllTilingData()) << std::endl;
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 4 4 4 4 4 2 2 2 2 1 1 1 1 3 3 3 3 1 1 1 1 ");
  int64_t num = 100;
  for (int64_t i = 0; i < num; i++) {
    iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  }
}

TEST_F(stried_slice_tiling, stried_slice_tiling_with_mask1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::StridedSlice("StridedSlice");
  vector<vector<int64_t>> input_shapes = {
      {10, 10, 3, 1, 2},
      {2},
      {2},
      {2},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);
  TensorDesc tensorInput3(ge::Shape(input_shapes[3]), ge::FORMAT_ND, dtypes[3]);

  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);

  vector<int32_t> begin = {0, 0};
  vector<int32_t> end = {3, 3};
  vector<int32_t> strides = {1, 1};

  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, begin, (const uint8_t*)begin.data(), begin.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, end, (const uint8_t*)end.data(), end.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput3, strides, (const uint8_t*)strides.data(), strides.size() * sizeof(int32_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);
  opParas.SetAttr("begin_mask", 0);
  opParas.SetAttr("end_mask", 0);
  opParas.SetAttr("ellipsis_mask", 1);
  opParas.SetAttr("new_axis_mask", 0);
  opParas.SetAttr("shrink_axis_mask", 2);

  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 1, "new_axis_mask": 0, "shrink_axis_mask": 2, "ub_size": 262144}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "5 2 300 2 300 1 0 0 300 1 1 1 ");
}

TEST_F(stried_slice_tiling, stried_slice_tiling_int64_const) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::StridedSlice("StridedSlice");
  vector<vector<int64_t>> input_shapes = {
      {10, 10, 3, 1, 2},
      {2},
      {2},
      {2},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);
  TensorDesc tensorInput3(ge::Shape(input_shapes[3]), ge::FORMAT_ND, dtypes[3]);

  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);

  vector<int64_t> begin = {0, 0};
  vector<int64_t> end = {3, 3};
  vector<int64_t> strides = {1, 1};

  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, begin, (const uint8_t*)begin.data(), begin.size() * sizeof(int64_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, end, (const uint8_t*)end.data(), end.size() * sizeof(int64_t));
  TENSOR_INPUT_CONST(opParas, tensorInput3, strides, (const uint8_t*)strides.data(), strides.size() * sizeof(int64_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);
  opParas.SetAttr("begin_mask", 0);
  opParas.SetAttr("end_mask", 0);
  opParas.SetAttr("ellipsis_mask", 1);
  opParas.SetAttr("new_axis_mask", 0);
  opParas.SetAttr("shrink_axis_mask", 2);

  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 1, "new_axis_mask": 0, "shrink_axis_mask": 2, "ub_size": 262144}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "5 2 300 2 300 1 0 0 300 1 1 1 ");
}

TEST_F(stried_slice_tiling, stried_slice_no_mask) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::StridedSlice("StridedSlice");
  vector<vector<int64_t>> input_shapes = {
      {10, 10, 3, 1, 2},
      {2},
      {2},
      {2},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);
  TensorDesc tensorInput3(ge::Shape(input_shapes[3]), ge::FORMAT_ND, dtypes[3]);

  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);

  vector<int32_t> begin = {0, 0};
  vector<int32_t> end = {3, 3};
  vector<int32_t> strides = {1, 1};

  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, begin, (const uint8_t*)begin.data(), begin.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, end, (const uint8_t*)end.data(), end.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput3, strides, (const uint8_t*)strides.data(), strides.size() * sizeof(int32_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);
  opParas.SetAttr("begin_mask", 0);
  opParas.SetAttr("end_mask", 0);
  opParas.SetAttr("ellipsis_mask", 1);
  opParas.SetAttr("new_axis_mask", 0);
  opParas.SetAttr("shrink_axis_mask", 2);

  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "end_mask": 0, "ellipsis_mask": 1, "new_axis_mask": 0, "shrink_axis_mask": 2}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  EXPECT_FALSE(ret);
}

TEST_F(stried_slice_tiling, stried_slice_tiling_no_inputs) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::StridedSlice("StridedSlice");
  vector<vector<int64_t>> input_shapes = {
      {10, 10, 3, 1, 2},
      {2},
      {2},
      {2},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);
  TensorDesc tensorInput3(ge::Shape(input_shapes[3]), ge::FORMAT_ND, dtypes[3]);

  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);

  vector<int32_t> begin = {0, 0};
  vector<int32_t> end = {3, 3};
  vector<int32_t> strides = {1, 1};

  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, begin, (const uint8_t*)begin.data(), begin.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, end, (const uint8_t*)end.data(), end.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput3, strides, (const uint8_t*)strides.data(), strides.size() * sizeof(int32_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);
  opParas.SetAttr("begin_mask", 0);
  opParas.SetAttr("end_mask", 0);
  opParas.SetAttr("ellipsis_mask", 1);
  opParas.SetAttr("new_axis_mask", 0);
  opParas.SetAttr("shrink_axis_mask", 2);

  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 1, "new_axis_mask": 0, "shrink_axis_mask": 2}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  ASSERT_FALSE(ret);
}

TEST_F(stried_slice_tiling, stried_slice_tiling_too_large_dims) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::StridedSlice("StridedSlice");
  vector<vector<int64_t>> input_shapes = {
      {10, 10, 3, 1, 2, 2, 2, 3, 3},
      {2},
      {2},
      {2},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);
  TensorDesc tensorInput3(ge::Shape(input_shapes[3]), ge::FORMAT_ND, dtypes[3]);

  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);

  vector<int32_t> begin = {0, 0};
  vector<int32_t> end = {3, 3};
  vector<int32_t> strides = {1, 1};

  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, begin, (const uint8_t*)begin.data(), begin.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, end, (const uint8_t*)end.data(), end.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput3, strides, (const uint8_t*)strides.data(), strides.size() * sizeof(int32_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);
  opParas.SetAttr("begin_mask", 0);
  opParas.SetAttr("end_mask", 0);
  opParas.SetAttr("ellipsis_mask", 1);
  opParas.SetAttr("new_axis_mask", 0);
  opParas.SetAttr("shrink_axis_mask", 2);

  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 1, "new_axis_mask": 0, "shrink_axis_mask": 2}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  ASSERT_FALSE(ret);
}

TEST_F(stried_slice_tiling, stried_slice_tiling_get_const_value_failed) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::StridedSlice("StridedSlice");
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {4},
      {4},
      {4},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);
  TensorDesc tensorInput3(ge::Shape(input_shapes[3]), ge::FORMAT_ND, dtypes[3]);

  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);

  vector<float> begin = {1, 1, 1, 1};
  vector<float> end = {3, 3, 3, 3};
  vector<float> strides = {1, 1, 1, 1};

  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, begin, (const uint8_t*)begin.data(), begin.size() * sizeof(float));
  TENSOR_INPUT_CONST(opParas, tensorInput2, end, (const uint8_t*)end.data(), end.size() * sizeof(float));
  TENSOR_INPUT_CONST(opParas, tensorInput3, strides, (const uint8_t*)strides.data(), strides.size() * sizeof(float));
  TENSOR_OUTPUT(opParas, tensorOutput, y);
  opParas.SetAttr("begin_mask", 0);
  opParas.SetAttr("end_mask", 0);
  opParas.SetAttr("ellipsis_mask", 0);
  opParas.SetAttr("new_axis_mask", 0);
  opParas.SetAttr("shrink_axis_mask", 0);

  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  ASSERT_FALSE(ret);
}

TEST_F(stried_slice_tiling, stried_slice_tiling_invalid_stride) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::StridedSlice("StridedSlice");
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {4},
      {4},
      {4},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);
  TensorDesc tensorInput3(ge::Shape(input_shapes[3]), ge::FORMAT_ND, dtypes[3]);

  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);

  vector<int32_t> begin = {1, 1, 1, 1};
  vector<int32_t> end = {3, 3, 3, 3};
  vector<int32_t> strides = {1, 1, 1, 0};

  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, begin, (const uint8_t*)begin.data(), begin.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, end, (const uint8_t*)end.data(), end.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput3, strides, (const uint8_t*)strides.data(), strides.size() * sizeof(int32_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);
  opParas.SetAttr("begin_mask", 0);
  opParas.SetAttr("end_mask", 0);
  opParas.SetAttr("ellipsis_mask", 0);
  opParas.SetAttr("new_axis_mask", 0);
  opParas.SetAttr("shrink_axis_mask", 0);

  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  ASSERT_FALSE(ret);
}

TEST_F(stried_slice_tiling, stried_slice_tiling_unsupported_stride) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::StridedSlice("StridedSlice");

  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {4},
      {4},
      {4},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);
  TensorDesc tensorInput3(ge::Shape(input_shapes[3]), ge::FORMAT_ND, dtypes[3]);

  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);

  vector<int32_t> begin = {1, 1, 1, 1};
  vector<int32_t> end = {3, 3, 3, 3};
  vector<int32_t> strides = {1, 1, 2, 1};

  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, begin, (const uint8_t*)begin.data(), begin.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, end, (const uint8_t*)end.data(), end.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput3, strides, (const uint8_t*)strides.data(), strides.size() * sizeof(int32_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);
  opParas.SetAttr("begin_mask", 0);
  opParas.SetAttr("end_mask", 0);
  opParas.SetAttr("ellipsis_mask", 0);
  opParas.SetAttr("new_axis_mask", 0);
  opParas.SetAttr("shrink_axis_mask", 0);

  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  ASSERT_FALSE(ret);
}

TEST_F(stried_slice_tiling, stried_slice_tiling_fused_dims) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::StridedSlice("StridedSlice");
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {4},
      {4},
      {4},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);
  TensorDesc tensorInput3(ge::Shape(input_shapes[3]), ge::FORMAT_ND, dtypes[3]);

  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);

  vector<int32_t> begin = {1, 1, 1, 0};
  vector<int32_t> end = {3, 3, 3, 4};
  vector<int32_t> strides = {1, 1, 1, 1};

  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, begin, (const uint8_t*)begin.data(), begin.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, end, (const uint8_t*)end.data(), end.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput3, strides, (const uint8_t*)strides.data(), strides.size() * sizeof(int32_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);
  opParas.SetAttr("begin_mask", 0);
  opParas.SetAttr("end_mask", 0);
  opParas.SetAttr("ellipsis_mask", 0);
  opParas.SetAttr("new_axis_mask", 0);
  opParas.SetAttr("shrink_axis_mask", 0);

  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 262144}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 3 4 4 16 2 2 8 1 1 4 3 3 12 1 1 1 ");
}

TEST_F(stried_slice_tiling, stried_slice_tiling_mode_3) {
   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::StridedSlice("StridedSlice");
  vector<vector<int64_t>> input_shapes = {
      {1, 5, 5, 5, 424, 35},
      {6},
      {6},
      {6},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);
  TensorDesc tensorInput3(ge::Shape(input_shapes[3]), ge::FORMAT_ND, dtypes[3]);

  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);
  vector<int32_t> begin = {0, 0, 1, 1, 233, 30};
  vector<int32_t> end = {1, 2, 5, 5, 423, 35};
  vector<int32_t> strides = {1, 1, 1, 1, 1, 1};
  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, begin, (const uint8_t*)begin.data(), begin.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, end, (const uint8_t*)end.data(), end.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput3, strides, (const uint8_t*)strides.data(), strides.size() * sizeof(int32_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);
  opParas.SetAttr("begin_mask", 0);
  opParas.SetAttr("end_mask", 0);
  opParas.SetAttr("ellipsis_mask", 0);
  opParas.SetAttr("new_axis_mask", 0);
  opParas.SetAttr("shrink_axis_mask", 0);

  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 261762}})";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "3 6 1 5 5 5 424 35 1 2 4 4 190 5 0 0 1 1 233 30 1 2 5 5 423 35 1 1 1 1 1 1 ");
}

TEST_F(stried_slice_tiling, stried_slice_tiling_mode_5) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::StridedSlice("StridedSlice");
  vector<vector<int64_t>> input_shapes = {
      {800, 800, 9},
      {3},
      {3},
      {3},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);
  TensorDesc tensorInput3(ge::Shape(input_shapes[3]), ge::FORMAT_ND, dtypes[3]);
  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);
  vector<int32_t> begin = {0, 0, 1};
  vector<int32_t> end = {800, 800, 9};
  vector<int32_t> strides = {1, 1, 1};
  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, begin, (const uint8_t*)begin.data(), begin.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, end, (const uint8_t*)end.data(), end.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput3, strides, (const uint8_t*)strides.data(), strides.size() * sizeof(int32_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);
  opParas.SetAttr("begin_mask", 0);
  opParas.SetAttr("end_mask", 0);
  opParas.SetAttr("ellipsis_mask", 0);
  opParas.SetAttr("new_axis_mask", 0);
  opParas.SetAttr("shrink_axis_mask", 0);

  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 261762}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "5 2 640000 9 640000 8 0 1 640000 9 1 1 ");
}

TEST_F(stried_slice_tiling, stried_slice_tiling_mode_6) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::StridedSlice("StridedSlice");
  vector<vector<int64_t>> input_shapes = {
      {10, 3944, 792},
      {3},
      {3},
      {3},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);
  TensorDesc tensorInput3(ge::Shape(input_shapes[3]), ge::FORMAT_ND, dtypes[3]);
  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);
  vector<int32_t> begin = {0, 0, 1};
  vector<int32_t> end = {10, 3944, 792};
  vector<int32_t> strides = {1, 1, 1};
  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, begin, (const uint8_t*)begin.data(), begin.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, end, (const uint8_t*)end.data(), end.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput3, strides, (const uint8_t*)strides.data(), strides.size() * sizeof(int32_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);
  opParas.SetAttr("begin_mask", 0);
  opParas.SetAttr("end_mask", 0);
  opParas.SetAttr("ellipsis_mask", 0);
  opParas.SetAttr("new_axis_mask", 0);
  opParas.SetAttr("shrink_axis_mask", 0);

  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 261762}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "6 2 39440 792 39440 791 0 1 39440 792 1 1 ");
}

TEST_F(stried_slice_tiling, stried_slice_tiling_mode_7) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::StridedSlice("StridedSlice");
  vector<vector<int64_t>> input_shapes = {
      {2, 2, 65536},
      {3},
      {3},
      {3},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);
  TensorDesc tensorInput3(ge::Shape(input_shapes[3]), ge::FORMAT_ND, dtypes[3]);
  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);
  vector<int32_t> begin = {0, 0, 0};
  vector<int32_t> end = {2, 2, 65536};
  vector<int32_t> strides = {1, 1, 1};
  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, begin, (const uint8_t*)begin.data(), begin.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, end, (const uint8_t*)end.data(), end.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput3, strides, (const uint8_t*)strides.data(), strides.size() * sizeof(int32_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);
  opParas.SetAttr("begin_mask", 0);
  opParas.SetAttr("end_mask", 0);
  opParas.SetAttr("ellipsis_mask", 0);
  opParas.SetAttr("new_axis_mask", 0);
  opParas.SetAttr("shrink_axis_mask", 0);

  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 261762}})";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "7 1 262144 262144 0 262144 1 ");
}

TEST_F(stried_slice_tiling, stried_slice_tiling_mode_8) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::StridedSlice("StridedSlice");
  vector<vector<int64_t>> input_shapes = {
      {1, 3193, 264},
      {3},
      {3},
      {3},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);
  TensorDesc tensorInput3(ge::Shape(input_shapes[3]), ge::FORMAT_ND, dtypes[3]);
  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);
  vector<int32_t> begin = {0, 0, 0};
  vector<int32_t> end = {1, 3193, 1};
  vector<int32_t> strides = {1, 1, 1};
  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, begin, (const uint8_t*)begin.data(), begin.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, end, (const uint8_t*)end.data(), end.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput3, strides, (const uint8_t*)strides.data(), strides.size() * sizeof(int32_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);
  opParas.SetAttr("begin_mask", 0);
  opParas.SetAttr("end_mask", 0);
  opParas.SetAttr("ellipsis_mask", 0);
  opParas.SetAttr("new_axis_mask", 0);
  opParas.SetAttr("shrink_axis_mask", 0);

  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 261762}})";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "8 2 3193 264 3193 1 0 0 3193 1 1 1 ");
}

TEST_F(stried_slice_tiling, stried_slice_outshape_0) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("StridedSlice");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::StridedSlice("StridedSlice");
  vector<vector<int64_t>> input_shapes = {
      {38, 38},
      {2},
      {2},
      {2},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);
  TensorDesc tensorInput3(ge::Shape(input_shapes[3]), ge::FORMAT_ND, dtypes[3]);
  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);
  vector<int32_t> begin = {1, 1};
  vector<int32_t> end = {1, 1};
  vector<int32_t> strides = {1, 1};
  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, begin, (const uint8_t*)begin.data(), begin.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, end, (const uint8_t*)end.data(), end.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput3, strides, (const uint8_t*)strides.data(), strides.size() * sizeof(int32_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);
  opParas.SetAttr("begin_mask", 0);
  opParas.SetAttr("end_mask", 0);
  opParas.SetAttr("ellipsis_mask", 0);
  opParas.SetAttr("new_axis_mask", 0);
  opParas.SetAttr("shrink_axis_mask", 0);

  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 261762}})";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 2 38 38 0 0 1 1 1 1 1 1 ");
}
