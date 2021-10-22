#include <stdlib.h>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include <register/op_tiling.h>
#include "selection_ops.h"
#include "test_common.h"
#include "array_ops.h"

using namespace std;
using namespace ge;

class stried_slice_v3_tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "stried_slice_v3_tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "stried_slice_v3_tiling TearDown" << std::endl;
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

TEST_F(stried_slice_v3_tiling, stried_slice_v3_tiling_no_mask) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("StridedSliceV3");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::StridedSliceV3("StridedSliceV3");
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4}, {4}, {4}, {4}, {4},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);
  TensorDesc tensorInput3(ge::Shape(input_shapes[3]), ge::FORMAT_ND, dtypes[3]);
  TensorDesc tensorInput4(ge::Shape(input_shapes[4]), ge::FORMAT_ND, dtypes[4]);

  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);

  vector<int32_t> begin = {1, 1, 1, 1};
  vector<int32_t> end = {3, 3, 3, 3};
  vector<int32_t> strides = {1, 1, 1, 1};
  vector<int32_t> axes = {0, 1, 2, 3};

  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, begin, (const uint8_t*)begin.data(), begin.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, end, (const uint8_t*)end.data(), end.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput3, strides, (const uint8_t*)strides.data(), strides.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput4, axes, (const uint8_t*)axes.data(), axes.size() * sizeof(int32_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);

  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 262144}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  std::cout << to_string(runInfo.GetAllTilingData()) << std::endl;
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 4 4 4 4 4 2 2 2 2 1 1 1 1 3 3 3 3 1 1 1 1 ");
}

TEST_F(stried_slice_v3_tiling, stried_slice_v3_tiling_no_axes) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("StridedSliceV3");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::StridedSliceV3("StridedSliceV3");
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4}, {4}, {4}, {}, {4},
      // no axes
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
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
  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 262144}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  std::cout << to_string(runInfo.GetAllTilingData()) << std::endl;
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 4 4 4 4 4 2 2 2 2 1 1 1 1 3 3 3 3 1 1 1 1 ");
}

TEST_F(stried_slice_v3_tiling, stried_slice_v3_tiling_pad_head) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("StridedSliceV3");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::StridedSliceV3("StridedSliceV3");
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4}, {2}, {2}, {2}, {2},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);
  TensorDesc tensorInput3(ge::Shape(input_shapes[3]), ge::FORMAT_ND, dtypes[3]);
  TensorDesc tensorInput4(ge::Shape(input_shapes[4]), ge::FORMAT_ND, dtypes[4]);

  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);

  vector<int64_t> begin = {1, 1};
  vector<int64_t> end = {3, 3};
  vector<int64_t> strides = {1, 1};
  vector<int64_t> axes = {0, 1};

  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, begin, (const uint8_t*)begin.data(), begin.size() * sizeof(int64_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, end, (const uint8_t*)end.data(), end.size() * sizeof(int64_t));
  TENSOR_INPUT_CONST(opParas, tensorInput3, strides, (const uint8_t*)strides.data(), strides.size() * sizeof(int64_t));
  TENSOR_INPUT_CONST(opParas, tensorInput4, axes, (const uint8_t*)axes.data(), axes.size() * sizeof(int64_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);
  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 262144}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  std::cout << to_string(runInfo.GetAllTilingData()) << std::endl;
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            // before fused is "1 4 4 4 4 4 2 2 4 4 1 1 0 0 3 3 4 4 1 1 1 1 "
            "2 2 4 64 2 32 1 16 3 48 1 1 ");
}

TEST_F(stried_slice_v3_tiling, stried_slice_v3_tiling_pad_tail) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("StridedSliceV3");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::StridedSliceV3("StridedSliceV3");
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4}, {2}, {2}, {2}, {2},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);
  TensorDesc tensorInput3(ge::Shape(input_shapes[3]), ge::FORMAT_ND, dtypes[3]);
  TensorDesc tensorInput4(ge::Shape(input_shapes[4]), ge::FORMAT_ND, dtypes[4]);

  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);

  vector<int32_t> begin = {1, 1};
  vector<int32_t> end = {3, 3};
  vector<int32_t> strides = {1, 1};
  vector<int32_t> axes = {2, 3};

  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, begin, (const uint8_t*)begin.data(), begin.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, end, (const uint8_t*)end.data(), end.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput3, strides, (const uint8_t*)strides.data(), strides.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput4, axes, (const uint8_t*)axes.data(), axes.size() * sizeof(int32_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);

  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 262144}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  std::cout << to_string(runInfo.GetAllTilingData()) << std::endl;
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 3 16 4 4 16 2 2 0 1 1 16 3 3 1 1 1 ");
}

TEST_F(stried_slice_v3_tiling, stried_slice_v3_tiling_no_begin) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("StridedSliceV3");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::StridedSliceV3("StridedSliceV3");
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4}, {}, {4}, {4}, {4},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  //   TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);
  TensorDesc tensorInput3(ge::Shape(input_shapes[3]), ge::FORMAT_ND, dtypes[3]);
  TensorDesc tensorInput4(ge::Shape(input_shapes[4]), ge::FORMAT_ND, dtypes[4]);

  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);

  vector<int32_t> end = {3, 3, 3, 3};
  vector<int32_t> strides = {1, 1, 1, 1};
  vector<int32_t> axes = {0, 1, 2, 3};

  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput2, end, (const uint8_t*)end.data(), end.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput3, strides, (const uint8_t*)strides.data(), strides.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput4, axes, (const uint8_t*)axes.data(), axes.size() * sizeof(int32_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);

  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 262144}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  std::cout << to_string(runInfo.GetAllTilingData()) << std::endl;
  ASSERT_FALSE(ret);
}

TEST_F(stried_slice_v3_tiling, stried_slice_v3_tiling_diff_size) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("StridedSliceV3");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::StridedSliceV3("StridedSliceV3");
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4}, {1}, {4}, {4}, {4},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);
  TensorDesc tensorInput3(ge::Shape(input_shapes[3]), ge::FORMAT_ND, dtypes[3]);
  TensorDesc tensorInput4(ge::Shape(input_shapes[4]), ge::FORMAT_ND, dtypes[4]);

  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);

  vector<int32_t> begin = {
      1,
  };
  vector<int32_t> end = {3, 3, 3, 3};
  vector<int32_t> strides = {1, 1, 1, 1};
  vector<int32_t> axes = {0, 1, 2, 3};

  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, begin, (const uint8_t*)begin.data(), begin.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, end, (const uint8_t*)end.data(), end.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput3, strides, (const uint8_t*)strides.data(), strides.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput4, axes, (const uint8_t*)axes.data(), axes.size() * sizeof(int32_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);
  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 262144}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  std::cout << to_string(runInfo.GetAllTilingData()) << std::endl;
  ASSERT_FALSE(ret);
}

TEST_F(stried_slice_v3_tiling, stried_slice_v3_tiling_no_mask_neg) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("StridedSliceV3");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::StridedSliceV3("StridedSliceV3");
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4}, {4}, {4}, {4}, {4},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);
  TensorDesc tensorInput3(ge::Shape(input_shapes[3]), ge::FORMAT_ND, dtypes[3]);
  TensorDesc tensorInput4(ge::Shape(input_shapes[4]), ge::FORMAT_ND, dtypes[4]);

  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);

  vector<int32_t> begin = {1, 1, 1, -3};
  vector<int32_t> end = {3, 3, 3, -1};
  vector<int32_t> strides = {1, 1, 1, 1};
  vector<int32_t> axes = {0, 1, 2, -1};
  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, begin, (const uint8_t*)begin.data(), begin.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, end, (const uint8_t*)end.data(), end.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput3, strides, (const uint8_t*)strides.data(), strides.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput4, axes, (const uint8_t*)axes.data(), axes.size() * sizeof(int32_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);

  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 262144}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  std::cout << to_string(runInfo.GetAllTilingData()) << std::endl;
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 4 4 4 4 4 2 2 2 2 1 1 1 1 3 3 3 3 1 1 1 1 ");
}

TEST_F(stried_slice_v3_tiling, stried_slice_v3_tiling_no_stride) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("StridedSliceV3");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::StridedSliceV3("StridedSliceV3");
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4}, {4}, {4}, {4}, {},  // no stride
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
  TensorDesc tensorInput0(ge::Shape(input_shapes[0]), ge::FORMAT_ND, dtypes[0]);
  TensorDesc tensorInput1(ge::Shape(input_shapes[1]), ge::FORMAT_ND, dtypes[1]);
  TensorDesc tensorInput2(ge::Shape(input_shapes[2]), ge::FORMAT_ND, dtypes[2]);
  TensorDesc tensorInput3(ge::Shape(input_shapes[3]), ge::FORMAT_ND, dtypes[3]);
  TensorDesc tensorInput4(ge::Shape(input_shapes[4]), ge::FORMAT_ND, dtypes[4]);

  TensorDesc tensorOutput;
  tensorOutput.SetShape(ge::Shape(input_shapes[0]));
  tensorOutput.SetDataType(ge::DT_FLOAT16);
  vector<int32_t> begin = {1, 1, 1, -1000};
  vector<int32_t> end = {3, 3, 3, 3000};
  vector<int32_t> axes = {0, 1, 2, -1};

  TENSOR_INPUT(opParas, tensorInput0, x);
  TENSOR_INPUT_CONST(opParas, tensorInput1, begin, (const uint8_t*)begin.data(), begin.size() * sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput2, end, (const uint8_t*)end.data(), end.size() * sizeof(int32_t));
  //   TENSOR_INPUT_CONST(opParas, tensorInput3, strides, (const uint8_t *) strides.data(), strides.size() *
  //   sizeof(int32_t));
  TENSOR_INPUT_CONST(opParas, tensorInput4, axes, (const uint8_t*)axes.data(), axes.size() * sizeof(int32_t));
  TENSOR_OUTPUT(opParas, tensorOutput, y);
  std::string compileInfo =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 262144}})";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  auto ret = iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  std::cout << to_string(runInfo.GetAllTilingData()) << std::endl;
  ASSERT_TRUE(ret);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 3 4 4 16 2 2 8 1 1 4 3 3 12 1 1 1 ");
}
