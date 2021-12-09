#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "selection_ops.h"
#include "array_ops.h"

using namespace std;

class ReverseV2Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ReverseV2Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ReverseV2Tiling TearDown" << std::endl;
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
#include "common/utils/ut_op_util.h"
using namespace ut_util;
/*
.INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                          DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                          DT_COMPLEX64, DT_COMPLEX128, DT_STRING}))
    .INPUT(axis, TensorType({DT_INT32,DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                           DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                           DT_COMPLEX64, DT_COMPLEX128, DT_STRING}))
*/

TEST_F(ReverseV2Tiling, rReverseV2_tiling_0) {
  std::string op_name = "ReverseV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"topk_threshold\": 16, \"max_elements\": 61440, \"max_elements_last_large_size\": 512, "
      "\"core_num\": 32, \"dtype_rate\": "
      "2}}";

  std::vector<int64_t> input{200, 200, 200, 4};
  std::vector<int64_t> axes_shape{1};
  std::vector<int32_t> axes_value{2};
  std::vector<int64_t> output{200, 200, 200, 4};

  auto opParas = op::ReverseV2("ReverseV2");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, axis, axes_shape, ge::DT_INT32, ge::FORMAT_ND, axes_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "11 1 1 1 1 1250 200 8 0 0 0 0 0 1 0 1 1 1 1 1 1 32 0 0 0 0 0 0 0 0 19 1250 3 1 ");
}

TEST_F(ReverseV2Tiling, rReverseV2_tiling_1) {
  std::string op_name = "ReverseV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"topk_threshold\": 16, \"max_elements\": 61440, \"max_elements_last_large_size\": 512, "
      "\"core_num\": 32, \"dtype_rate\": "
      "2}}";

  std::vector<int64_t> input{200, 200, 200, 200};
  std::vector<int64_t> axes_shape{2};
  std::vector<int32_t> axes_value{0, 2};
  std::vector<int64_t> output{200, 200, 200, 200};

  auto opParas = op::ReverseV2("ReverseV2");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, axis, axes_shape, ge::DT_INT32, ge::FORMAT_ND, axes_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "1 1 1 1 1 1 200 400 0 0 0 0 0 1 0 1 1 1 1 25 8 200 0 0 0 0 1 1 0 1 153 200 2 3 ");
}

TEST_F(ReverseV2Tiling, rReverseV2_tiling_2) {
  std::string op_name = "ReverseV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"topk_threshold\": 16, \"max_elements\": 61440, \"max_elements_last_large_size\": 512, "
      "\"core_num\": 32, \"dtype_rate\": "
      "2}}";

  std::vector<int64_t> input{200, 200, 200, 23};
  std::vector<int64_t> axes_shape{1};
  std::vector<int32_t> axes_value{2};
  std::vector<int64_t> output{200, 200, 200, 23};

  auto opParas = op::ReverseV2("ReverseV2");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, axis, axes_shape, ge::DT_INT32, ge::FORMAT_ND, axes_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "2 1 1 1 1 1250 200 46 0 0 0 0 0 1 0 1 1 1 1 1 1 32 0 0 0 0 0 0 0 0 6 1250 3 1 ");
}

TEST_F(ReverseV2Tiling, rReverseV2_tiling_3) {
  std::string op_name = "ReverseV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"topk_threshold\": 16, \"max_elements\": 61440, \"max_elements_last_large_size\": 512, "
      "\"core_num\": 32, \"dtype_rate\": "
      "2}}";

  std::vector<int64_t> input{200, 200, 200, 6400};
  std::vector<int64_t> axes_shape{1};
  std::vector<int32_t> axes_value{2};
  std::vector<int64_t> output{200, 200, 200, 6400};

  auto opParas = op::ReverseV2("ReverseV2");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, axis, axes_shape, ge::DT_INT32, ge::FORMAT_ND, axes_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "3 1 1 1 1 1 1 12800 0 0 0 0 0 0 0 1 1 1 1 32 1250 200 0 0 0 0 0 0 1 0 512 12800 1 3 ");
}

TEST_F(ReverseV2Tiling, rReverseV2_tiling_4) {
  std::string op_name = "ReverseV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"topk_threshold\": 16, \"max_elements\": 61440, \"max_elements_last_large_size\": 512, "
      "\"core_num\": 32, \"dtype_rate\": "
      "1}}";

  std::vector<int64_t> input{64, 64, 64, 4};
  std::vector<int64_t> axes_shape{3};
  std::vector<int64_t> axes_value{0, 1, 3};
  std::vector<int64_t> output{66, 66, 66, 4};

  auto opParas = op::ReverseV2("ReverseV2");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, axis, axes_shape, ge::DT_INT64, ge::FORMAT_ND, axes_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "11 1 1 1 1 128 64 4 0 0 0 0 1 0 1 1 1 1 1 1 1 32 0 0 0 0 0 0 1 1 60 128 3 1 ");
}

TEST_F(ReverseV2Tiling, rReverseV2_tiling_5) {
  std::string op_name = "ReverseV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"topk_threshold\": 16, \"max_elements\": 61440, \"max_elements_last_large_size\": 512, "
      "\"core_num\": 32, \"dtype_rate\": "
      "1}}";

  std::vector<int64_t> input{64, 64, 64, 129};
  std::vector<int64_t> axes_shape{3};
  std::vector<int64_t> axes_value{3};
  std::vector<int64_t> output{66, 66, 66, 129};

  auto opParas = op::ReverseV2("ReverseV2");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, axis, axes_shape, ge::DT_INT64, ge::FORMAT_ND, axes_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "5 1 1 1 1 1 8192 129 0 0 0 0 0 0 1 1 1 1 1 1 1 32 0 0 0 0 0 0 0 0 240 8192 2 1 ");
}

TEST_F(ReverseV2Tiling, rReverseV2_tiling_6) {
  std::string op_name = "ReverseV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"topk_threshold\": 16, \"max_elements\": 61440, \"max_elements_last_large_size\": 512, "
      "\"core_num\": 32, \"dtype_rate\": "
      "1}}";

  std::vector<int64_t> input{64, 64, 64, 6400};
  std::vector<int64_t> axes_shape{3};
  std::vector<int64_t> axes_value{3};
  std::vector<int64_t> output{66, 66, 66, 6400};

  auto opParas = op::ReverseV2("ReverseV2");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, axis, axes_shape, ge::DT_INT64, ge::FORMAT_ND, axes_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "6 1 1 1 1 1 1 6400 0 0 0 0 0 0 1 1 1 1 1 1 32 8192 0 0 0 0 0 0 0 1 512 6400 1 2 ");
}

TEST_F(ReverseV2Tiling, rReverseV2_only_one_dim) {
  std::string op_name = "ReverseV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"topk_threshold\": 16, \"max_elements\": 61440, \"max_elements_last_large_size\": 512, "
      "\"core_num\": 32, \"dtype_rate\": "
      "1}}";

  std::vector<int64_t> input{2};
  std::vector<int64_t> axes_shape{1};
  std::vector<int32_t> axes_value{0};
  std::vector<int64_t> output{2};

  auto opParas = op::ReverseV2("ReverseV2");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, axis, axes_shape, ge::DT_INT32, ge::FORMAT_ND, axes_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "4 1 1 1 1 1 1 2 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 1 2 2 1 0 ");
}

TEST_F(ReverseV2Tiling, rReverseV2_only_one_dim_1) {
  std::string op_name = "ReverseV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"topk_threshold\": 16, \"max_elements\": 61440, \"max_elements_last_large_size\": 512, "
      "\"core_num\": 32, \"dtype_rate\": "
      "1}}";

  std::vector<int64_t> input{1};
  std::vector<int64_t> axes_shape{1};
  std::vector<int32_t> axes_value{0};
  std::vector<int64_t> output{1};

  auto opParas = op::ReverseV2("ReverseV2");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, axis, axes_shape, ge::DT_INT32, ge::FORMAT_ND, axes_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "0 1 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 ");
}
TEST_F(ReverseV2Tiling, rReverseV2_only_one_dim_scalar) {
  std::string op_name = "ReverseV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"topk_threshold\": 16, \"max_elements\": 61440, \"max_elements_last_large_size\": 512, "
      "\"core_num\": 32, \"dtype_rate\": "
      "1}}";

  std::vector<int64_t> input;
  std::vector<int64_t> axes_shape{1};
  std::vector<int32_t> axes_value{0};
  std::vector<int64_t> output;

  auto opParas = op::ReverseV2("ReverseV2");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, axis, axes_shape, ge::DT_INT32, ge::FORMAT_ND, axes_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "0 1 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 ");
}
TEST_F(ReverseV2Tiling, rReverseV2_test_1) {
  std::string op_name = "ReverseV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"topk_threshold\": 16, \"max_elements\": 61440, \"max_elements_last_large_size\": 512, "
      "\"core_num\": 32, \"dtype_rate\": "
      "1}}";

  std::vector<int64_t> input{1, 1, 31, 36};
  std::vector<int64_t> axes_shape{4};
  std::vector<int32_t> axes_value{0, 1, 2, 3};
  std::vector<int64_t> output{1, 1, 31, 36};

  auto opParas = op::ReverseV2("ReverseV2");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, axis, axes_shape, ge::DT_INT32, ge::FORMAT_ND, axes_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "4 1 1 1 1 31 1 36 0 0 0 0 1 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 1 31 31 3 0 ");
}

TEST_F(ReverseV2Tiling, rReverseV2_test_big_first) {
  std::string op_name = "ReverseV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"topk_threshold\": 16, \"max_elements\": 61440, \"max_elements_last_large_size\": 512, "
      "\"core_num\": 32, \"dtype_rate\": "
      "2}}";

  std::vector<int64_t> input{1, 79, 79, 3};
  std::vector<int64_t> axes_shape{1};
  std::vector<int32_t> axes_value{3};
  std::vector<int64_t> output{1, 79, 79, 3};

  auto opParas = op::ReverseV2("ReverseV2");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, axis, axes_shape, ge::DT_INT32, ge::FORMAT_ND, axes_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "11 1 1 1 1 79 3 2 0 0 0 0 0 1 0 1 1 1 1 1 1 79 0 0 0 0 0 0 0 0 79 79 3 1 ");
}
