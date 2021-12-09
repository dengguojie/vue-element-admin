#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "selection_ops.h"
#include "array_ops.h"

using namespace std;

class OneHotTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "OneHotTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "OneHotTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream& tiling_data) {
  auto data = tiling_data.str();
  string result;
  int32_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
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
.INPUT(x, TensorType({DT_UINT8, DT_INT32, DT_INT64}))
    .INPUT(depth, TensorType({DT_INT32}))
    .INPUT(on_value, TensorType::BasicType())
    .INPUT(off_value, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(axis, Int, -1)
*/

TEST_F(OneHotTiling, one_hot_tiling_0) {
  std::string op_name = "OneHot";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":-1}}";

  std::vector<int64_t> input0{3, 3, 32, 32, 16};
  std::vector<int64_t> input1{1};
  std::vector<int32_t> depth{32};
  std::vector<int32_t> off_value{1};
  std::vector<int64_t> output{3, 3, 32, 32, 16, 32};

  auto opParas = op::OneHot("OneHot");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input0, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, depth, input1, ge::DT_INT32, ge::FORMAT_ND, depth);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, off_value, input1, ge::DT_INT32, ge::FORMAT_ND, off_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_INT32, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 4608 2 32 147456 147456 1 4718592 4608 0 0 ");
}

TEST_F(OneHotTiling, one_hot_tiling_1) {
  std::string op_name = "OneHot";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":0}}";

  std::vector<int64_t> input0{2, 16, 8, 8, 16};
  std::vector<int64_t> input1{1};
  std::vector<int32_t> depth{32};
  std::vector<int32_t> off_value{1};
  std::vector<int64_t> output{32, 2, 16, 8, 8, 16};

  auto opParas = op::OneHot("OneHot");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input0, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, depth, input1, ge::DT_INT32, ge::FORMAT_ND, depth);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, off_value, input1, ge::DT_INT32, ge::FORMAT_ND, off_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_INT32, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 0 8 32 32768 1 32768 1048576 0 1 1 ");
}

TEST_F(OneHotTiling, one_hot_tiling_2) {
  std::string op_name = "OneHot";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":0}}";

  std::vector<int64_t> input0{2, 16, 8, 8, 16};
  std::vector<int64_t> input1{1};
  std::vector<int32_t> depth{32};
  std::vector<int32_t> off_value{1};
  std::vector<int64_t> output{32, 2, 16, 8, 8, 16};

  auto opParas = op::OneHot("OneHot");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input0, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, depth, input1, ge::DT_INT32, ge::FORMAT_ND, depth);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, off_value, input1, ge::DT_INT32, ge::FORMAT_ND, off_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_INT32, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 0 8 32 32768 1 32768 1048576 0 1 1 ");
}

TEST_F(OneHotTiling, one_hot_tiling_3) {
  std::string op_name = "OneHot";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":-1}}";

  std::vector<int64_t> input0{2, 16, 8, 8, 16};
  std::vector<int64_t> input1{1};
  std::vector<int32_t> depth{32};
  std::vector<int32_t> off_value{1};
  std::vector<int64_t> output{2, 16, 8, 8, 16, 32};

  auto opParas = op::OneHot("OneHot");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input0, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, depth, input1, ge::DT_INT32, ge::FORMAT_ND, depth);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, off_value, input1, ge::DT_INT32, ge::FORMAT_ND, off_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_INT32, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 1024 1 32 32768 32768 1 1048576 1024 0 0 ");
}

TEST_F(OneHotTiling, one_hot_tiling_4) {
  std::string op_name = "OneHot";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":1}}";

  std::vector<int64_t> input0{2, 16, 8, 8, 16};
  std::vector<int64_t> input1{1};
  std::vector<int32_t> depth{32};
  std::vector<int32_t> off_value{1};
  std::vector<int64_t> output{2, 32, 16, 8, 8, 16};

  auto opParas = op::OneHot("OneHot");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input0, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, depth, input1, ge::DT_INT32, ge::FORMAT_ND, depth);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, off_value, input1, ge::DT_INT32, ge::FORMAT_ND, off_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_INT32, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 1 11 2 32768 2 16384 1048576 1 0 0 ");
}

TEST_F(OneHotTiling, one_hot_tiling_5) {
  std::string op_name = "OneHot";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":2}}";

  std::vector<int64_t> input0{2, 16, 8, 8, 16};
  std::vector<int64_t> input1{1};
  std::vector<int32_t> depth{32};
  std::vector<int32_t> off_value{1};
  std::vector<int64_t> output{2, 16, 32, 8, 8, 16};

  auto opParas = op::OneHot("OneHot");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input0, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, depth, input1, ge::DT_INT32, ge::FORMAT_ND, depth);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, off_value, input1, ge::DT_INT32, ge::FORMAT_ND, off_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_INT32, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 1 10 32 32768 32 1024 1048576 1 0 0 ");
}

TEST_F(OneHotTiling, one_hot_tiling_6) {
  std::string op_name = "OneHot";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":3}}";

  std::vector<int64_t> input0{2, 16, 8, 8, 16};
  std::vector<int64_t> input1{1};
  std::vector<int32_t> depth{32};
  std::vector<int32_t> off_value{1};
  std::vector<int64_t> output{2, 16, 8, 32, 8, 16};

  auto opParas = op::OneHot("OneHot");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input0, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, depth, input1, ge::DT_INT32, ge::FORMAT_ND, depth);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, off_value, input1, ge::DT_INT32, ge::FORMAT_ND, off_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_INT32, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 8 10 32 32768 256 128 1048576 8 0 0 ");
}

TEST_F(OneHotTiling, one_hot_tiling_7) {
  std::string op_name = "OneHot";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":4}}";

  std::vector<int64_t> input0{2, 16, 8, 8, 16};
  std::vector<int64_t> input1{1};
  std::vector<int32_t> depth{32};
  std::vector<int32_t> off_value{1};
  std::vector<int64_t> output{2, 16, 8, 8, 32, 16};

  auto opParas = op::OneHot("OneHot");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input0, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, depth, input1, ge::DT_INT32, ge::FORMAT_ND, depth);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, off_value, input1, ge::DT_INT32, ge::FORMAT_ND, off_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_INT32, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 64 10 32 32768 2048 16 1048576 64 0 0 ");
}

TEST_F(OneHotTiling, one_hot_tiling_8) {
  std::string op_name = "OneHot";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":-1}}";

  std::vector<int64_t> input0{2, 2, 8, 8, 16, 8};
  std::vector<int64_t> input1{1};
  std::vector<int32_t> depth{32};
  std::vector<int32_t> off_value{1};
  std::vector<int64_t> output{2, 2, 8, 8, 16, 8, 32};

  auto opParas = op::OneHot("OneHot");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input0, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, depth, input1, ge::DT_INT32, ge::FORMAT_ND, depth);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, off_value, input1, ge::DT_INT32, ge::FORMAT_ND, off_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_INT32, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 1024 1 32 32768 32768 1 1048576 1024 0 0 ");
}

TEST_F(OneHotTiling, one_hot_tiling_9) {
  std::string op_name = "OneHot";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":0}}";

  std::vector<int64_t> input0{2, 2, 8, 8, 16, 8};
  std::vector<int64_t> input1{1};
  std::vector<int32_t> depth{32};
  std::vector<int32_t> off_value{1};
  std::vector<int64_t> output{32, 2, 2, 8, 8, 16, 8};

  auto opParas = op::OneHot("OneHot");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input0, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, depth, input1, ge::DT_INT32, ge::FORMAT_ND, depth);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, off_value, input1, ge::DT_INT32, ge::FORMAT_ND, off_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_INT32, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 0 8 32 32768 1 32768 1048576 0 1 1 ");
}

TEST_F(OneHotTiling, one_hot_tiling_10) {
  std::string op_name = "OneHot";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":1}}";

  std::vector<int64_t> input0{2, 2, 8, 8, 16, 8};
  std::vector<int64_t> input1{1};
  std::vector<int32_t> depth{32};
  std::vector<int32_t> off_value{1};
  std::vector<int64_t> output{2, 32, 2, 8, 8, 16, 8};

  auto opParas = op::OneHot("OneHot");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input0, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, depth, input1, ge::DT_INT32, ge::FORMAT_ND, depth);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, off_value, input1, ge::DT_INT32, ge::FORMAT_ND, off_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_INT32, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 1 11 2 32768 2 16384 1048576 1 0 0 ");
}

TEST_F(OneHotTiling, one_hot_tiling_11) {
  std::string op_name = "OneHot";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":2}}";

  std::vector<int64_t> input0{2, 2, 8, 8, 16, 8};
  std::vector<int64_t> input1{1};
  std::vector<int32_t> depth{32};
  std::vector<int32_t> off_value{1};
  std::vector<int64_t> output{2, 2, 32, 8, 8, 16, 8};

  auto opParas = op::OneHot("OneHot");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input0, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, depth, input1, ge::DT_INT32, ge::FORMAT_ND, depth);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, off_value, input1, ge::DT_INT32, ge::FORMAT_ND, off_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_INT32, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 1 11 4 32768 4 8192 1048576 1 0 0 ");
}

TEST_F(OneHotTiling, one_hot_tiling_12) {
  std::string op_name = "OneHot";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":3}}";

  std::vector<int64_t> input0{2, 2, 8, 8, 16, 8};
  std::vector<int64_t> input1{1};
  std::vector<int32_t> depth{32};
  std::vector<int32_t> off_value{1};
  std::vector<int64_t> output{2, 2, 8, 32, 8, 16, 8};

  auto opParas = op::OneHot("OneHot");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input0, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, depth, input1, ge::DT_INT32, ge::FORMAT_ND, depth);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, off_value, input1, ge::DT_INT32, ge::FORMAT_ND, off_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_INT32, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 1 10 32 32768 32 1024 1048576 1 0 0 ");
}

TEST_F(OneHotTiling, one_hot_tiling_13) {
  std::string op_name = "OneHot";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\": 4}}";

  std::vector<int64_t> input0{2, 2, 8, 8, 16, 8};
  std::vector<int64_t> input1{1};
  std::vector<int32_t> depth{32};
  std::vector<int32_t> off_value{1};
  std::vector<int64_t> output{2, 2, 8, 8, 32, 16, 8};

  auto opParas = op::OneHot("OneHot");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input0, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, depth, input1, ge::DT_INT32, ge::FORMAT_ND, depth);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, off_value, input1, ge::DT_INT32, ge::FORMAT_ND, off_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_INT32, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 8 10 32 32768 256 128 1048576 8 0 0 ");
}

TEST_F(OneHotTiling, one_hot_tiling_14) {
  std::string op_name = "OneHot";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":5}}";

  std::vector<int64_t> input0{2, 2, 8, 8, 16, 8};
  std::vector<int64_t> input1{1};
  std::vector<int32_t> depth{32};
  std::vector<int32_t> off_value{1};
  std::vector<int64_t> output{2, 2, 8, 8, 16, 32, 8};

  auto opParas = op::OneHot("OneHot");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input0, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, depth, input1, ge::DT_INT32, ge::FORMAT_ND, depth);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, off_value, input1, ge::DT_INT32, ge::FORMAT_ND, off_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_INT32, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 128 10 32 32768 4096 8 1048576 128 0 0 ");
}

TEST_F(OneHotTiling, one_hot_tiling_15) {
  std::string op_name = "OneHot";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":-1}}";

  std::vector<int64_t> input0{2, 2, 8, 8, 16, 8, 2};
  std::vector<int64_t> input1{1};
  std::vector<int32_t> depth{16};
  std::vector<int32_t> off_value{1};
  std::vector<int64_t> output{2, 2, 8, 8, 16, 8, 2, 16};

  auto opParas = op::OneHot("OneHot");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input0, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, depth, input1, ge::DT_INT32, ge::FORMAT_ND, depth);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, off_value, input1, ge::DT_INT32, ge::FORMAT_ND, off_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_INT32, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 2048 1 32 65536 65536 1 1048576 2048 0 0 ");
}

TEST_F(OneHotTiling, one_hot_tiling_16) {
  std::string op_name = "OneHot";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":1}}";

  std::vector<int64_t> input0{2, 21120};
  std::vector<int64_t> input1{1};
  std::vector<int32_t> depth{323};
  std::vector<int32_t> off_value{1};
  std::vector<int64_t> output{2, 323, 21120};

  auto opParas = op::OneHot("OneHot");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input0, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, depth, input1, ge::DT_INT32, ge::FORMAT_ND, depth);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, off_value, input1, ge::DT_INT32, ge::FORMAT_ND, off_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_INT32, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 1 12 2 42240 2 21120 13643520 1 0 0 ");
}

TEST_F(OneHotTiling, one_hot_tiling_17) {
  std::string op_name = "OneHot";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":1}}";

  std::vector<int64_t> input0{1, 4224};
  std::vector<int64_t> input1{1};
  std::vector<int32_t> depth{323};
  std::vector<int32_t> off_value{1};
  std::vector<int64_t> output{1, 323, 4224};

  auto opParas = op::OneHot("OneHot");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input0, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, depth, input1, ge::DT_INT32, ge::FORMAT_ND, depth);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, off_value, input1, ge::DT_INT32, ge::FORMAT_ND, off_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_INT32, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 1 11 1 4224 1 4224 1364352 1 0 0 ");
}
TEST_F(OneHotTiling, one_hot_tiling_18) {
  std::string op_name = "OneHot";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":3}}";

  std::vector<int64_t> input0{2, 16, 8, 8, 16};
  std::vector<int64_t> input1{1};
  std::vector<int32_t> depth{32};
  std::vector<int32_t> off_value{1};
  std::vector<int64_t> output{2, 32, 16, 8, 8, 16};

  auto opParas = op::OneHot("OneHot");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input0, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, depth, input1, ge::DT_INT32, ge::FORMAT_ND, depth);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, off_value, input1, ge::DT_INT32, ge::FORMAT_ND, off_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_INT32, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 8 10 32 32768 256 128 1048576 8 0 0 ");
}