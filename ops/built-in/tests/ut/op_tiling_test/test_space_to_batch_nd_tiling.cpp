#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#define private public
#include <register/op_tiling.h>
#include "test_common.h"
#include "array_ops.h"
#include "selection_ops.h"
#include "common/utils/ut_op_util.h"
#include "transformation_ops.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class SpaceToBatchNDTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SpaceToBatchNDTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SpaceToBatchNDTiling TearDown" << std::endl;
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

TEST_F(SpaceToBatchNDTiling, SpaceToBatchND_tiling_0) {
  std::string op_name = "SpaceToBatchND";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"block_size\": 2}}";

  auto opParas = op::SpaceToBatchND("SpaceToBatchND");
  vector<vector<int64_t>> input_shapes = {
      {4, 2, 2, 2, 16},
      {2, 2},
  };

  vector<vector<int64_t>> output_shapes = {
      {4, 2, 2, 2, 16},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> pads{1, 1, 1, 1};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_NC1HWC0, {});
  SetGetOriginFormat(opParas, "x", ge::FORMAT_NHWC);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, paddings, input_shapes[1], dtypes[1], FORMAT_ND, pads);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shapes[0], ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 8 1 1 4 0 2 2 0 0 1 1 1 1 0 2 2 2 16 16 0 2 2 ");
}

TEST_F(SpaceToBatchNDTiling, SpaceToBatchND_tiling_1) {
  std::string op_name = "SpaceToBatchND";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"block_size\": 0}}";

  auto opParas = op::SpaceToBatchND("SpaceToBatchND");
  vector<vector<int64_t>> input_shapes = {{2, 2, 2, 2, 2, 16}, {3}, {2, 2}};
  vector<vector<int64_t>> output_shapes = {
      {16, 2, 2, 2, 2, 16},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> block{2, 2, 2};
  std::vector<int32_t> pads{1, 1, 1, 1, 1, 1};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_NDC1HWC0, {});
  SetGetOriginFormat(opParas, "x", ge::FORMAT_NDHWC);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, block_shape, input_shapes[1], dtypes[1], FORMAT_ND, block);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, paddings, input_shapes[2], dtypes[2], FORMAT_ND, pads);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shapes[0], ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);

  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "6 2 1 1 2 2 2 2 1 1 1 1 1 1 2 2 2 2 16 16 2 2 2 ");
}

TEST_F(SpaceToBatchNDTiling, SpaceToBatchND_tiling_2) {
  std::string op_name = "SpaceToBatchND";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"block_size\": 0}}";

  auto opParas = op::SpaceToBatchND("SpaceToBatchND");
  vector<vector<int64_t>> input_shapes = {{4, 2, 2, 2, 16}, {3}, {3, 2}};

  vector<vector<int64_t>> output_shapes = {
      {16, 2, 2, 2, 16},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> block{1, 2, 2};
  std::vector<int32_t> pads{0, 0, 1, 1, 1, 1};
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_NC1HWC0, {});
  SetGetOriginFormat(opParas, "x", ge::FORMAT_NCHW);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, block_shape, input_shapes[1], dtypes[1], FORMAT_ND, block);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, paddings, input_shapes[2], dtypes[2], FORMAT_ND, pads);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shapes[0], ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);

  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 8 1 1 4 0 2 2 0 0 1 1 1 1 0 2 2 2 16 16 0 2 2 ");
}

TEST_F(SpaceToBatchNDTiling, SpaceToBatchND_tiling_3) {
  std::string op_name = "SpaceToBatchND";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"block_size\": 0}}";

  auto opParas = op::SpaceToBatchND("SpaceToBatchND");
  vector<vector<int64_t>> input_shapes = {{2, 2, 2, 2, 2, 16}, {4}, {4, 2}};

  vector<vector<int64_t>> output_shapes = {
      {16, 2, 2, 2, 16},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> block{1, 2, 2, 2};
  std::vector<int32_t> pads{0, 0, 1, 1, 1, 1, 1, 1};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_NDC1HWC0, {});
  SetGetOriginFormat(opParas, "x", ge::FORMAT_NCDHW);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, block_shape, input_shapes[1], dtypes[1], FORMAT_ND, block);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, paddings, input_shapes[2], dtypes[2], FORMAT_ND, pads);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shapes[0], ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);

  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "6 2 1 1 2 2 2 2 1 1 1 1 1 1 2 2 2 2 16 16 2 2 2 ");
}

TEST_F(SpaceToBatchNDTiling, SpaceToBatchND_tiling_4) {
  std::string op_name = "SpaceToBatchND";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"block_size\": 0}}";

  auto opParas = op::SpaceToBatchND("SpaceToBatchND");
  vector<vector<int64_t>> input_shapes = {{2, 2, 7952, 1, 16}, {3}, {3, 2}};

  vector<vector<int64_t>> output_shapes = {
      {2, 2, 7952, 1, 16},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> block{1, 1, 1};
  std::vector<int32_t> pads{0, 0, 0, 0, 0, 0};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_NC1HWC0, {});
  SetGetOriginFormat(opParas, "x", ge::FORMAT_NCHW);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, block_shape, input_shapes[1], dtypes[1], FORMAT_ND, block);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, paddings, input_shapes[2], dtypes[2], FORMAT_ND, pads);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shapes[0], ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);

  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "12 32 249 233 2 0 1 1 0 0 0 0 0 0 0 2 7952 1 16 2 0 7952 1 ");
}

TEST_F(SpaceToBatchNDTiling, SpaceToBatchND_tiling_5) {
  std::string op_name = "SpaceToBatchND";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"block_size\": 0}}";

  auto opParas = op::SpaceToBatchND("SpaceToBatchND");
  vector<vector<int64_t>> input_shapes = {{2, 2, 1, 4487, 16}, {3}, {3, 2}};
  vector<vector<int64_t>> output_shapes = {
      {4, 2, 1, 2244, 16},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> block{1, 1, 2};
  std::vector<int32_t> pads{0, 0, 0, 0, 0, 1};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_NC1HWC0, {});
  SetGetOriginFormat(opParas, "x", ge::FORMAT_NCHW);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, block_shape, input_shapes[1], dtypes[1], FORMAT_ND, block);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, paddings, input_shapes[2], dtypes[2], FORMAT_ND, pads);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shapes[0], ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);

  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 4 1 1 2 0 2 1 0 0 0 1 0 0 0 2 4487 1 16 4 0 2244 1 ");
}

TEST_F(SpaceToBatchNDTiling, SpaceToBatchND_tiling_6) {
  std::string op_name = "SpaceToBatchND";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"block_size\": 0}}";

  auto opParas = op::SpaceToBatchND("SpaceToBatchND");
  vector<vector<int64_t>> input_shapes = {{2, 2, 4487, 1, 16}, {1}, {1, 2}};

  vector<vector<int64_t>> output_shapes = {
      {4, 2, 1, 2244, 16},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> block{2};
  std::vector<int32_t> pads{0, 1};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_NC1HWC0, {});
  SetGetOriginFormat(opParas, "x", ge::FORMAT_NHWC);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, block_shape, input_shapes[1], dtypes[1], FORMAT_ND, block);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, paddings, input_shapes[2], dtypes[2], FORMAT_ND, pads);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shapes[0], ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);

  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 4 1 1 2 0 2 1 0 0 0 1 0 0 0 2 4487 1 16 4 0 2244 1 ");
}

TEST_F(SpaceToBatchNDTiling, SpaceToBatchND_tiling_7) {
  std::string op_name = "SpaceToBatchND";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"block_size\": 0}}";

  auto opParas = op::SpaceToBatchND("SpaceToBatchND");
  vector<vector<int64_t>> input_shapes = {{2, 2, 4487, 1, 16}, {1}, {1, 2}};
  vector<vector<int64_t>> output_shapes = {
      {4, 2, 1, 2244, 16},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> block{2};
  std::vector<int32_t> pads{0, 1};

  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, block_shape, input_shapes[1], dtypes[1], FORMAT_ND, block);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, paddings, input_shapes[2], dtypes[2], FORMAT_ND, pads);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shapes[0], ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(SpaceToBatchNDTiling, SpaceToBatchND_tiling_8) {
  std::string op_name = "SpaceToBatchND";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"block_size\": 0}}";

  auto opParas = op::SpaceToBatchND("SpaceToBatchND");
  vector<vector<int64_t>> input_shapes = {{2, 2, 4487, 1, 16}, {1}, {1, 2}};

  vector<vector<int64_t>> output_shapes = {
      {4, 2, 1, 2244, 16},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> block{2};
  std::vector<int32_t> pads{0, 1};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_NC1HWC0, {});
  SetGetOriginFormat(opParas, "x", ge::FORMAT_NHWC);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, paddings, input_shapes[2], dtypes[2], FORMAT_ND, pads);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shapes[0], ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(SpaceToBatchNDTiling, SpaceToBatchND_tiling_9) {
  std::string op_name = "SpaceToBatchND";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"block_size\": 0}}";

  auto opParas = op::SpaceToBatchND("SpaceToBatchND");
  vector<vector<int64_t>> input_shapes = {{2, 2, 4487, 1}, {1}, {1, 2}};

  vector<vector<int64_t>> output_shapes = {
      {4, 2, 1, 2244, 16},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> block{2};
  std::vector<int32_t> pads{0, 1};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_NC1HWC0, {});
  SetGetOriginFormat(opParas, "x", ge::FORMAT_NHWC);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, block_shape, input_shapes[1], dtypes[1], FORMAT_ND, block);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, paddings, input_shapes[2], dtypes[2], FORMAT_ND, pads);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shapes[0], ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(SpaceToBatchNDTiling, SpaceToBatchND_tiling_10) {
  std::string op_name = "SpaceToBatchND";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"block_size\": 0}}";

  auto opParas = op::SpaceToBatchND("SpaceToBatchND");
  vector<vector<int64_t>> input_shapes = {{2, 2, 2, 2, 2}, {4}, {4, 2}};

  vector<vector<int64_t>> output_shapes = {
      {16, 2, 2, 2, 16},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> block{1, 2, 2, 2};
  std::vector<int32_t> pads{0, 0, 1, 1, 1, 1, 1, 1};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_NDC1HWC0, {});
  SetGetOriginFormat(opParas, "x", ge::FORMAT_NCDHW);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, block_shape, input_shapes[1], dtypes[1], FORMAT_ND, block);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, paddings, input_shapes[2], dtypes[2], FORMAT_ND, pads);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shapes[0], ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}