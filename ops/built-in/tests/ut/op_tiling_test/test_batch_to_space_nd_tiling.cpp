#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "transformation_ops.h"
#include "array_ops.h"
#include "op_tiling/op_tiling_util.h"
#include "common/utils/ut_op_util.h"
#include "test_common.h"
using namespace std;
using namespace ut_util;
using namespace ge;

class BatchToSpaceNDTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BatchToSpaceNDTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BatchToSpaceNDTiling TearDown" << std::endl;
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

/*
.INPUT(x, TensorType::BasicType())
    .INPUT(block_shape, TensorType::IndexNumberType())
    .INPUT(crops, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
*/

TEST_F(BatchToSpaceNDTiling, batchtospacend_tiling_0) {
  std::string op_name = "BatchToSpaceND";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"block_size\": 2}}";

  std::vector<int64_t> input{16, 2, 2, 32};
  std::vector<int64_t> input_crops{2, 2};
  std::vector<int32_t> crops_value{1, 1, 1, 1};
  std::vector<int64_t> output{4, 2, 2, 2, 16};

  auto opParas = op::BatchToSpace(op_name);
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT16, FORMAT_NHWC, {});
  TransformerOpBaseFormat(opParas, "x", FORMAT_NC1HWC0);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input, DT_FLOAT16, FORMAT_NC1HWC0, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, crops, input_crops, DT_INT32, FORMAT_ND, crops_value);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 8 1 1 16 0 2 2 0 0 1 1 1 1 0 2 2 2 16 4 0 2 2 ");
  for (int64_t i = 0; i < 10; i++) {
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  }
}

TEST_F(BatchToSpaceNDTiling, batchtospacend_tiling_1) {
  std::string op_name = "BatchToSpaceND";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"block_size\": 0}}";

  std::vector<int64_t> input{16, 2, 2, 2, 32};
  std::vector<int64_t> input_block{3};
  std::vector<int32_t> block{2, 2, 2};
  std::vector<int64_t> input_crops{3, 2};
  std::vector<int32_t> crops_value{1, 1, 1, 1, 1, 1};
  std::vector<int64_t> output{2, 2, 2, 2, 2, 16};

  auto opParas = op::BatchToSpaceND(op_name);
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT16, FORMAT_NDHWC, {});
  TransformerOpBaseFormat(opParas, "x", FORMAT_NDC1HWC0);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input, DT_FLOAT16, FORMAT_NDC1HWC0, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, block_shape, input_block, DT_INT32, FORMAT_ND, block);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, crops, input_crops, DT_INT32, FORMAT_ND, crops_value);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "6 2 1 1 16 2 2 2 1 1 1 1 1 1 2 2 2 2 16 2 2 2 2 ");
  for (int64_t i = 0; i < 10; i++) {
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  }
}

TEST_F(BatchToSpaceNDTiling, batchtospacend_tiling_2) {
  std::string op_name = "BatchToSpaceND";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"block_size\": 0}}";

  std::vector<int64_t> input{16, 32, 2, 2};
  std::vector<int64_t> input_block{3};
  std::vector<int32_t> block{1, 2, 2};
  std::vector<int64_t> input_crops{3, 2};
  std::vector<int32_t> crops_value{0, 0, 1, 1, 1, 1};
  std::vector<int64_t> output{4, 2, 2, 2, 16};

  auto opParas = op::BatchToSpaceND(op_name);
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT16, FORMAT_NCHW, {});
  TransformerOpBaseFormat(opParas, "x", FORMAT_NC1HWC0);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input, DT_FLOAT16, FORMAT_NC1HWC0, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, block_shape, input_block, DT_INT32, FORMAT_ND, block);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, crops, input_crops, DT_INT32, FORMAT_ND, crops_value);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 8 1 1 16 0 2 2 0 0 1 1 1 1 0 2 2 2 16 4 0 2 2 ");
  for (int64_t i = 0; i < 10; i++) {
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  }
}

TEST_F(BatchToSpaceNDTiling, batchtospacend_tiling_3) {
  std::string op_name = "BatchToSpaceND";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"block_size\": 0}}";

  std::vector<int64_t> input{16, 32, 2, 2, 2};
  std::vector<int64_t> input_block{4};
  std::vector<int32_t> block{1, 2, 2, 2};
  std::vector<int64_t> input_crops{4, 2};
  std::vector<int32_t> crops_value{0, 0, 1, 1, 1, 1, 1, 1};
  std::vector<int64_t> output{2, 2, 2, 2, 2, 16};

  auto opParas = op::BatchToSpaceND(op_name);
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT16, FORMAT_NCDHW, {});
  TransformerOpBaseFormat(opParas, "x", FORMAT_NDC1HWC0);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input, DT_FLOAT16, FORMAT_NDC1HWC0, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, block_shape, input_block, DT_INT32, FORMAT_ND, block);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, crops, input_crops, DT_INT32, FORMAT_ND, crops_value);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "6 2 1 1 16 2 2 2 1 1 1 1 1 1 2 2 2 2 16 2 2 2 2 ");
  for (int64_t i = 0; i < 10; i++) {
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  }
}

TEST_F(BatchToSpaceNDTiling, batchtospacend_tiling_4) {
  std::string op_name = "BatchToSpaceND";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"block_size\": 0}}";

  std::vector<int64_t> input{2, 32, 1, 4487};
  std::vector<int64_t> input_block{3};
  std::vector<int32_t> block{1, 1, 2};
  std::vector<int64_t> input_crops{3, 2};
  std::vector<int32_t> crops_value{0, 0, 0, 0, 0, 1};
  std::vector<int64_t> output{1, 2, 1, 8973, 16};

  auto opParas = op::BatchToSpaceND(op_name);
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT16, FORMAT_NCHW, {});
  TransformerOpBaseFormat(opParas, "x", FORMAT_NC1HWC0);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input, DT_FLOAT16, FORMAT_NC1HWC0, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, block_shape, input_block, DT_INT32, FORMAT_ND, block);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, crops, input_crops, DT_INT32, FORMAT_ND, crops_value);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "12 32 141 116 2 0 2 1 0 0 0 1 0 0 0 2 4487 1 16 1 0 8973 1 ");
  for (int64_t i = 0; i < 10; i++) {
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  }
}

TEST_F(BatchToSpaceNDTiling, batchtospacend_tiling_5) {
  std::string op_name = "BatchToSpaceND";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"block_size\": 0}}";

  std::vector<int64_t> input{2, 4487, 1, 32};
  std::vector<int64_t> input_block{1};
  std::vector<int32_t> block{2};
  std::vector<int64_t> input_crops{1, 2};
  std::vector<int32_t> crops_value{0, 1};
  std::vector<int64_t> output{1, 2, 8973, 1, 16};

  auto opParas = op::BatchToSpaceND(op_name);
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT16, FORMAT_NHWC, {});
  TransformerOpBaseFormat(opParas, "x", FORMAT_NC1HWC0);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input, DT_FLOAT16, FORMAT_NC1HWC0, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, block_shape, input_block, DT_INT32, FORMAT_ND, block);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, crops, input_crops, DT_INT32, FORMAT_ND, crops_value);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "12 32 141 116 2 0 2 1 0 0 0 1 0 0 0 2 4487 1 16 1 0 8973 1 ");
  for (int64_t i = 0; i < 10; i++) {
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  }
}
