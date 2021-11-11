#include <gtest/gtest.h>

#include <iostream>
#include <vector>
#define private public
#include <iostream>

#include "array_ops.h"
#include "common/utils/ut_op_util.h"
#include "op_tiling/op_tiling_util.h"
#include "register/op_tiling_registry.h"
#include "selection_ops.h"
#include "test_common.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class UnsortedSegmentProdTiling : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "UnsortedSegmentProdTiling SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "UnsortedSegmentProdTiling TearDown" << std::endl; }
};

static string to_string(const std::stringstream &tiling_data) {
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

TEST_F(UnsortedSegmentProdTiling, unsortedsegmentprod_tiling_0) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentProd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentProd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 261632, \"core_num\": 32, \"dtype\":\"float16\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{3, 16, 10419, 3};
  std::vector<int64_t> inputB{3};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments{
      8,
  };
  std::vector<int64_t> output{8, 16, 10419, 3};

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputA));
  tensor_inputA.SetDataType(ge::DT_FLOAT16);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputB));
  tensor_inputB.SetDataType(ge::DT_INT32);
  TensorDesc tensor_inputC;
  tensor_inputC.SetShape(ge::Shape(inputC));
  tensor_inputC.SetDataType(ge::DT_INT32);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::UnsortedSegmentProd(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, x);
  TENSOR_INPUT(opParas, tensor_inputB, segment_ids);
  TENSOR_INPUT_CONST(opParas, tensor_inputC, num_segments, (const uint8_t *)num_segments.data(),
                     num_segments.size() * 4);
  TENSOR_OUTPUT(opParas, tensor_output, y);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "50 32 3 3 1 3 3 96 5248 1552 1 8 8 1 1 1 5248 500112 41 13 0 328 97 1 ");
}
TEST_F(UnsortedSegmentProdTiling, unsortedsegmentprod_tiling_1) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentProd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentProd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 261632, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{4, 31, 4};
  std::vector<int64_t> inputB{4};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments{
      16,
  };
  std::vector<int64_t> output{16, 31, 4};

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputA));
  tensor_inputA.SetDataType(ge::DT_FLOAT);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputB));
  tensor_inputB.SetDataType(ge::DT_INT32);
  TensorDesc tensor_inputC;
  tensor_inputC.SetShape(ge::Shape(inputC));
  tensor_inputC.SetDataType(ge::DT_INT32);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(ge::DT_FLOAT);

  auto opParas = op::UnsortedSegmentProd(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, x);
  TENSOR_INPUT(opParas, tensor_inputB, segment_ids);
  TENSOR_INPUT_CONST(opParas, tensor_inputC, num_segments, (const uint8_t *)num_segments.data(),
                     num_segments.size() * 4);
  TENSOR_OUTPUT(opParas, tensor_output, y);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "40 2 1 1 1 4 4 2 64 60 8 2 2 8 1 1 64 124 1 1 4 8 8 1 ");
}
