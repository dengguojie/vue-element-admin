#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "math_ops.h"
#include "array_ops.h"

using namespace std;

class SparseSegmentSumGradTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SparseSegmentSumGradTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SparseSegmentSumGradTiling TearDown" << std::endl;
  }
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
  std::cout << "to_string" << std::endl;
  std::cout << result << std::endl;
  return result;
}

using namespace ge;
#include "common/utils/ut_op_util.h"
using namespace ut_util;

TEST_F(SparseSegmentSumGradTiling, SparseSegmentSumGrad_tiling_0) {
  using namespace optiling;
  std::string op_name = "SparseSegmentSumGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SparseSegmentSumGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{2,19994};
  std::vector<int64_t> inputB{5};
  std::vector<int64_t> inputC{5};
  std::vector<int64_t> inputD{1};
  std::vector<int32_t> output_dim0_value{1,};
  std::vector<int64_t> gradient{1,19994};

  auto opParas = op::SparseSegmentSumGrad("SparseSegmentSumGrad");
  TENSOR_INPUT_WITH_SHAPE(opParas, grad, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, segment_ids, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputC, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, output_dim0, inputD, ge::DT_INT32, FORMAT_ND, output_dim0_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output, gradient, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 5 1 0 19994 ");
}
TEST_F(SparseSegmentSumGradTiling, SparseSegmentSumGrad_tiling_1) {
  using namespace optiling;
  std::string op_name = "SparseSegmentSumGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SparseSegmentSumGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{1024,80};
  std::vector<int64_t> inputB{2048};
  std::vector<int64_t> inputC{2048};
  std::vector<int64_t> inputD{1};
  std::vector<int32_t> output_dim0_value{300,};
  std::vector<int64_t> gradient{300,80};

  auto opParas = op::SparseSegmentSumGrad("SparseSegmentSumGrad");
  TENSOR_INPUT_WITH_SHAPE(opParas, grad, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, segment_ids, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputC, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, output_dim0, inputD, ge::DT_INT32, FORMAT_ND, output_dim0_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output, gradient, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 32 64 0 80 ");
}
TEST_F(SparseSegmentSumGradTiling, SparseSegmentSumGrad_tiling_2) {
  using namespace optiling;
  std::string op_name = "SparseSegmentSumGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SparseSegmentSumGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{46,44};
  std::vector<int64_t> inputB{50};
  std::vector<int64_t> inputC{50};
  std::vector<int64_t> inputD{1};
  std::vector<int32_t> output_dim0_value{100,};
  std::vector<int64_t> gradient{100,44};

  auto opParas = op::SparseSegmentSumGrad("SparseSegmentSumGrad");
  TENSOR_INPUT_WITH_SHAPE(opParas, grad, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, segment_ids, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputC, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, output_dim0, inputD, ge::DT_INT32, FORMAT_ND, output_dim0_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output, gradient, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 32 1 18 44 ");
}
TEST_F(SparseSegmentSumGradTiling, SparseSegmentSumGrad_tiling_3) {
  using namespace optiling;
  std::string op_name = "SparseSegmentSumGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SparseSegmentSumGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{20,44};
  std::vector<int64_t> inputB{0};
  std::vector<int64_t> inputC{0};
  std::vector<int64_t> inputD{1};
  std::vector<int32_t> output_dim0_value{100,};
  std::vector<int64_t> gradient{100,44};

  auto opParas = op::SparseSegmentSumGrad("SparseSegmentSumGrad");
  TENSOR_INPUT_WITH_SHAPE(opParas, grad, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, segment_ids, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputC, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, output_dim0, inputD, ge::DT_INT32, FORMAT_ND, output_dim0_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output, gradient, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 1 0 0 0 ");
}
TEST_F(SparseSegmentSumGradTiling, SparseSegmentSumGrad_tiling_4) {
  using namespace optiling;
  std::string op_name = "SparseSegmentSumGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SparseSegmentSumGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{2,40000};
  std::vector<int64_t> inputB{5};
  std::vector<int64_t> inputC{5};
  std::vector<int64_t> inputD{1};
  std::vector<int32_t> output_dim0_value{3,};
  std::vector<int64_t> gradient{3,40000};

  auto opParas = op::SparseSegmentSumGrad("SparseSegmentSumGrad");
  TENSOR_INPUT_WITH_SHAPE(opParas, grad, inputA, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, segment_ids, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputC, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, output_dim0, inputD, ge::DT_INT32, FORMAT_ND, output_dim0_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output, gradient, ge::DT_FLOAT16, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 5 1 0 40000 ");
}
TEST_F(SparseSegmentSumGradTiling, SparseSegmentSumGrad_tiling_5) {
  using namespace optiling;
  std::string op_name = "SparseSegmentSumGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SparseSegmentSumGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{1024,1};
  std::vector<int64_t> inputB{512};
  std::vector<int64_t> inputC{512};
  std::vector<int64_t> inputD{1};
  std::vector<int32_t> output_dim0_value{300,};
  std::vector<int64_t> gradient{300,1};

  auto opParas = op::SparseSegmentSumGrad("SparseSegmentSumGrad");
  TENSOR_INPUT_WITH_SHAPE(opParas, grad, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, segment_ids, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputC, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, output_dim0, inputD, ge::DT_INT32, FORMAT_ND, output_dim0_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output, gradient, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 32 16 0 1 ");
}
TEST_F(SparseSegmentSumGradTiling, SparseSegmentSumGrad_tiling_6) {
  using namespace optiling;
  std::string op_name = "SparseSegmentSumGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SparseSegmentSumGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{1024,19994};
  std::vector<int64_t> inputB{2048};
  std::vector<int64_t> inputC{2048};
  std::vector<int64_t> inputD{1};
  std::vector<int32_t> output_dim0_value{300,};
  std::vector<int64_t> gradient{300,19994};

  auto opParas = op::SparseSegmentSumGrad("SparseSegmentSumGrad");
  TENSOR_INPUT_WITH_SHAPE(opParas, grad, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, segment_ids, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputC, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, output_dim0, inputD, ge::DT_INT32, FORMAT_ND, output_dim0_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output, gradient, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 32 64 0 19994 ");
}