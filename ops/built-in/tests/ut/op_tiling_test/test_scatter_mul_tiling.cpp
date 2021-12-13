#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "matrix_calculation_ops.h"
#include "array_ops.h"

using namespace std;

class ScatterMulTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ScatterMulTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ScatterMulTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream &tiling_data) {
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

TEST_F(ScatterMulTiling, scatter_mul_tiling_0) {
  using namespace optiling;
  std::string op_name = "ScatterMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterMul");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{33,1,32,2,11};
  std::vector<int64_t> inputB{11,12,2,8};
  std::vector<int64_t> inputC{11,12,2,8,1,32,2,11};
  std::vector<int64_t> output{33,1,32,2,11};

  auto opParas = op::ScatterMax("ScatterMax");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 2 17 704 0 2112 1486848 0 704 23232 0 23232 176 88 33 1408 ");
}

TEST_F(ScatterMulTiling, scatter_mul_tiling_1) {
  using namespace optiling;
  std::string op_name = "ScatterMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterMul");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{1039,34,23,31};
  std::vector<int64_t> inputB{1039};
  std::vector<int64_t> inputC{1039,34,23,31};
  std::vector<int64_t> output{1039,34,23,31};

  auto opParas = op::ScatterMax("ScatterMax");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "10 33 32 24242 0 1039 25187438 1 434 25187438 1 434 0 0 1039 0 ");
}


TEST_F(ScatterMulTiling, scatter_mul_tiling_4) {
  using namespace optiling;
  std::string op_name = "ScatterMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterMul");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\":253952, \"core_num\":32, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{33, 1, 32, 2, 11};
  std::vector<int64_t> inputB{11, 12, 2, 8};
  std::vector<int64_t> inputC{11, 12, 2, 8, 1, 32, 2, 11};
  std::vector<int64_t> output{33, 1, 32, 22};

  auto opParas = op::ScatterMax("ScatterMax");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterMulTiling, scatter_mul_tiling_5) {
  using namespace optiling;
  std::string op_name = "ScatterMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterMul");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\":253952, \"core_num\":32, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{33, 1, 32, 2, 11};
  std::vector<int64_t> inputB{1};
  std::vector<int64_t> inputC{1, 32, 2, 11};
  std::vector<int64_t> output{33, 1, 32, 2, 11};

  auto opParas = op::ScatterMax("ScatterMax");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterMulTiling, scatter_mul_tiling_6) {
  using namespace optiling;
  std::string op_name = "ScatterMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterMul");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\":253952, \"core_num\":32, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{33, 1, 32, 2, 11};
  std::vector<int64_t> inputB{11, 12, 2, 8};
  std::vector<int64_t> inputC{11, 12, 2, 8, 32, 2, 11};
  std::vector<int64_t> output{33, 1, 32, 2, 11};

  auto opParas = op::ScatterMax("ScatterMax");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterMulTiling, scatter_mul_tiling_7) {
  using namespace optiling;
  std::string op_name = "ScatterMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterMul");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{33,1,32,2,11};
  std::vector<int64_t> inputB{11,12,2,8};
  std::vector<int64_t> inputC{11,12,2,8,1,32,2,11};
  std::vector<int64_t> output{33,1,32,2,11};

  auto opParas = op::ScatterMax("ScatterMax");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterMulTiling, scatter_mul_tiling_8) {
  using namespace optiling;
  std::string op_name = "ScatterMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterMul");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{33,1,32,2,11};
  std::vector<int64_t> inputB{11,12,2,8};
  std::vector<int64_t> inputC{11,12,2,8,1,32,2,11};
  std::vector<int64_t> output{33,1,32,2,11};

  auto opParas = op::ScatterMax("ScatterMax");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterMulTiling, scatter_mul_tiling_9) {
  using namespace optiling;
  std::string op_name = "ScatterMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterMul");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"indices_size\":4}}";

  std::vector<int64_t> inputA{33,1,32,2,11};
  std::vector<int64_t> inputB{11,12,2,8};
  std::vector<int64_t> inputC{11,12,2,8,1,32,2,11};
  std::vector<int64_t> output{33,1,32,2,11};

  auto opParas = op::ScatterMax("ScatterMax");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterMulTiling, scatter_mul_tiling_10) {
  using namespace optiling;
  std::string op_name = "ScatterMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterMul");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\":4}}";

  std::vector<int64_t> inputA{33,1,32,2,11};
  std::vector<int64_t> inputB{11,12,2,8};
  std::vector<int64_t> inputC{11,12,2,8,1,32,2,11};
  std::vector<int64_t> output{33,1,32,2,11};

  auto opParas = op::ScatterMax("ScatterMax");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterMulTiling, scatter_mul_tiling_13) {
  using namespace optiling;
  std::string op_name = "ScatterMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterMul");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\": 4, \"indices_size\": 4}}";

  std::vector<int64_t> inputA{};
  std::vector<int64_t> inputB{11,12,2,8};
  std::vector<int64_t> inputC{11,12,2,8,1,32,2,11};
  std::vector<int64_t> output{33,1,32,2,11};

  auto opParas = op::ScatterMax("ScatterMax");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterMulTiling, scatter_mul_tiling_14) {
  using namespace optiling;
  std::string op_name = "ScatterMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterMul");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\": 4, \"indices_size\": 4}}";

  std::vector<int64_t> inputA{33,1,32,2,11};
  std::vector<int64_t> inputB{11,12,2,8};
  std::vector<int64_t> inputC{11,12,2,8,1,32,2,11};
  std::vector<int64_t> output{};

  auto opParas = op::ScatterMax("ScatterMax");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterMulTiling, scatter_mul_tiling_15) {
  using namespace optiling;
  std::string op_name = "ScatterMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterMul");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 0, \"core_num\": 0, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{33,1,32,2,11};
  std::vector<int64_t> inputB{11,12,2,8};
  std::vector<int64_t> inputC{11,12,2,8,1,32,2,11};
  std::vector<int64_t> output{33,1,32,2,11};

  auto opParas = op::ScatterMax("ScatterMax");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}
