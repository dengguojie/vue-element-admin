#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "selection_ops.h"
#include "array_ops.h"

using namespace std;

class GatherNdTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GatherNdTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GatherNdTiling TearDown" << std::endl;
  }
};

/*
 * be careful of the to_string fuction
 * the type of tiling_data in other ops is int64 while int32 here
 */
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
.INPUT(x, TensorType::BasicType())
    .INPUT(indices, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
*/

TEST_F(GatherNdTiling, gather_nd_tiling_0) {
  std::string op_name = "GatherNd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("GatherNd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":4, "
      "\"params_dsize\":2}}";

  std::vector<int64_t> inputA{87552};
  std::vector<int64_t> inputB{174, 1};
  std::vector<int64_t> output{174};

  auto opParas = op::GatherNd("GatherNd");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputA, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT16, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "1 2 0 87 0 0 32000 87 64000 32000 0 87 0 1 1 87552 0 0 19 1 0 0 0 0 0 0 0 ");
}

TEST_F(GatherNdTiling, gather_nd_tiling_1) {
  std::string op_name = "GatherNd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("GatherNd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":4, "
      "\"params_dsize\":2}}";

  std::vector<int64_t> inputA{5, 10, 13, 31};
  std::vector<int64_t> inputB{2, 4};
  std::vector<int64_t> output{2};

  auto opParas = op::GatherNd("GatherNd");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputA, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT16, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "2 1 0 2 0 0 4800 2 38400 4800 0 2 0 1 4 20150 0 0 19 4030 403 31 1 0 0 0 0 ");
}

TEST_F(GatherNdTiling, gather_nd_tiling_2) {
  std::string op_name = "GatherNd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("GatherNd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":4, "
      "\"params_dsize\":2}}";

  std::vector<int64_t> inputA{7, 6, 81, 6, 32};
  std::vector<int64_t> inputB{2, 6, 3};
  std::vector<int64_t> output{2, 6, 6, 32};

  auto opParas = op::GatherNd("GatherNd");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputA, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT16, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "3 12 0 1 0 0 10666 1 333 10 32 1 0 192 3 653184 0 0 19 93312 15552 192 0 0 0 0 0 ");
}

TEST_F(GatherNdTiling, gather_nd_tiling_3) {
  std::string op_name = "GatherNd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("GatherNd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":4, "
      "\"params_dsize\":2}}";

  std::vector<int64_t> inputA{81, 6, 32};
  std::vector<int64_t> inputB{1};
  std::vector<int64_t> output{6, 32};

  auto opParas = op::GatherNd("GatherNd");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputA, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT16, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "5 1 0 1 0 0 19200 1 200 0 96 1 0 192 1 15552 0 0 19 192 0 0 0 0 0 0 0 ");
}

TEST_F(GatherNdTiling, gather_nd_tiling_4) {
  std::string op_name = "GatherNd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("GatherNd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":4, "
      "\"params_dsize\":2}}";

  std::vector<int64_t> inputA{81, 600, 310};
  std::vector<int64_t> inputB{1};
  std::vector<int64_t> output{600, 310};

  auto opParas = op::GatherNd("GatherNd");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputA, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT16, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "9 32 0 1 0 0 0 0 0 0 0 55952 1 186000 1 15066000 0 0 19 186000 0 0 0 0 0 0 0 ");
}

TEST_F(GatherNdTiling, gather_nd_tiling_5) {
  std::string op_name = "GatherNd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("GatherNd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":4, "
      "\"params_dsize\":2}}";

  std::vector<int64_t> inputA{800, 600, 320};
  std::vector<int64_t> inputB{4, 2};
  std::vector<int64_t> output{4, 320};

  auto opParas = op::GatherNd("GatherNd");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputA, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT16, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "4 4 0 1 0 0 16000 1 200 0 80 1 0 320 2 153600000 0 0 19 192000 320 0 0 0 0 0 0 ");
}

TEST_F(GatherNdTiling, gather_nd_tiling_6) {
  std::string op_name = "GatherNd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("GatherNd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":4, "
      "\"params_dsize\":2}}";

  std::vector<int64_t> inputA{800, 611, 1111};
  std::vector<int64_t> inputB{2, 2};
  std::vector<int64_t> output{2, 1111};

  auto opParas = op::GatherNd("GatherNd");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputA, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT16, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "6 2 0 1 0 0 16000 1 57 40 280 1 0 1111 2 543056800 0 0 19 678821 1111 0 0 0 0 0 0 ");
}

TEST_F(GatherNdTiling, gather_nd_tiling_7) {
  std::string op_name = "GatherNd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("GatherNd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":4, "
      "\"params_dsize\":2}}";

  std::vector<int64_t> inputA{800, 611, 1111};
  std::vector<int64_t> inputB{2, 0};
  std::vector<int64_t> output{2, 0};

  auto opParas = op::GatherNd("GatherNd");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputA, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT16, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "8 2 0 1 0 0 0 0 0 0 4175 106400 0 543056800 0 543056800 0 0 19 0 0 0 0 0 0 0 0 ");
}