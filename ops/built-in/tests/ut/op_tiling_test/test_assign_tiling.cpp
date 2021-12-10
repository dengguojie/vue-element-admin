#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "selection_ops.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"
#define private public
#include "register/op_tiling_registry.h"
#include "common/utils/ut_op_util.h"
#include "test_common.h"
using namespace ge;
using namespace ut_util;
using namespace std;

class AssignTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AssignTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AssignTiling TearDown" << std::endl;
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

TEST_F(AssignTiling, Assign_tiling1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Assign");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::Assign("Assign");

  vector<vector<int64_t>> input{
      {4, 4, 4, 4},
      {4, 4, 4, 4},
  };

  TENSOR_INPUT_WITH_SHAPE(opParas, ref, input[0], DT_FLOAT16, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, value, input[1], DT_FLOAT16, FORMAT_ND, {});

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\": 256000}}";
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4 4 4 ");
}
