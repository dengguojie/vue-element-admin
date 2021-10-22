#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "test_common.h"
#include "nn_training_ops.h"
#include "array_ops.h"
#include "common/utils/ut_op_util.h"

using namespace std;
using namespace ge;

class DynamicAtomicAddrCleanTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DynamicAtomicAddrClean SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DynamicAtomicAddrClean TearDown" << std::endl;
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

TEST_F(DynamicAtomicAddrCleanTiling, DynamicAtomicAddrClean_tiling_1) {
  std::string op_name = "DynamicAtomicAddrClean";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compile_info = R"({"_workspace_size_list": [32], "vars": {"ub_size": 126976, "core_num": 2, "workspace_num": 1}})";
  std::vector<uint32_t> workspace_size{1, 2, 3, 4, 5, 6, 7, 8};
  auto opParas = op::DynamicAtomicAddrClean("DynamicAtomicAddrClean");
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compile_info, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 1 16320 2040 8 0 0 1 1 8 0 0 1 1 ");
}

TEST_F(DynamicAtomicAddrCleanTiling, DynamicAtomicAddrClean_tiling_2) {
  std::string op_name = "DynamicAtomicAddrClean";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  auto opParas = op::DynamicAtomicAddrClean("DynamicAtomicAddrClean");

  std::string compile_info = R"({"_workspace_size_list": [32], "vars": {"ub_size": 126976, "core_num": 2}})";
  std::vector<int32_t> workspace_size{1, 2, 3, 4, 5, 6, 7, 8};

  optiling::utils::OpRunInfo runInfo;

  RUN_TILING_V3(opParas, iter->second, compile_info, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 1 16320 2040 8 0 0 1 1 8 0 0 1 1 ");
}

TEST_F(DynamicAtomicAddrCleanTiling, DynamicAtomicAddrClean_tiling_3) {
  std::string op_name = "DynamicAtomicAddrClean";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compile_info = R"({"_workspace_size_list": [32], "vars": {"ub_size": 126976, "core_num": 2}})";
  std::vector<uint32_t> workspace_size{1, 2, 3, 4, 5, 6, 7, 8};
  auto opParas = op::DynamicAtomicAddrClean("DynamicAtomicAddrClean");

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compile_info, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 1 16320 2040 8 0 0 1 1 8 0 0 1 1 ");
}

TEST_F(DynamicAtomicAddrCleanTiling, DynamicAtomicAddrClean_tiling_4) {
  std::string op_name = "DynamicAtomicAddrClean";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compile_info = R"({"_workspace_size_list": [0], "vars": {"ub_size": 126976, "core_num": 2}})";
  std::vector<uint32_t> workspace_size{};
  auto opParas = op::DynamicAtomicAddrClean("DynamicAtomicAddrClean");

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compile_info, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 1 16320 2040 0 0 0 0 0 0 0 0 0 0 ");
}
