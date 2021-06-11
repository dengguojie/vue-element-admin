#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

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
  using namespace optiling;
  std::string op_name = "DynamicAtomicAddrClean";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 126976, \"core_num\": 2, \"workspace_num\": 1}}";
  std::vector<uint32_t> workspace_size{1,2,3,4,5,6,7,8};

  TeOpParas opParas;
  opParas.const_inputs["workspace_size"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
      (const uint8_t*)workspace_size.data(), workspace_size.size() * 4, ge::Tensor());
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 1 16320 2040 8 0 0 1 1 8 0 0 1 1 ");
}
TEST_F(DynamicAtomicAddrCleanTiling, DynamicAtomicAddrClean_tiling_2) {
  using namespace optiling;
  std::string op_name = "DynamicAtomicAddrClean";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 126976, \"core_num\": 2}}";
  std::vector<uint32_t> workspace_size{1,2,3,4,5,6,7,8};

  TeOpParas opParas;
  opParas.const_inputs["workspace_size"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
      (const uint8_t*)workspace_size.data(), workspace_size.size() * 4, ge::Tensor());
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 1 16320 2040 8 0 0 1 1 8 0 0 1 1 ");
} 
