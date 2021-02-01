#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class ConcatOffsetTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ConcatOffsetTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ConcatOffsetTiling TearDown" << std::endl;
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

TEST_F(ConcatOffsetTiling, concat_offset_tiling_0) {
  using namespace optiling;
  std::string op_name = "ConcatOffset";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("ConcatOffset");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32}}";

  TeOpParas opParas;
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}

