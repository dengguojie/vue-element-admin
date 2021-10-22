#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "transformation_ops.h"
#include "array_ops.h"

using namespace std;
using namespace ge;

class AsStridedTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
  }

  static void TearDownTestCase() {
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

#include "test_common.h"

TEST_F(AsStridedTiling, tiling1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("AsStrided");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::AsStrided("AsStrided");
  TensorDesc tensorOutput;
  tensorOutput.SetDataType(ge::DT_FLOAT16);
  TENSOR_OUTPUT(opParas, tensorOutput, y);
  std::string compileInfo = "{\"vars\": {\"ub_size\": 196608, \"core_num\" : 96, \"dtype\": \"int32\"}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  EXPECT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
}

