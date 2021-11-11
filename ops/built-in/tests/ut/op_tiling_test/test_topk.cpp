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

class TopKDTiling : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "TopKDTiling SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "TopKDTiling TearDown" << std::endl; }
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

TEST_F(TopKDTiling, topkd_tiling_0) {
  std::string op_name = "TopKD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("TopKD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"batch_cols_padding\":4835, \"k_num\":139, \"max_k\":4096}}";

  std::vector<int64_t> input_tensor_shape{32, 20308};
  std::vector<int64_t> indices_tensor_shape{8192};
  std::vector<int64_t> out_tensor_shape{139};
  std::vector<int64_t> out_indices_tensor_shape{139};

  TensorDesc input_tensor;
  input_tensor.SetShape(ge::Shape(input_tensor_shape));
  input_tensor.SetDataType(ge::DT_FLOAT16);
  TensorDesc indices_tensor;
  indices_tensor.SetShape(ge::Shape(indices_tensor_shape));
  indices_tensor.SetDataType(ge::DT_INT32);
  TensorDesc out_tensor;
  out_tensor.SetShape(ge::Shape(out_tensor_shape));
  out_tensor.SetDataType(ge::DT_FLOAT16);
  TensorDesc out_indices_tensor;
  out_indices_tensor.SetShape(ge::Shape(out_indices_tensor_shape));
  out_indices_tensor.SetDataType(ge::DT_INT32);

  auto opParas = op::TopKD(op_name);
  TENSOR_INPUT(opParas, input_tensor, x);
  TENSOR_INPUT(opParas, indices_tensor, assist_seq);
  TENSOR_OUTPUT(opParas, out_tensor, values);
  TENSOR_OUTPUT(opParas, out_indices_tensor, indices);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "32 32 20308 139 6 1 1 32 ");
}
