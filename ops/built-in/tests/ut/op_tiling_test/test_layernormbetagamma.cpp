#include <gtest/gtest.h>

#include <iostream>
#include <vector>
#define private public
#include <iostream>

#include "array_ops.h"
#include "common/utils/ut_op_util.h"
#include "nn_norm_ops.h"
#include "op_tiling/op_tiling_util.h"
#include "register/op_tiling_registry.h"
#include "test_common.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class LayerNormBetaGammaTiling : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "LayerNormBetaGammaTiling SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "LayerNormBetaGammaTiling TearDown" << std::endl; }
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

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_0) {
  std::string op_name = "LayerNormBetaGammaBackprop";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("LayerNormBetaGammaBackprop");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"batch_cols_padding\":4835, \"k_num\":139}}";

  std::vector<int64_t> input_tensor_shape{2, 3, 512};

  TensorDesc input_tensor;
  input_tensor.SetShape(ge::Shape(input_tensor_shape));
  input_tensor.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::LayerNormBetaGammaBackprop(op_name);
  TENSOR_INPUT(opParas, input_tensor, dy);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_1) {
  std::string op_name = "LayerNormBetaGammaBackpropV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("LayerNormBetaGammaBackpropV2");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, "
      "\"shape_gamma\":[512],\"dynamic_reduce\":true,\"dynamic_normal\":false}";

  std::vector<int64_t> input_tensor_shape{2, 3, 512};

  TensorDesc input_tensor;
  input_tensor.SetShape(ge::Shape(input_tensor_shape));
  input_tensor.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::LayerNormBetaGammaBackpropV2(op_name);
  TENSOR_INPUT(opParas, input_tensor, dy);
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_2) {
  std::string op_name = "LayerNormBetaGammaBackpropV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("LayerNormBetaGammaBackpropV2");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, "
      "\"shape_gamma\":[512],\"dynamic_reduce\":true,\"dynamic_normal\":false}";

  std::vector<int64_t> input_tensor_shape{32, 10, 512};

  TensorDesc input_tensor;
  input_tensor.SetShape(ge::Shape(input_tensor_shape));
  input_tensor.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::LayerNormBetaGammaBackpropV2(op_name);
  TENSOR_INPUT(opParas, input_tensor, dy);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_3) {
  std::string op_name = "LayerNormBetaGammaBackpropV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("LayerNormBetaGammaBackpropV2");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, "
      "\"shape_gamma\":[512],\"dynamic_reduce\":true,\"dynamic_normal\":false}";

  std::vector<int64_t> input_tensor_shape{32, 51, 512};
  TensorDesc input_tensor;
  input_tensor.SetShape(ge::Shape(input_tensor_shape));
  input_tensor.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::LayerNormBetaGammaBackpropV2(op_name);
  TENSOR_INPUT(opParas, input_tensor, dy);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_4) {
  std::string op_name = "LayerNormBetaGammaBackpropV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("LayerNormBetaGammaBackpropV2");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, "
      "\"shape_gamma\":[4096],\"dynamic_reduce\":true,\"dynamic_normal\":false}";

  std::vector<int64_t> input_tensor_shape{32, 51, 4096};

  TensorDesc input_tensor;
  input_tensor.SetShape(ge::Shape(input_tensor_shape));
  input_tensor.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::LayerNormBetaGammaBackpropV2(op_name);
  TENSOR_INPUT(opParas, input_tensor, dy);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_5) {
  std::string op_name = "LayerNormBetaGammaBackpropV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("LayerNormBetaGammaBackpropV2");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, "
      "\"shape_gamma\":[512],\"dynamic_reduce\":false,\"dynamic_normal\":true}";

  std::vector<int64_t> input_tensor_shape{2, 3, 512};
  TensorDesc input_tensor;
  input_tensor.SetShape(ge::Shape(input_tensor_shape));
  input_tensor.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::LayerNormBetaGammaBackpropV2(op_name);
  TENSOR_INPUT(opParas, input_tensor, dy);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_6) {
  std::string op_name = "LayerNormBetaGammaBackpropV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("LayerNormBetaGammaBackpropV2");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, "
      "\"shape_gamma\":[512],\"dynamic_reduce\":false,\"dynamic_normal\":true}";

  std::vector<int64_t> input_tensor_shape{32, 10, 512};
  TensorDesc input_tensor;
  input_tensor.SetShape(ge::Shape(input_tensor_shape));
  input_tensor.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::LayerNormBetaGammaBackpropV2(op_name);
  TENSOR_INPUT(opParas, input_tensor, dy);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_7) {
  std::cout << "tiling7 start." << std::endl;
  std::string op_name = "LayerNormBetaGammaBackpropV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("LayerNormBetaGammaBackpropV2");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, "
      "\"shape_gamma\":[512],\"dynamic_reduce\":false,\"dynamic_normal\":true}";

  std::vector<int64_t> input_tensor_shape{32, 100, 512};
  TensorDesc input_tensor;
  input_tensor.SetShape(ge::Shape(input_tensor_shape));
  input_tensor.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::LayerNormBetaGammaBackpropV2(op_name);
  TENSOR_INPUT(opParas, input_tensor, dy);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_8) {
  std::cout << "tiling8 start." << std::endl;
  std::string op_name = "LayerNormBetaGammaBackpropV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("LayerNormBetaGammaBackpropV2");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, "
      "\"shape_gamma\":[4096],\"dynamic_reduce\":false,\"dynamic_normal\":true}";

  std::vector<int64_t> input_tensor_shape{32, 51, 4096};

  TensorDesc input_tensor;
  input_tensor.SetShape(ge::Shape(input_tensor_shape));
  input_tensor.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::LayerNormBetaGammaBackpropV2(op_name);
  TENSOR_INPUT(opParas, input_tensor, dy);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_9) {
  std::cout << "tiling9 start." << std::endl;
  std::string op_name = "LayerNormBetaGammaBackpropV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("LayerNormBetaGammaBackpropV2");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, "
      "\"shape_gamma\":[512],\"dynamic_reduce\":true,\"dynamic_normal\":true}";

  std::vector<int64_t> input_tensor_shape{2, 3, 512};
  TensorDesc input_tensor;
  input_tensor.SetShape(ge::Shape(input_tensor_shape));
  input_tensor.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::LayerNormBetaGammaBackpropV2(op_name);
  TENSOR_INPUT(opParas, input_tensor, dy);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_10) {
  std::cout << "tiling10 start." << std::endl;
  std::string op_name = "LayerNormBetaGammaBackpropV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("LayerNormBetaGammaBackpropV2");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, "
      "\"shape_gamma\":[512],\"dynamic_reduce\":true,\"dynamic_normal\":true}";

  std::vector<int64_t> input_tensor_shape{32, 10, 512};
  TensorDesc input_tensor;
  input_tensor.SetShape(ge::Shape(input_tensor_shape));
  input_tensor.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::LayerNormBetaGammaBackpropV2(op_name);
  TENSOR_INPUT(opParas, input_tensor, dy);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_11) {
  std::cout << "tiling11 start." << std::endl;
  std::string op_name = "LayerNormBetaGammaBackpropV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("LayerNormBetaGammaBackpropV2");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, "
      "\"shape_gamma\":[512],\"dynamic_reduce\":true,\"dynamic_normal\":true}";

  std::vector<int64_t> input_tensor_shape{32, 100, 512};
  TensorDesc input_tensor;
  input_tensor.SetShape(ge::Shape(input_tensor_shape));
  input_tensor.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::LayerNormBetaGammaBackpropV2(op_name);
  TENSOR_INPUT(opParas, input_tensor, dy);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_12) {
  std::cout << "tiling12 start." << std::endl;
  std::string op_name = "LayerNormBetaGammaBackpropV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("LayerNormBetaGammaBackpropV2");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, "
      "\"shape_gamma\":[4096],\"dynamic_reduce\":true,\"dynamic_normal\":true}";

  std::vector<int64_t> input_tensor_shape{32, 51, 4096};
  TensorDesc input_tensor;
  input_tensor.SetShape(ge::Shape(input_tensor_shape));
  input_tensor.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::LayerNormBetaGammaBackpropV2(op_name);
  TENSOR_INPUT(opParas, input_tensor, dy);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_13) {
  std::cout << "tiling13 start." << std::endl;
  std::string op_name = "LayerNormBetaGammaBackpropV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("LayerNormBetaGammaBackpropV2");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, "
      "\"shape_gamma\":[2,3,512],\"dynamic_reduce\":true,\"dynamic_normal\":false}";

  std::vector<int64_t> input_tensor_shape{2, 3, 512};
  TensorDesc input_tensor;
  input_tensor.SetShape(ge::Shape(input_tensor_shape));
  input_tensor.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::LayerNormBetaGammaBackpropV2(op_name);
  TENSOR_INPUT(opParas, input_tensor, dy);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_14) {
  std::cout << "tiling14 start." << std::endl;
  std::string op_name = "LayerNormBetaGammaBackpropV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("LayerNormBetaGammaBackpropV2");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, "
      "\"shape_gamma\":[32,100,512],\"dynamic_reduce\":true,\"dynamic_normal\":false}";

  std::vector<int64_t> input_tensor_shape{32, 100, 512};
  TensorDesc input_tensor;
  input_tensor.SetShape(ge::Shape(input_tensor_shape));
  input_tensor.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::LayerNormBetaGammaBackpropV2(op_name);
  TENSOR_INPUT(opParas, input_tensor, dy);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_15) {
  std::cout << "tiling15 start." << std::endl;
  std::string op_name = "LayerNormBetaGammaBackpropV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("LayerNormBetaGammaBackpropV2");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"core_num\": 0, \"max_reduce_factor\":50, \"max_last_factor\":2048, "
      "\"shape_gamma\":[32,100,512],\"dynamic_reduce\":true,\"dynamic_normal\":false}";

  std::vector<int64_t> input_tensor_shape{32, 100, 512};
  TensorDesc input_tensor;
  input_tensor.SetShape(ge::Shape(input_tensor_shape));
  input_tensor.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::LayerNormBetaGammaBackpropV2(op_name);
  TENSOR_INPUT(opParas, input_tensor, dy);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}
