#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "common/utils/ut_op_util.h"

using namespace std;
using namespace ge;
using namespace op;
using namespace ut_util;

class KLDivTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "KLDivTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "KLDivTiling TearDown" << std::endl;
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

TEST_F(KLDivTiling, KLDivTiling_test_1)
{
  using namespace optiling;
  std::string op_name = "KLDiv";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = R"({ "_ori_axis": [0, 1], "_pattern": "CommReduce", "push_status": 0, "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5, 4, 9], "_ub_info": [16256, 16000, 16256], "_ub_info_rf": [16256, 16000, 16256], "reduce_mean_cof_dtype": "float16", "reduce_mode": 1})";

  ge::Graph graph("KLDivTiling_test_1");

  auto input_x_shape = std::vector<int64_t>({20480, 16});
  ge::TensorDesc input_x_desc(ge::Shape(input_x_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  auto input_x = op::Data("input_x_shape");
  input_x.update_input_desc_x(input_x_desc);
  input_x.update_output_desc_y(input_x_desc);

  auto input_target_shape = std::vector<int64_t>({20480, 16});
  ge::TensorDesc input_target_desc(ge::Shape(input_target_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  auto input_target = op::Data("input_target_shape");
  input_target.update_input_desc_x(input_target_desc);
  input_target.update_output_desc_y(input_target_desc);

  auto klDiv = op::KLDiv(op_name.c_str())
    .set_input_x(input_x)
    .set_attr_reduction("batchmean")
    .set_input_target(input_target);

  auto output_shape = std::vector<int64_t>({1});
  ge::TensorDesc output_desc(ge::Shape(output_shape), ge::FORMAT_ND, ge::DT_FLOAT16);

  klDiv.update_input_desc_x(input_x_desc);
  klDiv.update_input_desc_target(input_target_desc);
  klDiv.update_output_desc_y(output_desc);

  std::vector<ge::Operator> inputs{input_x, input_target};
  std::vector<ge::Operator> outputs{klDiv};

  graph.SetInputs(inputs).SetOutputs(outputs);

  optiling::utils::OpCompileInfo op_compile_info("KLDivTiling_test_1", compileInfo);
  optiling::utils::OpRunInfo runInfo;

  RUN_TILING_V3(klDiv, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 327680 10240 1 819 ");
}

TEST_F(KLDivTiling, KLDivTiling_test_2)
{
  using namespace optiling;
  std::string op_name = "KLDiv";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = R"({ "_ori_axis": [0, 1], "_pattern": "CommReduce", "push_status": 0, "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5, 4, 9], "_ub_info": [16256, 16000, 16256], "_ub_info_rf": [16256, 16000, 16256], "reduce_mean_cof_dtype": "float32", "reduce_mode": 1})";

  ge::Graph graph("KLDivTiling_test_2");

  auto input_x_shape = std::vector<int64_t>({1024, 16, 32});
  ge::TensorDesc input_x_desc(ge::Shape(input_x_shape), ge::FORMAT_ND, ge::DT_FLOAT);
  auto input_x = op::Data("input_x_shape");
  input_x.update_input_desc_x(input_x_desc);
  input_x.update_output_desc_y(input_x_desc);

  auto input_target_shape = std::vector<int64_t>({1024, 16, 32});
  ge::TensorDesc input_target_desc(ge::Shape(input_target_shape), ge::FORMAT_ND, ge::DT_FLOAT);
  auto input_target = op::Data("input_target_shape");
  input_target.update_input_desc_x(input_target_desc);
  input_target.update_output_desc_y(input_target_desc);

  auto klDiv = op::KLDiv(op_name.c_str())
    .set_input_x(input_x)
    .set_attr_reduction("batchmean")
    .set_input_target(input_target);

  auto output_shape = std::vector<int64_t>({1});
  ge::TensorDesc output_desc(ge::Shape(output_shape), ge::FORMAT_ND, ge::DT_FLOAT);

  klDiv.update_input_desc_x(input_x_desc);
  klDiv.update_input_desc_target(input_target_desc);
  klDiv.update_output_desc_y(output_desc);

  std::vector<ge::Operator> inputs{input_x, input_target};
  std::vector<ge::Operator> outputs{klDiv};

  graph.SetInputs(inputs).SetOutputs(outputs);

  optiling::utils::OpCompileInfo op_compile_info("KLDivTiling_test_2", compileInfo);
  optiling::utils::OpRunInfo runInfo;

  RUN_TILING_V3(klDiv, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 16384 32 512 508 981467136 ");
}

TEST_F(KLDivTiling, KLDivTiling_test_3)
{
  using namespace optiling;
  std::string op_name = "KLDiv";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = R"({ "_ub_factor_align": 128,
                                 "_pattern": "ElemWise",
                                 "push_status": 0,
                                 "_flag_info": [false, false, false, true, false, false],
                                 "_base_info": {"100": [32, 4, 13088, 6544]},
                                 "_custom_vars": { "210000000": ["cof"],
                                                   "210010000": ["cof"],
                                                   "2147483647": ["cof"]},
                                 "_elewise_vars": { "210000000": [ 10000, 20000, 30000 ],
                                                    "210010000": [ 10000, 20000, 30000 ] },
                                 "_normal_vars": { "210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"],
                                                   "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"],
                                                   "2147483647": [] },
                                 "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0", "cof"],
                                            "210010000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0", "cof"],
                                            "2147483647": ["cof"] },
                                 "reduce_mean_cof_dtype": "float32",
                                 "reduce_mode": 0})";

  ge::Graph graph("KLDivTiling_test_3");

  auto input_x_shape = std::vector<int64_t>({1024, 16, 32});
  ge::TensorDesc input_x_desc(ge::Shape(input_x_shape), ge::FORMAT_ND, ge::DT_FLOAT);
  auto input_x = op::Data("input_x_shape");
  input_x.update_input_desc_x(input_x_desc);
  input_x.update_output_desc_y(input_x_desc);

  auto input_target_shape = std::vector<int64_t>({1024, 16, 32});
  ge::TensorDesc input_target_desc(ge::Shape(input_target_shape), ge::FORMAT_ND, ge::DT_FLOAT);
  auto input_target = op::Data("input_target_shape");
  input_target.update_input_desc_x(input_target_desc);
  input_target.update_output_desc_y(input_target_desc);

  auto klDiv = op::KLDiv(op_name.c_str())
    .set_input_x(input_x)
    .set_attr_reduction("none")
    .set_input_target(input_target);

  auto output_shape = std::vector<int64_t>({1});
  ge::TensorDesc output_desc(ge::Shape(output_shape), ge::FORMAT_ND, ge::DT_FLOAT);

  klDiv.update_input_desc_x(input_x_desc);
  klDiv.update_input_desc_target(input_target_desc);
  klDiv.update_output_desc_y(output_desc);

  std::vector<ge::Operator> inputs{input_x, input_target};
  std::vector<ge::Operator> outputs{klDiv};

  graph.SetInputs(inputs).SetOutputs(outputs);

  optiling::utils::OpCompileInfo op_compile_info("KLDivTiling_test_3", compileInfo);
  optiling::utils::OpRunInfo runInfo;

  RUN_TILING_V3(klDiv, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "524288 16384 5504 981467136 ");
}