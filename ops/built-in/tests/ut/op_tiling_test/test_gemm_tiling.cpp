#include <fstream>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "array_ops.h"
#include "matrix_calculation_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#define private public
#include "register/op_tiling_registry.h"
#include "common/utils/ut_op_util.h"

using namespace std;
using namespace ge;
using namespace op;

class GEMMTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GEMMTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GEMMTiling TearDown" << std::endl;
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

  return result;
}

TEST_F(GEMMTiling, GEMM_op_tiling_obj_matmul) {
  using namespace optiling;
  std::string op_name = "MatMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo = R"({"_pattern": "Matmul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode":"dynamic_mkn", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 3, 1, 3, 1, 3]}, "block_dim": {"10000": 2}, "attrs":{"transpose_a": false, "transpose_b": false}})";

  ge::Graph graph("matmul_op_tiling_test_0");

  auto x1_shape = vector<int64_t>({2, 3});
  ge::TensorDesc desc_x1(ge::Shape(x1_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x1.SetOriginShape(ge::Shape(x1_shape));
  auto x1 = op::Data("x1");
  x1.update_input_desc_x(desc_x1);
  x1.update_output_desc_y(desc_x1);

  auto x2_shape = vector<int64_t>({3, 4});
  ge::TensorDesc desc_x2(ge::Shape(x2_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x2.SetOriginShape(ge::Shape(x2_shape));
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(desc_x2);
  x2.update_output_desc_y(desc_x2);

  auto matmul = op::MatMul(op_name)
      .set_input_x1(x1)
      .set_input_x2(x2);

  auto y_shape = vector<int64_t>({2, 4});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  matmul.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x1, x2};
  std::vector<Operator> outputs{matmul};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(matmul, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 1 1 ");
}

TEST_F(GEMMTiling, GEMM_op_tiling_obj_batchmatmul_repo) {
  using namespace optiling;
  std::string op_name = "BatchMatMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo =
      R"({"_pattern": "MatMul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode":"dynamic_mknb",
"repo_seeds": {"10114": [128, 32, 32, 32]}, "repo_range": {"10114": [126, 130, 30, 34, 30, 34, 32, 2147483647]},
"attrs":{"transpose_a": false, "transpose_b": false}, "block_dim": {"10114": 32}, "correct_range_flag":null,
"_vars":{"10114":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
"n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
"kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]},
"_custom_vars":{"10114":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
"n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
"kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]}, "_normal_vars":{"10114":[]},
"_attr_vars":{"10114":[]}})";

  ge::Graph graph("batchmatmul_op_tiling_test_0");

  auto x1_shape = vector<int64_t>({32, 2048, 512});
  ge::TensorDesc desc_x1(ge::Shape(x1_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x1.SetOriginShape(ge::Shape(x1_shape));
  auto x1 = op::Data("x1");
  x1.update_input_desc_x(desc_x1);
  x1.update_output_desc_y(desc_x1);

  auto x2_shape = vector<int64_t>({512, 512});
  ge::TensorDesc desc_x2(ge::Shape(x2_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x2.SetOriginShape(ge::Shape(x2_shape));
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(desc_x2);
  x2.update_output_desc_y(desc_x2);

  auto matmul = op::BatchMatMul(op_name).set_input_x1(x1).set_input_x2(x2);

  auto y_shape = vector<int64_t>({32, 2048, 512});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_ND,
                               ge::DT_FLOAT16);
  matmul.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x1, x2};
  std::vector<Operator> outputs{matmul};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr =
      ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(matmul, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(runInfo.GetTilingKey(), 10114);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "32 128 32 32 ");
}

TEST_F(GEMMTiling, GEMM_op_tiling_obj_batchmatmul_formula01) {
  using namespace optiling;
  std::string op_name = "BatchMatMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo =
      R"({"_pattern": "MatMul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode":"dynamic_mknb",
"repo_seeds": {}, "repo_range": {}, "attrs":{"transpose_a": false, "transpose_b": false}, "block_dim": {"2000000": 32}, "correct_range_flag":null,
"_vars":{"2000000":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
"n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
"kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]},
"_custom_vars":{"2000000":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
"n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
"kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]}, "_normal_vars":{"2000000":[]},
"_attr_vars":{"2000000":[]}})";

  ge::Graph graph("batchmatmul_op_tiling_test_1");

  auto x1_shape = vector<int64_t>({1, 16, 512});
  ge::TensorDesc desc_x1(ge::Shape(x1_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x1.SetOriginShape(ge::Shape(x1_shape));
  auto x1 = op::Data("x1");
  x1.update_input_desc_x(desc_x1);
  x1.update_output_desc_y(desc_x1);

  auto x2_shape = vector<int64_t>({512, 16});
  ge::TensorDesc desc_x2(ge::Shape(x2_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x2.SetOriginShape(ge::Shape(x2_shape));
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(desc_x2);
  x2.update_output_desc_y(desc_x2);

  auto matmul = op::BatchMatMul(op_name).set_input_x1(x1).set_input_x2(x2);

  auto y_shape = vector<int64_t>({1, 16, 16});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_ND,
                               ge::DT_FLOAT16);
  matmul.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x1, x2};
  std::vector<Operator> outputs{matmul};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr =
      ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(matmul, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(runInfo.GetTilingKey(), 2000000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "32 1 1 1 32 1 1 1 1 1 1 1 1 1 1 1 32 1 1 1 1 1 32 32 1 ");
}

TEST_F(GEMMTiling, GEMM_op_tiling_obj_batchmatmul_formula02) {
  using namespace optiling;
  std::string op_name = "BatchMatMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo =
      R"({"_pattern": "MatMul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode":"dynamic_mknb",
"repo_seeds": {}, "repo_range": {}, "attrs":{"transpose_a": false, "transpose_b": false}, "block_dim": {"2000100": 32}, "correct_range_flag":null,
"_vars":{"2000100":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
"n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
"kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]},
"_custom_vars":{"2000100":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
"n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
"kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]}, "_normal_vars":{"2000100":[]},
"_attr_vars":{"2000100":[]}})";

  ge::Graph graph("batchmatmul_op_tiling_test_2");

  auto x1_shape = vector<int64_t>({1, 240, 768});
  ge::TensorDesc desc_x1(ge::Shape(x1_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x1.SetOriginShape(ge::Shape(x1_shape));
  auto x1 = op::Data("x1");
  x1.update_input_desc_x(desc_x1);
  x1.update_output_desc_y(desc_x1);

  auto x2_shape = vector<int64_t>({768, 6000});
  ge::TensorDesc desc_x2(ge::Shape(x2_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x2.SetOriginShape(ge::Shape(x2_shape));
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(desc_x2);
  x2.update_output_desc_y(desc_x2);

  auto matmul = op::BatchMatMul(op_name).set_input_x1(x1).set_input_x2(x2);

  auto y_shape = vector<int64_t>({1, 240, 6000});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_ND,
                               ge::DT_FLOAT16);
  matmul.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x1, x2};
  std::vector<Operator> outputs{matmul};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr =
      ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(matmul, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 15);
  EXPECT_EQ(runInfo.GetTilingKey(), 2000100);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "48 15 375 1 48 1 1 1 1 15 1 1 3 5 5 5 12 1 4 4 1 1 48 48 1 ");
}

TEST_F(GEMMTiling, GEMM_op_tiling_obj_batchmatmul_formula03) {
  using namespace optiling;
  std::string op_name = "BatchMatMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo =
      R"({"_pattern": "MatMul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode":"dynamic_mknb",
"repo_seeds": {}, "repo_range": {}, "attrs":{"transpose_a": false, "transpose_b": false}, "block_dim": {"8022100": 32}, "correct_range_flag":null,
"_vars":{"8022100":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
"n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
"kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]},
"_custom_vars":{"8022100":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
"n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
"kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]}, "_normal_vars":{"8022100":[]},
"_attr_vars":{"8022100":[]}})";

  ge::Graph graph("batchmatmul_op_tiling_test_3");

  auto x1_shape = vector<int64_t>({1, 16, 2048});
  ge::TensorDesc desc_x1(ge::Shape(x1_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x1.SetOriginShape(ge::Shape(x1_shape));
  auto x1 = op::Data("x1");
  x1.update_input_desc_x(desc_x1);
  x1.update_output_desc_y(desc_x1);

  auto x2_shape = vector<int64_t>({2048, 1008});
  ge::TensorDesc desc_x2(ge::Shape(x2_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x2.SetOriginShape(ge::Shape(x2_shape));
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(desc_x2);
  x2.update_output_desc_y(desc_x2);

  auto matmul = op::BatchMatMul(op_name).set_input_x1(x1).set_input_x2(x2);

  auto y_shape = vector<int64_t>({1, 16, 1008});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_ND,
                               ge::DT_FLOAT16);
  matmul.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x1, x2};
  std::vector<Operator> outputs{matmul};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr =
      ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(matmul, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 7);
  EXPECT_EQ(runInfo.GetTilingKey(), 8022100);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "128 1 63 1 128 1 1 1 1 7 1 1 1 1 9 1 4 1 8 8 4 4 32 32 1 ");
}

TEST_F(GEMMTiling, GEMM_op_tiling_obj_batchmatmul_formula04) {
  using namespace optiling;
  std::string op_name = "BatchMatMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo =
      R"({"_pattern": "MatMul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode":"dynamic_mknb",
"repo_seeds": {}, "repo_range": {}, "attrs":{"transpose_a": false, "transpose_b": false}, "block_dim": {"7022000": 32}, "correct_range_flag":null,
"_vars":{"7022000":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
"n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
"kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]},
"_custom_vars":{"7022000":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
"n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
"kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]}, "_normal_vars":{"7022000":[]},
"_attr_vars":{"7022000":[]}})";

  ge::Graph graph("batchmatmul_op_tiling_test_4");

  auto x1_shape = vector<int64_t>({1, 4096, 3136});
  ge::TensorDesc desc_x1(ge::Shape(x1_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x1.SetOriginShape(ge::Shape(x1_shape));
  auto x1 = op::Data("x1");
  x1.update_input_desc_x(desc_x1);
  x1.update_output_desc_y(desc_x1);

  auto x2_shape = vector<int64_t>({3136, 1024});
  ge::TensorDesc desc_x2(ge::Shape(x2_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x2.SetOriginShape(ge::Shape(x2_shape));
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(desc_x2);
  x2.update_output_desc_y(desc_x2);

  auto matmul = op::BatchMatMul(op_name).set_input_x1(x1).set_input_x2(x2);

  auto y_shape = vector<int64_t>({1, 4096, 1024});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_ND,
                               ge::DT_FLOAT16);
  matmul.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x1, x2};
  std::vector<Operator> outputs{matmul};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr =
      ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(matmul, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(runInfo.GetTilingKey(), 7022000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "196 256 64 1 196 1 2 1 1 4 8 1 1 1 16 16 4 1 1 1 49 49 4 4 1 ");
}

TEST_F(GEMMTiling, GEMM_op_tiling_obj_batchmatmul_formula05) {
  using namespace optiling;
  std::string op_name = "BatchMatMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo =
      R"({"_pattern": "MatMul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode":"dynamic_mknb",
"repo_seeds": {}, "repo_range": {}, "attrs":{"transpose_a": false, "transpose_b": false}, "block_dim": {"6221100": 32}, "correct_range_flag":null,
"_vars":{"6221100":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
"n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
"kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]},
"_custom_vars":{"6221100":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
"n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
"kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]}, "_normal_vars":{"6221100":[]},
"_attr_vars":{"6221100":[]}})";

  ge::Graph graph("batchmatmul_op_tiling_test_5");

  auto x1_shape = vector<int64_t>({1, 16000, 1024});
  ge::TensorDesc desc_x1(ge::Shape(x1_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x1.SetOriginShape(ge::Shape(x1_shape));
  auto x1 = op::Data("x1");
  x1.update_input_desc_x(desc_x1);
  x1.update_output_desc_y(desc_x1);

  auto x2_shape = vector<int64_t>({1024, 512});
  ge::TensorDesc desc_x2(ge::Shape(x2_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x2.SetOriginShape(ge::Shape(x2_shape));
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(desc_x2);
  x2.update_output_desc_y(desc_x2);

  auto matmul = op::BatchMatMul(op_name).set_input_x1(x1).set_input_x2(x2);

  auto y_shape = vector<int64_t>({1, 16000, 512});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_ND,
                               ge::DT_FLOAT16);
  matmul.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x1, x2};
  std::vector<Operator> outputs{matmul};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr =
      ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(matmul, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 25);
  EXPECT_EQ(runInfo.GetTilingKey(), 6221100);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "64 1000 32 1 64 1 5 2 1 1 25 1 1 1 16 8 4 1 4 16 4 1 16 64 4 ");
}

TEST_F(GEMMTiling, GEMM_op_tiling_obj_batchmatmul_formula06) {
  using namespace optiling;
  std::string op_name = "BatchMatMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo =
      R"({"_pattern": "MatMul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode":"dynamic_mknb",
"repo_seeds": {}, "repo_range": {}, "attrs":{"transpose_a": false, "transpose_b": false}, "block_dim": {"2000000": 32}, "correct_range_flag":null,
"_vars":{"2000000":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
"n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
"kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]},
"_custom_vars":{"2000000":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
"n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
"kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]}, "_normal_vars":{"2000000":[]},
"_attr_vars":{"2000000":[]}})";

  ge::Graph graph("batchmatmul_op_tiling_test_6");

  auto x1_shape = vector<int64_t>({1, 128, 128});
  ge::TensorDesc desc_x1(ge::Shape(x1_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x1.SetOriginShape(ge::Shape(x1_shape));
  auto x1 = op::Data("x1");
  x1.update_input_desc_x(desc_x1);
  x1.update_output_desc_y(desc_x1);

  auto x2_shape = vector<int64_t>({128, 64});
  ge::TensorDesc desc_x2(ge::Shape(x2_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x2.SetOriginShape(ge::Shape(x2_shape));
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(desc_x2);
  x2.update_output_desc_y(desc_x2);

  auto matmul = op::BatchMatMul(op_name).set_input_x1(x1).set_input_x2(x2);

  auto y_shape = vector<int64_t>({1, 128, 64});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  matmul.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x1, x2};
  std::vector<Operator> outputs{matmul};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(matmul, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(runInfo.GetTilingKey(), 2000000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "8 8 4 1 8 1 1 1 1 4 8 1 1 1 1 1 8 1 1 1 1 1 8 8 1 ");
}

TEST_F(GEMMTiling, GEMM_op_tiling_arr) {
  using namespace optiling;
  std::string op_name = "MatMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo = R"([{"_pattern": "Matmul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ",
  "dynamic_mode":"dynamic_mkn", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 3, 1, 3, 1, 3]},
  "block_dim": {"10000": 2}, "attrs":{"transpose_a": false, "transpose_b": false}},{"_pattern": "Matmul",
  "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode":"dynamic_mkn", "repo_seeds": {}, "repo_range": {},
  "cost_range": {"10001": [1, 3, 1, 3, 4, 7]}, "block_dim": {"10001": 2}, "attrs":{"transpose_a": false,
  "transpose_b": false}}])";

  ge::Graph graph("matmul_op_tiling_test_1");

  auto x1_shape = vector<int64_t>({2, 3});
  ge::TensorDesc desc_x1(ge::Shape(x1_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x1.SetOriginShape(ge::Shape(x1_shape));
  auto x1 = op::Data("x1");
  x1.update_input_desc_x(desc_x1);
  x1.update_output_desc_y(desc_x1);

  auto x2_shape = vector<int64_t>({3, 4});
  ge::TensorDesc desc_x2(ge::Shape(x2_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x2.SetOriginShape(ge::Shape(x2_shape));
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(desc_x2);
  x2.update_output_desc_y(desc_x2);

  auto matmul = op::MatMul(op_name)
      .set_input_x1(x1)
      .set_input_x2(x2);

  auto y_shape = vector<int64_t>({2, 4});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  matmul.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x1, x2};
  std::vector<Operator> outputs{matmul};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(matmul, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 1 1 ");
}

TEST_F(GEMMTiling, GEMM_op_tiling_obj_nd) {
  using namespace optiling;
  std::string op_name = "MatMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo = R"({"_pattern": "Matmul", "format_a": "ND", "format_b": "ND", "dynamic_mode":"dynamic_mkn", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 3, 1, 3, 1, 3]}, "block_dim": {"10000": 2}, "attrs":{"transpose_a": false, "transpose_b": false}})";

  ge::Graph graph("matmul_op_tiling_test_2");

  auto x1_shape = vector<int64_t>({2, 3});
  ge::TensorDesc desc_x1(ge::Shape(x1_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x1.SetOriginShape(ge::Shape(x1_shape));
  auto x1 = op::Data("x1");
  x1.update_input_desc_x(desc_x1);
  x1.update_output_desc_y(desc_x1);

  auto x2_shape = vector<int64_t>({3, 4});
  ge::TensorDesc desc_x2(ge::Shape(x2_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x2.SetOriginShape(ge::Shape(x2_shape));
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(desc_x2);
  x2.update_output_desc_y(desc_x2);

  auto matmul = op::MatMul(op_name)
      .set_input_x1(x1)
      .set_input_x2(x2);

  auto y_shape = vector<int64_t>({2, 4});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  matmul.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x1, x2};
  std::vector<Operator> outputs{matmul};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(matmul, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "3 2 4 ");
}

TEST_F(GEMMTiling, GEMM_op_tiling_nd_aligned_pattern) {
  using namespace optiling;
  std::string op_name = "MatMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo = R"({"_pattern": "Matmul", "format_a": "ND", "format_b": "ND", "dynamic_mode":"dynamic_mkn", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 3, 2, 4, 1, 3]}, "block_dim": {"10000": 2}, "attrs":{"transpose_a": false, "transpose_b": false}})";

  ge::Graph graph("op_tiling_test_nd_aligned_pattern");

  auto x1_shape = vector<int64_t>({16, 32});
  ge::TensorDesc desc_x1(ge::Shape(x1_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x1.SetOriginShape(ge::Shape(x1_shape));
  auto x1 = op::Data("x1");
  x1.update_input_desc_x(desc_x1);
  x1.update_output_desc_y(desc_x1);

  auto x2_shape = vector<int64_t>({32, 16});
  ge::TensorDesc desc_x2(ge::Shape(x2_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x2.SetOriginShape(ge::Shape(x2_shape));
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(desc_x2);
  x2.update_output_desc_y(desc_x2);

  auto matmul = op::MatMul(op_name)
      .set_input_x1(x1)
      .set_input_x2(x2);

  auto y_shape = vector<int64_t>({16, 16});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  matmul.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x1, x2};
  std::vector<Operator> outputs{matmul};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(matmul, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  // In Aligned Mode. The key is changed
  EXPECT_EQ(runInfo.GetTilingKey(), 20000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "32 16 16 ");
}

TEST_F(GEMMTiling, GEMM_op_tiling_nd_nonrange_pattern) {
  using namespace optiling;
  std::string op_name = "BatchMatMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo =
      R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":false,"nd_flag":true, "split_k_flag":false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mknb",
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  ge::Graph graph("op_tiling_nd_nonrange_pattern");

  auto x1_shape = vector<int64_t>({4352, 16, 64});
  ge::TensorDesc desc_x1(ge::Shape(x1_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x1.SetOriginShape(ge::Shape(x1_shape));
  auto x1 = op::Data("x1");
  x1.update_input_desc_x(desc_x1);
  x1.update_output_desc_y(desc_x1);

  auto x2_shape = vector<int64_t>({4352, 64, 32});
  ge::TensorDesc desc_x2(ge::Shape(x2_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x2.SetOriginShape(ge::Shape(x2_shape));
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(desc_x2);
  x2.update_output_desc_y(desc_x2);

  auto matmul = op::BatchMatMul(op_name)
      .set_input_x1(x1)
      .set_input_x2(x2);

  auto y_shape = vector<int64_t>({4352, 16, 32});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  matmul.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x1, x2};
  std::vector<Operator> outputs{matmul};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(matmul, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  // In Aligned Mode. The key is changed
  EXPECT_EQ(runInfo.GetTilingKey(), 800001100);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "64 1 2 4352 4 136 1 1 32 1 1 1 1 1 2 1 4 1 1 1 1 1 4 4 1 1 2 4 4 1 1 1 1 80 1 1280 2048 ");
}

TEST_F(GEMMTiling, GEMM_op_tiling_nd_nonrange_pattern_02) {
  using namespace optiling;
  std::string op_name = "BatchMatMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find (op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo =
      R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":true},
      "binary_attrs":{"bias_flag":false,"nd_flag":true, "split_k_flag":false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mknb",
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  ge::Graph graph("op_tiling_nd_nonrange_pattern_02");

  auto x1_shape = vector<int64_t>({4352, 64, 16});
  ge::TensorDesc desc_x1(ge::Shape(x1_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x1.SetOriginShape(ge::Shape(x1_shape));
  auto x1 = op::Data("x1");
  x1.update_input_desc_x(desc_x1);
  x1.update_output_desc_y(desc_x1);

  auto x2_shape = vector<int64_t>({4352, 32, 64});
  ge::TensorDesc desc_x2(ge::Shape(x2_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x2.SetOriginShape(ge::Shape(x2_shape));
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(desc_x2);
  x2.update_output_desc_y(desc_x2);

  auto matmul = op::BatchMatMul(op_name)
      .set_input_x1(x1)
      .set_input_x2(x2);

  auto y_shape = vector<int64_t>({4352, 16, 32});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  matmul.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x1, x2};
  std::vector<Operator> outputs{matmul};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(matmul, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  // In Aligned Mode. The key is changed
  EXPECT_EQ(runInfo.GetTilingKey(), 800001100);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "64 1 2 4352 4 136 1 1 32 1 1 1 1 1 2 1 4 1 1 1 1 1 4 4 1 1 2 4 4 1 1 1 1 1 80 1024 2560 ");
}

TEST_F(GEMMTiling, GEMM_op_tiling_nd_nonrange_pattern_03) {
  using namespace optiling;
  std::string op_name = "MatMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo =
      R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true},
      "binary_attrs":{"bias_flag":false,"nd_flag":true, "split_k_flag":false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mknb",
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  ge::Graph graph("op_tiling_nd_nonrange_pattern_02");

  auto x1_shape = vector<int64_t>({8192, 512});
  ge::TensorDesc desc_x1(ge::Shape(x1_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x1.SetOriginShape(ge::Shape(x1_shape));
  auto x1 = op::Data("x1");
  x1.update_input_desc_x(desc_x1);
  x1.update_output_desc_y(desc_x1);

  auto x2_shape = vector<int64_t>({512, 512});
  ge::TensorDesc desc_x2(ge::Shape(x2_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x2.SetOriginShape(ge::Shape(x2_shape));
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(desc_x2);
  x2.update_output_desc_y(desc_x2);

  auto matmul = op::MatMul(op_name)
      .set_input_x1(x1)
      .set_input_x2(x2);

  auto y_shape = vector<int64_t>({8192, 512});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  matmul.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x1, x2};
  std::vector<Operator> outputs{matmul};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(matmul, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  // In Aligned Mode. The key is changed
  EXPECT_EQ(runInfo.GetTilingKey(), 622010000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "512 512 32 1 32 1 8 1 1 4 8 1 1 1 4 8 8 2 2 4 2 1 16 32 2 4 1 16 32 8 2 1 1 272 528 17408 8448 ");
}

TEST_F(GEMMTiling, GEMM_op_tiling_nd_nonrange_pattern_04) {
  using namespace optiling;
  std::string op_name = "MatMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo =
      R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false},
      "binary_attrs":{"bias_flag":false,"nd_flag":true, "split_k_flag":false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mknb",
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  ge::Graph graph("op_tiling_nd_nonrange_pattern_04");

  auto x1_shape = vector<int64_t>({40960, 512});
  ge::TensorDesc desc_x1(ge::Shape(x1_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x1.SetOriginShape(ge::Shape(x1_shape));
  auto x1 = op::Data("x1");
  x1.update_input_desc_x(desc_x1);
  x1.update_output_desc_y(desc_x1);

  auto x2_shape = vector<int64_t>({40960, 1024});
  ge::TensorDesc desc_x2(ge::Shape(x2_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x2.SetOriginShape(ge::Shape(x2_shape));
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(desc_x2);
  x2.update_output_desc_y(desc_x2);

  auto matmul = op::MatMul(op_name)
      .set_input_x1(x1)
      .set_input_x2(x2);

  auto y_shape = vector<int64_t>({512, 1024});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  matmul.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x1, x2};
  std::vector<Operator> outputs{matmul};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(matmul, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  // In Aligned Mode. The key is changed
  EXPECT_EQ(runInfo.GetTilingKey(), 802210000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "40960 32 64 1 2560 1 1 1 1 8 4 1 1 1 2 8 8 4 2 2 160 160 16 16 1 8 8 8 4 1 1 2 4 144 144 18432 9216 ");
}

TEST_F(GEMMTiling, GEMM_op_tiling_nd_nonrange_pattern_05) {
  using namespace optiling;
  std::string op_name = "MatMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo =
      R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":false,"nd_flag":true, "split_k_flag":false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mknb",
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  ge::Graph graph("op_tiling_nd_nonrange_pattern_05");

  auto x1_shape = vector<int64_t>({11776, 2048});
  ge::TensorDesc desc_x1(ge::Shape(x1_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x1.SetOriginShape(ge::Shape(x1_shape));
  auto x1 = op::Data("x1");
  x1.update_input_desc_x(desc_x1);
  x1.update_output_desc_y(desc_x1);

  auto x2_shape = vector<int64_t>({2048, 512});
  ge::TensorDesc desc_x2(ge::Shape(x2_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x2.SetOriginShape(ge::Shape(x2_shape));
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(desc_x2);
  x2.update_output_desc_y(desc_x2);

  auto matmul = op::MatMul(op_name)
      .set_input_x1(x1)
      .set_input_x2(x2);

  auto y_shape = vector<int64_t>({11776, 512});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  matmul.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x1, x2};
  std::vector<Operator> outputs{matmul};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(matmul, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  // In Aligned Mode. The key is changed
  EXPECT_EQ(runInfo.GetTilingKey(), 522011000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2048 736 32 1 128 1 4 1 1 4 8 1 1 1 1 23 2 8 2 64 32 1 4 128 32 23 8 4 2 1 1 1 64 80 144 29440 4608 ");
}

TEST_F(GEMMTiling, GEMM_op_tiling_nd_nonrange_pattern_split_k) {
  using namespace optiling;
  std::string op_name = "MatMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo =
      R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false},
      "binary_attrs":{"bias_flag":false,"nd_flag":true, "split_k_flag":true, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn",
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  ge::Graph graph("op_tiling_nd_nonrange_pattern_split_k");

  auto x1_shape = vector<int64_t>({11776, 512});
  ge::TensorDesc desc_x1(ge::Shape(x1_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x1.SetOriginShape(ge::Shape(x1_shape));
  auto x1 = op::Data("x1");
  x1.update_input_desc_x(desc_x1);
  x1.update_output_desc_y(desc_x1);

  auto x2_shape = vector<int64_t>({11776, 512});
  ge::TensorDesc desc_x2(ge::Shape(x2_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x2.SetOriginShape(ge::Shape(x2_shape));
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(desc_x2);
  x2.update_output_desc_y(desc_x2);

  auto matmul = op::MatMul(op_name)
      .set_input_x1(x1)
      .set_input_x2(x2);

  auto y_shape = vector<int64_t>({512, 512});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT);
  matmul.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x1, x2};
  std::vector<Operator> outputs{matmul};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(matmul, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  // In Aligned Mode. The key is changed
  EXPECT_EQ(runInfo.GetTilingKey(), 1622001000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "11776 32 32 736 1 2 1 1 4 2 4 1 1 4 8 8 2 1 23 23 1 8 184 23 8 8 8 4 1 1 1 46 144 144 18432 9216 ");
}

TEST_F(GEMMTiling, GEMM_op_tiling_nd_nonrange_ub_reused) {
  using namespace optiling;
  std::string op_name = "MatMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo =
      R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true},
      "binary_attrs":{"bias_flag":false,"nd_flag":true, "split_k_flag":false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn",
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  ge::Graph graph("op_tiling_nd_nonrange_ub_reused");

  auto x1_shape = vector<int64_t>({8192, 512});
  ge::TensorDesc desc_x1(ge::Shape(x1_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x1.SetOriginShape(ge::Shape(x1_shape));
  auto x1 = op::Data("x1");
  x1.update_input_desc_x(desc_x1);
  x1.update_output_desc_y(desc_x1);

  auto x2_shape = vector<int64_t>({1024, 512});
  ge::TensorDesc desc_x2(ge::Shape(x2_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x2.SetOriginShape(ge::Shape(x2_shape));
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(desc_x2);
  x2.update_output_desc_y(desc_x2);

  auto matmul = op::MatMul(op_name)
      .set_input_x1(x1)
      .set_input_x2(x2);

  auto y_shape = vector<int64_t>({8192, 1024});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  matmul.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x1, x2};
  std::vector<Operator> outputs{matmul};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(matmul, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetTilingKey(), 200010000);
}

TEST_F(GEMMTiling, GEMM_op_tiling_fractal_z) {
  using namespace optiling;
  std::string op_name = "MatMul";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo =
      R"([{"_pattern": "Matmul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode":"dynamic_mkn",
      "repo_seeds": {}, "repo_range": {}, "cost_range": {}, "block_dim": {"2000000": 6},
      "attrs":{"transpose_a": false, "transpose_b": true}},{"_pattern": "Matmul", "format_a": "FRACTAL_NZ",
      "format_b": "FRACTAL_NZ", "dynamic_mode":"dynamic_mkn", "repo_seeds": {}, "repo_range": {}, "cost_range": {},
      "block_dim": {"2000000": 6},
      "attrs":{"transpose_a": false, "transpose_b": true}}])";

  ge::Graph graph("matmul_op_tiling_test_7");

  auto x1_shape = vector<int64_t>({1, 256});
  ge::TensorDesc desc_x1(ge::Shape(x1_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x1.SetOriginShape(ge::Shape(x1_shape));
  auto x1 = op::Data("x1");
  x1.update_input_desc_x(desc_x1);
  x1.update_output_desc_y(desc_x1);

  auto x2_shape = vector<int64_t>({96, 256});
  ge::TensorDesc desc_x2(ge::Shape(x2_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_x2.SetOriginShape(ge::Shape(x2_shape));
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(desc_x2);
  x2.update_output_desc_y(desc_x2);

  auto matmul = op::MatMul(op_name)
      .set_input_x1(x1)
      .set_input_x2(x2);

  matmul.SetAttr("input_size", 17);
  matmul.SetAttr("hidden_size", 50);

  auto y_shape = vector<int64_t>({1, 96});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
  matmul.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x1, x2};
  std::vector<Operator> outputs{matmul};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(matmul, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 6);
  EXPECT_EQ(runInfo.GetTilingKey(), 2000000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "16 1 6 16 1 1 1 1 6 1 1 1 1 1 1 16 1 1 1 1 1 16 16 1 ");
}