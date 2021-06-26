#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "array_ops.h"
#include "matrix_calculation_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "register/op_tiling_registry.h"

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

TEST_F(GEMMTiling, GEMM_op_tiling_obj) {
  using namespace optiling;
  std::string op_name = "MatMul";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Matmul", "dynamic_mode":"dynamic_mkn", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 3, 1, 3, 1, 3]}, "block_dim": {"10000": 2}, "attrs":{"transpose_a": false, "transpose_b": false}})";

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

  optiling::utils::OpCompileInfo op_compile_info("Matmul", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(matmul, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
}

TEST_F(GEMMTiling, GEMM_op_tiling_arr) {
  using namespace optiling;
  std::string op_name = "MatMul";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  const ge::AscendString compileInfo = R"([{"_pattern": "Matmul", "dynamic_mode":"dynamic_mkn", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 3, 1, 3, 1, 3]}, "block_dim": {"10000": 2}, "attrs":{"transpose_a": false, "transpose_b": false}},{"_pattern": "Matmul", "dynamic_mode":"dynamic_mkn", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10001": [1, 3, 1, 3, 4, 7]}, "block_dim": {"10001": 2}, "attrs":{"transpose_a": false, "transpose_b": false}}])";

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

  optiling::utils::OpCompileInfo op_compile_info("GEMM_op_tiling_arr", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(matmul, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
}