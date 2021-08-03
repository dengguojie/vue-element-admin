#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/attr_utils.h"
#include "nn_norm_ops.h"
#include "array_ops.h"

using namespace std;
using namespace ge;

class NormTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
      std::cout << "NormTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "NormTiling TearDown" << std::endl;
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

TEST_F(NormTiling, NormTiling1) {
  using namespace optiling;

  ge::Graph graph("NormTiling1");
  std::vector<int64_t> input{2, 10496, 41};
  std::vector<int64_t> output{2, 10496, 41};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT16);

  auto x = op::Data("x");
  x.update_input_desc_x(tensor_input);
  x.update_output_desc_y(tensor_output);

  auto softmax_op = op::SoftmaxV2("SoftmaxV2_1");
  softmax_op.set_input_x(x);
  softmax_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x};
  std::vector<Operator> outputs{softmax_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string op_name = "AutoTiling";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  std::string compileInfo = R"({ "_ori_axis": [2], "_pattern": "Norm", "_common_info": [32, 16, 1, 16080, 16120], "_workspace_info": {"_workspace_type": [1, 0, 0], "_workspace_bytes": [4, 4, 4]}, "_vars": {"400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]}})";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo.c_str());
  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(iter->second(softmax_op, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "20992 41 656 328 ");
}

TEST_F(NormTiling, NormTiling2) {
  using namespace optiling;

  ge::Graph graph("NormTiling2");
  std::vector<int64_t> input{16, 5, 15003};
  std::vector<int64_t> output{16, 5, 15003};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x = op::Data("x");
  x.update_input_desc_x(tensor_input);
  x.update_output_desc_y(tensor_output);

  auto softmax_op = op::SoftmaxV2("SoftmaxV2_2");
  softmax_op.set_input_x(x);
  softmax_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x};
  std::vector<Operator> outputs{softmax_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string op_name = "AutoTiling";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  std::string compileInfo = R"({ "_ori_axis": [2], "_pattern": "Norm", "_common_info": [32, 8, 1, 12896, 12896], "_workspace_info": {"_workspace_type": [0], "_workspace_bytes": [4]}, "_vars": {"100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]}})";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo.c_str());
  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(iter->second(softmax_op, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 10);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "80 15003 8 7502 ");
}

TEST_F(NormTiling, NormTiling3) {
  using namespace optiling;

  ge::Graph graph("NormTiling3");
  std::vector<int64_t> input{16, 5, 15003};
  std::vector<int64_t> output{16, 5, 15003};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x = op::Data("x");
  x.update_input_desc_x(tensor_input);
  x.update_output_desc_y(tensor_output);

  auto softmax_op = op::SoftmaxV2("SoftmaxV2_3");
  softmax_op.set_input_x(x);
  softmax_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x};
  std::vector<Operator> outputs{softmax_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string op_name = "AutoTiling";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  std::string compileInfo = R"({ "_ori_axis": [1], "_pattern": "Norm", "_common_info": [32, 8, 1, 16336, 16360], "_workspace_info": {"_workspace_type": [1, 0, 0], "_workspace_bytes": [4, 4, 4]}, "_vars": {"2100900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"]}})";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo.c_str());
  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(iter->second(softmax_op, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "16 5 15003 7504 2 ");
}

TEST_F(NormTiling, NormTiling4) {
  using namespace optiling;

  ge::Graph graph("NormTiling4");
  std::vector<int64_t> input{31, 2400};
  std::vector<int64_t> output{31, 2400};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x = op::Data("x");
  x.update_input_desc_x(tensor_input);
  x.update_output_desc_y(tensor_output);

  auto softmax_op = op::SoftmaxV2("SoftmaxV2_4");
  softmax_op.set_input_x(x);
  softmax_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x};
  std::vector<Operator> outputs{softmax_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string op_name = "AutoTiling";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  std::string compileInfo = R"({ "_ori_axis": [0], "_pattern": "Norm", "_common_info": [32, 8, 1, 16336, 16360], "_workspace_info": {"_workspace_type": [1, 0, 0], "_workspace_bytes": [4, 4, 4]}, "_vars": {"1000500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]}})";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo.c_str());
  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(iter->second(softmax_op, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 30);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "31 2400 80 31 ");
}

TEST_F(NormTiling, NormTiling5) {
  using namespace optiling;

  ge::Graph graph("NormTiling5");
  std::vector<int64_t> input{1968, 3, 3};
  std::vector<int64_t> output{1968, 3, 3};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x = op::Data("x");
  x.update_input_desc_x(tensor_input);
  x.update_output_desc_y(tensor_output);

  auto softmax_op = op::SoftmaxV2("SoftmaxV2_5");
  softmax_op.set_input_x(x);
  softmax_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x};
  std::vector<Operator> outputs{softmax_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string op_name = "AutoTiling";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  std::string compileInfo = R"({ "_ori_axis": [0, 2], "_pattern": "Norm", "_common_info": [32, 8, 1, 16216, 16248], "_workspace_info": {"_workspace_type": [1, 0, 0], "_workspace_bytes": [4, 4, 4]}, "_vars": {"21001200": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"]}})";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo.c_str());
  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(iter->second(softmax_op, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1968 3 3 3 677 ");
}

TEST_F(NormTiling, NormTiling6) {
  using namespace optiling;

  ge::Graph graph("NormTiling5");
  std::vector<int64_t> input{1, 7, 543, 76};
  std::vector<int64_t> output{1, 7, 543, 76};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x = op::Data("x");
  x.update_input_desc_x(tensor_input);
  x.update_output_desc_y(tensor_output);

  auto softmax_op = op::SoftmaxV2("SoftmaxV2_6");
  softmax_op.set_input_x(x);
  softmax_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x};
  std::vector<Operator> outputs{softmax_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string op_name = "AutoTiling";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  std::string compileInfo = R"({ "_ori_axis": [3], "_pattern": "Norm", "_common_info": [32, 8, 1, 16216, 16248], "_workspace_info": {"_workspace_type": [1, 0, 0], "_workspace_bytes": [4, 4, 4]}, "_reduce_shape_known": true, "_const_shape_post": true, "_const_tiling_key": 10000400, "_block_dims": 32, "_vars": {"10000400": []}})";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo.c_str());
  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(iter->second(softmax_op, op_compile_info, runInfo));
}
