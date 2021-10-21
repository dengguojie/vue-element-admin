//
// Created by xukaiwei on 7/1/21.
//

#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#define private public
#include "register/op_tiling_registry.h"

#include "graph/graph.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "op_tiling/vector_tiling.h"

#include "reduce_ops.h"
#include "array_ops.h"
#include "test_common.h"


using namespace std;
using namespace ge;

class ReduceTilingV2 : public testing::Test {
protected:
   static void SetUpTestCase() {
     std::cout << "ReduceTilingV2 SetUp" << std::endl;
   }

   static void TearDownTestCase() {
     std::cout << "ReduceTilingV2 TearDown" << std::endl;
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

/* Test Case
 * **/
TEST_F(ReduceTilingV2, ReduceTiling1) {
  using namespace optiling;

  std::vector<int64_t> input{1};
  std::vector<int64_t> output{1};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto reduce_sum_d_op = op::ReduceSumD("ReduceSumD_1");
  reduce_sum_d_op.set_input_x(x1);
  reduce_sum_d_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{reduce_sum_d_op};
  ge::Graph graph("ReduceTiling1");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);


  std::string op_name = "AutoTiling";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  std::string compileInfo = R"({ "_ori_axis": [0], "_pattern": "CommReduce","push_status": 0,"_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5], "_ub_info": [16256], "_ub_info_rf": [16256], "_vars": {"-1000500": ["_dim_1_0", "_block_factor", "_ub_factor"]}})";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(iter->second(reduce_sum_d_op, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 1 1 ");
}

TEST_F(ReduceTilingV2, ReduceTiling2) {
  using namespace optiling;

  std::vector<int64_t> input{2, 39, 0};
  std::vector<int64_t> output{2, 39, 1};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto reduce_sum_d_op = op::ReduceSumD("ReduceSumD_2");
  reduce_sum_d_op.set_input_x(x1);
  reduce_sum_d_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{reduce_sum_d_op};
  ge::Graph graph("ReduceTiling2");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string op_name = "AutoTiling";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  std::string compileInfo = R"({ "_ori_axis": [2], "_pattern": "CommReduce", "push_status": 0, "_zero_ub_factor": 25600, "_vars": {"10": ["_dim_1", "_ub_factor"]}})";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(iter->second(reduce_sum_d_op, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "78 25600 ");
}

TEST_F(ReduceTilingV2, ReduceTiling3) {
  using namespace optiling;

  std::vector<int64_t> input{2, 39, 0};
  std::vector<int64_t> output{};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto reduce_sum_d_op = op::ReduceSumD("ReduceSumD_3");
  reduce_sum_d_op.set_input_x(x1);
  reduce_sum_d_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{reduce_sum_d_op};
  ge::Graph graph("ReduceTiling3");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string op_name = "AutoTiling";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  std::string compileInfo = R"({ "_ori_axis": [1], "_pattern": "CommReduce", "push_status": 0, "_zero_ub_factor": 32128, "_vars": {"110": ["_dim_2", "_ub_factor"]}})";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(iter->second(reduce_sum_d_op, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 128 ");
}

TEST_F(ReduceTilingV2, ReduceTiling4) {
  using namespace optiling;

  std::vector<int64_t> input{64, 64};
  std::vector<int64_t> output{1,64};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto reduce_sum_d_op = op::ReduceSumD("ReduceSumD_4");
  reduce_sum_d_op.set_input_x(x1);
  reduce_sum_d_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{reduce_sum_d_op};
  ge::Graph graph("ReduceTiling4");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string op_name = "AutoTiling";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  std::string compileInfo = R"({"_ori_axis": [0],"_pattern": "CommReduce", "push_status": 0, "_zero_ub_factor": 32512, "_common_info": [32,1,8,1,1], "_pattern_info": [1], "_ub_info":[32512], "_ub_info_rf": [32512], "_reduce_shape_known": true, "_const_shape_post": true, "_compile_pattern": 1, "_block_dims":{"1":32},
     "_atomic_flags":{"1": true},
     "_vars": {"1": []}})";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(iter->second(reduce_sum_d_op, op_compile_info, runInfo));
}

TEST_F(ReduceTilingV2, ReduceTiling5) {
  using namespace optiling;

  std::vector<int64_t> input{1};
  std::vector<int64_t> output{1};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto reduce_sum_d_op = op::ReduceSumD("ReduceSumD_5");
  reduce_sum_d_op.set_input_x(x1);
  reduce_sum_d_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{reduce_sum_d_op};
  ge::Graph graph("ReduceTiling5");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string op_name = "AutoTiling";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  std::string compileInfo = R"({ "_ori_axis": [0], "_pattern": "CommReduce","push_status": 0,"common_info": [32, 1, 8, 1, 1], "pattern_info": [20000], "ub_info": [16256], "_ub_info_rf": [16256], "_vars": {"-1000500": ["dim_1_0", "block_factor", "ub_factor"]}})";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(!iter->second(reduce_sum_d_op, op_compile_info, runInfo));
}

TEST_F(ReduceTilingV2, ReduceTiling6) {
  using namespace optiling;

  std::vector<int64_t> input{1};
  std::vector<int64_t> output{1};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto reduce_sum_d_op = op::ReduceSumD("ReduceSumD_6");
  reduce_sum_d_op.set_input_x(x1);
  reduce_sum_d_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{reduce_sum_d_op};
  ge::Graph graph("ReduceTiling6");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string op_name = "AutoTiling";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  std::string compileInfo = R"({ "axes_idx": 0, "_pattern": "CommReduce","push_status": 0,"common_info": [32, 1, 8, 1, 1], "pattern_info": [20000], "ub_info": [16256], "_ub_info_rf": [16256], "_vars": {"-1000500": ["dim_1_0", "block_factor", "ub_factor"]}})";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(!iter->second(reduce_sum_d_op, op_compile_info, runInfo));
}

// ReduceTiling7 const_tensor

// FineTuning tune0
TEST_F(ReduceTilingV2, ReduceTiling8) {
  using namespace optiling;

  std::vector<int64_t> input{10000, 9, 80};
  std::vector<int64_t> output{1, 9, 80};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT16);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto reduce_sum_d_op = op::ReduceSumD("ReduceSumD_8");
  reduce_sum_d_op.set_input_x(x1);
  reduce_sum_d_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{reduce_sum_d_op};
  ge::Graph graph("ReduceTiling8");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string op_name = "AutoTiling";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  std::string compileInfo = R"({"_ori_axis": [0],"_pattern": "CommReduce", "push_status": 0,
                               "_zero_ub_factor": 32512, "_common_info": [32,1,16,0,1],
                               "_pattern_info": [5,4,9], "_ub_info":[21632, 21376, 21632],
                               "_ub_info_rf": [21632,16000,21632],
                               "_pattern": "CommReduce",
                               "_vars": {"1": []}})";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(iter->second(reduce_sum_d_op, op_compile_info, runInfo));
}

// FineTuning tune1
TEST_F(ReduceTilingV2, ReduceTiling9) {
  using namespace optiling;

  std::vector<int64_t> input{16, 1, 8, 38, 1, 16, 16};
  std::vector<int64_t> output{1, 16};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto reduce_sum_d_op = op::ReduceSumD("ReduceSumD_9");
  reduce_sum_d_op.set_input_x(x1);
  reduce_sum_d_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{reduce_sum_d_op};
  ge::Graph graph("ReduceTiling9");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string op_name = "AutoTiling";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  std::string compileInfo = R"({"_ori_axis": [0,2,3,4,5],"_pattern": "CommReduce",
                               "_common_info": [32,1,8,1,1],
                               "_pattern_info": [5,4,9], "_ub_info":[32512, 32128, 16128],
                               "_ub_info_rf": [32512, 21376, 32512],
                               "_pattern": "CommReduce",
                               "_vars": {"1": []}})";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(iter->second(reduce_sum_d_op, op_compile_info, runInfo));
}

// for new interface
TEST_F(ReduceTilingV2, ReduceTiling10) {
  using namespace optiling;

  std::vector<int64_t> input{64, 64};
  std::vector<int64_t> output{1,64};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto reduce_sum_d_op = op::ReduceSumD("ReduceSumD_4");
  reduce_sum_d_op.set_input_x(x1);
  reduce_sum_d_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{reduce_sum_d_op};
  ge::Graph graph("ReduceTiling4");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string op_name = "AutoTiling";
  std::string compileInfo = R"({"_ori_axis": [0],"_pattern": "CommReduce", "push_status": 0, "_zero_ub_factor": 32512, "_common_info": [32,1,8,1,1], "_pattern_info": [1], "_ub_info":[32512], "_ub_info_rf": [32512], "_reduce_shape_known": true, "_const_shape_post": true, "_compile_pattern": 1, "_block_dims":{"1":32},
       "_atomic_flags":{"1": true},
       "_vars": {"1": []}})";

  // new interface
  nlohmann::json json_info = nlohmann::json::parse(compileInfo.c_str());
  optiling::utils::OpRunInfo runInfo;
  std::vector<std::vector<int64_t>> input_shapes{input,};
  optiling::OpInfo c_op_info(input_shapes, DT_FLOAT);
  ASSERT_TRUE(optiling::ReduceTiling("CustomOP", reduce_sum_d_op, json_info, runInfo, c_op_info));
}


TEST_F(ReduceTilingV2, ReduceTiling11) {
  using namespace optiling;

  std::vector<int64_t> input{1};
  std::vector<int64_t> output{1};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto reduce_sum_d_op = op::ReduceSumD("ReduceSumD_11");
  reduce_sum_d_op.set_input_x(x1);
  reduce_sum_d_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{reduce_sum_d_op};
  ge::Graph graph("ReduceTiling11");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string op_name = "AutoTiling";
  std::string compileInfo = R"({"_ori_axis": [0],
                                "_pattern": "CommReduce",
                                "_common_info": [32, 1, 8, 1, 1],
                                "_pattern_info": [5],
                                "_ub_info": [16256],
                                "_ub_info_rf": [16256],
                                "_vars": {"-1000500": ["_dim_1_0", "_block_factor", "_ub_factor"]}})";

  // new interface
  nlohmann::json json_info = nlohmann::json::parse(compileInfo.c_str());
  optiling::utils::OpRunInfo runInfo;
  std::vector<std::vector<int64_t>> input_shapes{};
  optiling::OpInfo c_op_info(input_shapes, DT_FLOAT);
  ASSERT_TRUE(optiling::ReduceTiling("CustomOP", reduce_sum_d_op, json_info, runInfo, c_op_info));
}

TEST_F(ReduceTilingV2, ReduceTiling12) {
  using namespace optiling;

  std::vector<int64_t> input{64, 64};
  std::vector<int64_t> output{1,64};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto reduce_sum_d_op = op::ReduceSumD("ReduceSumD_12");
  reduce_sum_d_op.set_input_x(x1);
  reduce_sum_d_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{reduce_sum_d_op};
  ge::Graph graph("ReduceTiling12");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string op_name = "AutoTiling";
  std::string compileInfo = R"({"_ori_axis": [-2],"_pattern": "CommReduce", "_zero_ub_factor": 32512, "_common_info": [32,1,8,1,1], "_pattern_info": [1], "_ub_info":[32512], "_ub_info_rf": [32512], "_reduce_shape_known": true, "_compile_pattern": 1, "_block_dims":{"1":32},
         "_atomic_flags":{"1": true},
         "_vars": {"1": []}})";

  // new interface
  nlohmann::json json_info = nlohmann::json::parse(compileInfo.c_str());
  optiling::utils::OpRunInfo runInfo;
  std::vector<std::vector<int64_t>> input_shapes{input,};
  optiling::OpInfo c_op_info(input_shapes, DT_FLOAT);
  ASSERT_TRUE(optiling::ReduceTiling("CustomOP", reduce_sum_d_op, json_info, runInfo, c_op_info));
}

static void ReduceSumCompute(std::vector<int64_t> inputA, std::vector<int64_t> inputB, std::vector<int32_t> axes,
                             std::vector<int64_t> output, ge::DataType dtypeA, ge::DataType dtypeB,
                             ge::DataType dtypeOutput, std::string compileInfo, bool isCustom, std::string caseName) {
  using namespace optiling;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputA));
  tensor_inputA.SetDataType(dtypeA);

  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputB));
  tensor_inputB.SetDataType(dtypeB);

  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(dtypeOutput);

  auto opParas = op::ReduceSum(caseName);
  TENSOR_INPUT(opParas, tensor_inputA, x);
  TENSOR_INPUT_CONST(opParas, tensor_inputB, axes, (const uint8_t*)axes.data(), axes.size() * 4);
  TENSOR_OUTPUT(opParas, tensor_output, y);

  nlohmann::json json_info = nlohmann::json::parse(compileInfo.c_str());
  optiling::utils::OpRunInfo runInfo;
  if (!isCustom) {
    ASSERT_TRUE(optiling::ReduceTiling("AutoTiling", opParas, json_info, runInfo));
  } else {
    std::vector<std::vector<int64_t>> input_shapes{inputA, inputB};
    optiling::OpInfo c_op_info(input_shapes, DT_FLOAT);
    ASSERT_TRUE(optiling::ReduceTiling("AutoTiling", opParas, json_info, runInfo, c_op_info));
  }
}

TEST_F(ReduceTilingV2, ReduceSumTiling1) {
  std::string caseName = "ReduceSumTiling1";
  std::string compileInfo = R"({"_pattern": "CommReduce", "_common_info": [32,1,8,1,1], "_pattern_info": [5, 4, 9], "_ub_info_rf": [32512, 21376, 32512], "_ub_info": [32512, 21376, 16128], "_idx_before_reduce": 0, "_vars": {"4293966796": ["_dim_1", "_block_factor", "_ub_factor"], "4293866796": ["_dim_1", "_block_factor", "_ub_factor"], "500": ["_dim_1", "_block_factor", "_ub_factor"], "100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "4294966896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294866896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1000400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294966396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1000900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1100900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1200900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"]}, "_normal_vars": {"4293966796": ["_dim_1", "_block_factor", "_ub_factor"], "4293866796": ["_dim_1", "_block_factor", "_ub_factor"], "500": ["_dim_1", "_block_factor", "_ub_factor"], "100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "4294966896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294866896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1000400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294966396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1000900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1100900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1200900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"]}, "_attr_vars": {"4293966796": [], "4293866796": [], "500": [], "100500": [], "2147483647": [], "4294966896": [], "4294866896": [], "1000400": [], "1100400": [], "4294966396": [], "4294866396": [], "4294766396": [], "4292866396": [], "4292766396": [], "1000900": [], "1100900": [], "1200900": []}, "_custom_vars": {"4293966796": [], "4293866796": [], "500": [], "100500": [], "2147483647": [], "4294966896": [], "4294866896": [], "1000400": [], "1100400": [], "4294966396": [], "4294866396": [], "4294766396": [], "4292866396": [], "4292766396": [], "1000900": [], "1100900": [], "1200900": []}})";
  std::vector<int64_t> inputA{32,256};
  std::vector<int64_t> inputB{1};
  std::vector<int32_t> axes{0};
  std::vector<int64_t> output{1,256};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  bool isCustom = true;
  ReduceSumCompute(inputA, inputB, axes, output, dtypeA, dtypeB, dtypeOutput, compileInfo, isCustom, caseName);
}

TEST_F(ReduceTilingV2, ReduceSumTiling2) {
  std::string caseName = "ReduceSumTiling2";
  std::string compileInfo = R"({"_pattern": "CommReduce", "_common_info": [32,1,8,1,1], "_pattern_info": [5, 4, 9], "_ub_info_rf": [32512, 21376, 32512], "_ub_info": [32512, 21376, 16128], "_idx_before_reduce": 0, "_vars": {"4293966796": ["_dim_1", "_block_factor", "_ub_factor"], "4293866796": ["_dim_1", "_block_factor", "_ub_factor"], "500": ["_dim_1", "_block_factor", "_ub_factor"], "100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "4294966896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294866896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1000400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294966396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1000900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1100900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1200900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"]}, "_normal_vars": {"4293966796": ["_dim_1", "_block_factor", "_ub_factor"], "4293866796": ["_dim_1", "_block_factor", "_ub_factor"], "500": ["_dim_1", "_block_factor", "_ub_factor"], "100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "4294966896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294866896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1000400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294966396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1000900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1100900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1200900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"]}, "_attr_vars": {"4293966796": [], "4293866796": [], "500": [], "100500": [], "2147483647": [], "4294966896": [], "4294866896": [], "1000400": [], "1100400": [], "4294966396": [], "4294866396": [], "4294766396": [], "4292866396": [], "4292766396": [], "1000900": [], "1100900": [], "1200900": []}, "_custom_vars": {"4293966796": [], "4293866796": [], "500": [], "100500": [], "2147483647": [], "4294966896": [], "4294866896": [], "1000400": [], "1100400": [], "4294966396": [], "4294866396": [], "4294766396": [], "4292866396": [], "4292766396": [], "1000900": [], "1100900": [], "1200900": []}})";
  std::vector<int64_t> inputA{32,256};
  std::vector<int64_t> inputB{1};
  std::vector<int32_t> axes{0};
  std::vector<int64_t> output{1,256};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  bool isCustom = false;
  ReduceSumCompute(inputA, inputB, axes, output, dtypeA, dtypeB, dtypeOutput, compileInfo, isCustom, caseName);
}
