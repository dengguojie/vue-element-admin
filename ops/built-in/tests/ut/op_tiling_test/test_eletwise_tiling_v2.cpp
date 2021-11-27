//
// Created by wangyu on 2021/6/18.
//

#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#define private public
#define private public
#include "register/op_tiling_registry.h"

#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"

#include "elewise_calculation_ops.h"
#include "nonlinear_fuc_ops.h"
#include "array_ops.h"
#include "op_tiling/vector_tiling.h"


using namespace std;
using namespace ge;

class EletwiseTilingV2 : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "EletwiseTilingV2 SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "EletwiseTilingV2 TearDown" << std::endl;
    }
};

static string to_string(const std::stringstream &tiling_data) {
    auto data = tiling_data.str();
    string result;
    int32_t tmp = 0;
    for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
        memcpy_s(&tmp, sizeof(&tmp), data.c_str() + i, sizeof(tmp));
        result += std::to_string(tmp);
        result += " ";
    }

    return result;
}

TEST_F(EletwiseTilingV2, Eletwise_tiling1) {
  // dynamic_op_add_267.static_op_add_269
  using namespace optiling;

  std::vector<int64_t> inputA{1, 5824};
  std::vector<int64_t> inputB{100, 1};
  std::vector<int64_t> output{100, 5824};

  TensorDesc tensor_inputA(ge::Shape(inputA), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_inputB(ge::Shape(inputB), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_inputA);
  x1.update_output_desc_y(tensor_inputA);
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(tensor_inputB);
  x2.update_output_desc_y(tensor_inputB);

  auto add_op = op::Add("Add_0");
  add_op.set_input_x1(x1).set_input_x2(x2);
  add_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1, x2};
  std::vector<Operator> outputs{add_op};
  ge::Graph graph("Eletwise_tiling1");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string compileInfo = R"({ "_pattern": "ElemWise", "_fusion_index": [[0], [1]], "push_status": 0, "_flag_info": [false, false, true, true, true, false, false], "_base_info": {"320": [32, 4, 21840, 10920], "000": [32, 4, 21840, 10920]}, "_elewise_vars": { "232000000": [10001, 20000, 30000], "0": [10100], "1": [10100, 20000, 30000], "2": [10100, 20000, 30001], "4": [10100, 20001, 30001] }, "_vars": { "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_1_0"], "1": ["_dim_1_0", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_1_0", "_block_factor_0", "_ub_factor_1"], "4": ["_dim_1_0", "_block_factor_1", "_ub_factor_1"] } })";

  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(optiling::EletwiseTiling(this->test_info_->name(), add_op, nlohmann::json::parse(compileInfo), runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 25);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "5824 4 2 ");
}

TEST_F(EletwiseTilingV2, Eletwise_tiling2) {
  // dynamic_op_exp_432.static_op_exp_433
  using namespace optiling;

  std::vector<int64_t> inputA{1, 33, 1089};
  std::vector<int64_t> output{1, 33, 1089};

  TensorDesc tensor_inputA(ge::Shape(inputA), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_inputA);
  x1.update_output_desc_y(tensor_inputA);
  auto exp_op = op::Exp("Exp_0");
  exp_op.set_input_x(x1);
  exp_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1,};
  std::vector<Operator> outputs{exp_op};
  ge::Graph graph("Eletwise_tiling2");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);




  std::string compileInfo = R"({ "_outs_uint1": false, "_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, false, true, false, false, false], "_base_info": {"100": [32, 4, 32768, 16384]}, "_elewise_vars": { "210000000": [ 10000, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";

  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(optiling::EletwiseTiling(this->test_info_->name(), exp_op, nlohmann::json::parse(compileInfo), runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "35937 1152 1152 ");
}

TEST_F(EletwiseTilingV2, Eletwise_tiling3) {
  // dynamic_op_cast_30.static_op_cast_30
  using namespace optiling;

  std::vector<int64_t> inputA{128, 128, 128, 128};
  TensorDesc tensor_inputA(ge::Shape(inputA), FORMAT_ND, DT_FLOAT);

  std::vector<int64_t> output{128, 128, 128, 128};
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT16);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_inputA);
  x1.update_output_desc_y(tensor_inputA);

  auto cast_op = op::Cast("Cast_0");
  cast_op.set_input_x(x1);
  cast_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1,};
  std::vector<Operator> outputs{cast_op};
  ge::Graph graph("Eletwise_tiling3");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string compileInfo = R"({ "_outs_uint1": false, "_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, false, true, false, false, false], "_base_info": {"100": [32, 4, 16384, 8192]}, "_elewise_vars": { "210000000": [ 10000, 20000, 30000 ], "210010000": [ 10000, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ], "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";


  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(optiling::EletwiseTiling(this->test_info_->name(), cast_op, nlohmann::json::parse(compileInfo), runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "268435456 8388608 8192 ");
}

TEST_F(EletwiseTilingV2, Eletwise_tiling_mul1) {
  // dynamic_op_fusedmuladd_267.static_op_fusedmuladd_269
  using namespace optiling;

  std::vector<int64_t> inputA{64,};
  std::vector<int64_t> inputB{64,};
  std::vector<int64_t> inputC{1,};
  std::vector<int64_t> output{64,};

  TensorDesc tensor_inputA(ge::Shape(inputA), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_inputB(ge::Shape(inputB), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_inputC(ge::Shape(inputC), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_inputA);
  x1.update_output_desc_y(tensor_inputA);
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(tensor_inputB);
  x2.update_output_desc_y(tensor_inputB);
  auto x3 = op::Data("x3");
  x2.update_input_desc_x(tensor_inputC);
  x2.update_output_desc_y(tensor_inputC);

  auto fusedmuladd_op = op::FusedMulAdd("fusedmuladd_0");
  fusedmuladd_op.set_input_x1(x1).set_input_x2(x2).set_input_x3(x3);
  fusedmuladd_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1, x2, x3};
  std::vector<Operator> outputs{fusedmuladd_op};
  ge::Graph graph("Eletwise_tiling1");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string compileInfo = R"({"_attr_vars": {"210000000": [], "210010000": [], "220000000": [] }, "_custom_vars": {"210000000": [], "210010000": [], "220000000": [] }, "_fusion_index": [[0]], "_pattern": "Broadcast", "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 43680, 21840], "200": [32, 2, 28656, 14320]}, "_elewise_vars": { "210000000": [ 10000, 20000, 30000 ], "210010000": [ 10000, 20000, 30000 ], "220000000": [ 10000, 10001, 10002, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ], "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ], "210010000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ], "220000000": [ "_dim_0_0", "_dim_0_1", "_dim_0_2", "_block_factor_0", "_ub_factor_0"] }, "_normal_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ], "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ], "210010000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ], "220000000": [ "_dim_0_0", "_dim_0_1", "_dim_0_2", "_block_factor_0", "_ub_factor_0"] }, "_outs_uint1": false })";

  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(optiling::EletwiseTiling(this->test_info_->name(), fusedmuladd_op, nlohmann::json::parse(compileInfo), runInfo));
}


TEST_F(EletwiseTilingV2, Eletwise_tiling4) {
  // dynamic_op_exp_432.static_op_exp_433
  using namespace optiling;

  std::vector<int64_t> inputA{1, 33, 1089};
  TensorDesc tensor_inputA(ge::Shape(inputA), FORMAT_ND, DT_UINT32);

  std::vector<int64_t> output{1, 33, 1089};
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_UINT32);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_inputA);
  x1.update_output_desc_y(tensor_inputA);


  auto exp_op = op::Exp("Exp_0");
  exp_op.set_input_x(x1);
  exp_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1,};
  std::vector<Operator> outputs{exp_op};
  ge::Graph graph("Eletwise_tiling4");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);




  std::string compileInfo = R"({ "_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, "2", true, true, false, false, false], "_base_info": {"101": [32, 4, 32768, 16384]}, "_elewise_vars": { "210000000": [ 10001, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";


  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(!optiling::EletwiseTiling(this->test_info_->name(), exp_op, nlohmann::json::parse(compileInfo), runInfo));
}

TEST_F(EletwiseTilingV2, Eletwise_tiling5) {
// dynamic_op_add_267.static_op_add_269
using namespace optiling;

std::vector<int64_t> inputA{1, 33, 1};
std::vector<int64_t> inputB{1, 33, 1089};
std::vector<int64_t> output{1, 33, 1089};

TensorDesc tensor_inputA(ge::Shape(inputA), FORMAT_ND, DT_FLOAT);
TensorDesc tensor_inputB(ge::Shape(inputB), FORMAT_ND, DT_FLOAT);
TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

auto x1 = op::Data("x1");
x1.update_input_desc_x(tensor_inputA);
x1.update_output_desc_y(tensor_inputA);
auto x2 = op::Data("x2");
x2.update_input_desc_x(tensor_inputB);
x2.update_output_desc_y(tensor_inputB);

auto add_op = op::Add("Add_0");
add_op.set_input_x1(x1).set_input_x2(x2);
add_op.update_output_desc_y(tensor_output);

std::vector<Operator> inputs{x1, x2};
std::vector<Operator> outputs{add_op};
ge::Graph graph("Eletwise_tiling5");
graph.SetInputs(inputs).SetOutputs(outputs);

ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);





std::string compileInfo = R"({ "_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"110": [32, 4, 32768, 16384]}, "_elewise_vars": { "210000000": [ 10000, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";

optiling::utils::OpRunInfo runInfo;

ASSERT_TRUE(!optiling::EletwiseTiling(this->test_info_->name(), add_op, nlohmann::json::parse(compileInfo), runInfo));
}

TEST_F(EletwiseTilingV2, Eletwise_tiling6) {
// dynamic_op_exp_432.static_op_exp_433
using namespace optiling;

std::vector<int64_t> inputA{1, 33, 1089};
TensorDesc tensor_inputA(ge::Shape(inputA), FORMAT_ND, DT_FLOAT);
std::vector<int64_t> output{1, 33, 1089};
TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

auto x1 = op::Data("x1");
x1.update_input_desc_x(tensor_inputA);
x1.update_output_desc_y(tensor_inputA);

auto exp_op = op::Exp("Exp_0");
exp_op.set_input_x(x1);
exp_op.update_output_desc_y(tensor_output);

std::vector<Operator> inputs{x1,};
std::vector<Operator> outputs{exp_op};
ge::Graph graph("Eletwise_tiling6");
graph.SetInputs(inputs).SetOutputs(outputs);

ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);





std::string compileInfo = R"({ "_outs_uint1": false, "_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, false, true, false, false, false], "_base_info": {"100": [32, 4, 32768, 16384]}, "_elewise_vars": { "2101000000": [ 10000, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";


optiling::utils::OpRunInfo runInfo;

ASSERT_TRUE(!optiling::EletwiseTiling(this->test_info_->name(), exp_op, nlohmann::json::parse(compileInfo), runInfo));
}

TEST_F(EletwiseTilingV2, Eletwise_tiling7) {
// dynamic_op_add_267.static_op_add_269
using namespace optiling;

std::vector<int64_t> inputA{1, 33, 1};
std::vector<int64_t> inputB{1, 33, 1089};
std::vector<int64_t> output{1, 33, 1089};

TensorDesc tensor_inputA(ge::Shape(inputA), FORMAT_ND, DT_FLOAT);
TensorDesc tensor_inputB(ge::Shape(inputB), FORMAT_ND, DT_FLOAT);
TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

auto x1 = op::Data("x1");
x1.update_input_desc_x(tensor_inputA);
x1.update_output_desc_y(tensor_inputA);
auto x2 = op::Data("x2");
x2.update_input_desc_x(tensor_inputB);
x2.update_output_desc_y(tensor_inputB);

auto add_op = op::Add("Add_0");
add_op.set_input_x1(x1).set_input_x2(x2);
add_op.update_output_desc_y(tensor_output);

std::vector<Operator> inputs{x1, x2};
std::vector<Operator> outputs{add_op};
ge::Graph graph("Eletwise_tiling7");
graph.SetInputs(inputs).SetOutputs(outputs);

ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);





std::string compileInfo = R"({ "_outs_uint1": false, "_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, false, true, false, false, false], "_base_info": {"100": [32, 4, 32768, 16384]}, "_elewise_vars": { "211000000": [ 10000, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";

optiling::utils::OpRunInfo runInfo;

ASSERT_TRUE(!optiling::EletwiseTiling(this->test_info_->name(), add_op, nlohmann::json::parse(compileInfo), runInfo));
}

TEST_F(EletwiseTilingV2, Eletwise_tiling8) {
// dynamic_op_add_267.static_op_add_269
using namespace optiling;

std::vector<int64_t> inputA{1, 33, 1};
std::vector<int64_t> inputB{1, 33, 1089};
std::vector<int64_t> output{1, 33, 1089};

TensorDesc tensor_inputA(ge::Shape(inputA), FORMAT_ND, DT_FLOAT);
TensorDesc tensor_inputB(ge::Shape(inputB), FORMAT_ND, DT_FLOAT);
TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

auto x1 = op::Data("x1");
x1.update_input_desc_x(tensor_inputA);
x1.update_output_desc_y(tensor_inputA);
auto x2 = op::Data("x2");
x2.update_input_desc_x(tensor_inputB);
x2.update_output_desc_y(tensor_inputB);

auto add_op = op::Add("Add_0");
add_op.set_input_x1(x1).set_input_x2(x2);
add_op.update_output_desc_y(tensor_output);

std::vector<Operator> inputs{x1, x2};
std::vector<Operator> outputs{add_op};
ge::Graph graph("Eletwise_tiling8");
graph.SetInputs(inputs).SetOutputs(outputs);

ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);





std::string compileInfo = R"({"_fusion_index":[1], "push_status": 0, "_pattern": "ElemWise", "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"120": [32, 4, 32768, 16384]}, "_elewise_vars": { "210000000": [ 10000, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";


optiling::utils::OpRunInfo runInfo;

ASSERT_TRUE(!optiling::EletwiseTiling(this->test_info_->name(), add_op, nlohmann::json::parse(compileInfo), runInfo));
}

TEST_F(EletwiseTilingV2, Eletwise_tiling9) {
// dynamic_op_add_267.static_op_add_269
using namespace optiling;

std::vector<int64_t> inputA{1, 33, 1};
std::vector<int64_t> inputB{1, 33, 1089};
std::vector<int64_t> output{1, 33, 1089};

TensorDesc tensor_inputA(ge::Shape(inputA), FORMAT_ND, DT_FLOAT);
TensorDesc tensor_inputB(ge::Shape(inputB), FORMAT_ND, DT_FLOAT);
TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

auto x1 = op::Data("x1");
x1.update_input_desc_x(tensor_inputA);
x1.update_output_desc_y(tensor_inputA);
auto x2 = op::Data("x2");
x2.update_input_desc_x(tensor_inputB);
x2.update_output_desc_y(tensor_inputB);

auto add_op = op::Add("Add_0");
add_op.set_input_x1(x1).set_input_x2(x2);
add_op.update_output_desc_y(tensor_output);

std::vector<Operator> inputs{x1, x2};
std::vector<Operator> outputs{add_op};
ge::Graph graph("Eletwise_tiling9");
graph.SetInputs(inputs).SetOutputs(outputs);

ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);





std::string compileInfo = R"({"_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, true, true, true, false, false, false], "_base_info": {"sdf": [32, 4, 32768, 16384]}, "_elewise_vars": { "210000000": [ 10000, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";


optiling::utils::OpRunInfo runInfo;

ASSERT_TRUE(!optiling::EletwiseTiling(this->test_info_->name(), add_op, nlohmann::json::parse(compileInfo), runInfo));
}

TEST_F(EletwiseTilingV2, Eletwise_tiling10) {
// dynamic_op_add_267.static_op_add_269
using namespace optiling;

std::vector<int64_t> inputA{64,64};
std::vector<int64_t> inputB{64,64};
std::vector<int64_t> output{64,64};

TensorDesc tensor_inputA(ge::Shape(inputA), FORMAT_ND, DT_FLOAT);
TensorDesc tensor_inputB(ge::Shape(inputB), FORMAT_ND, DT_FLOAT);
TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

auto x1 = op::Data("x1");
x1.update_input_desc_x(tensor_inputA);
x1.update_output_desc_y(tensor_inputA);
auto x2 = op::Data("x2");
x2.update_input_desc_x(tensor_inputB);
x2.update_output_desc_y(tensor_inputB);

auto add_op = op::Add("Add_0");
add_op.set_input_x1(x1).set_input_x2(x2);
add_op.update_output_desc_y(tensor_output);

std::vector<Operator> inputs{x1, x2};
std::vector<Operator> outputs{add_op};
ge::Graph graph("Eletwise_tiling10");
graph.SetInputs(inputs).SetOutputs(outputs);

ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);





std::string compileInfo = R"({"_fusion_index": [[0, 1]],"_const_shapes":[[64,64]],"_const_block_dims":[32],"_pattern": "ElemWise","_flag_info": [false, true, true, true, false, false, false],"_vars": { "100000000": [] },"push_status": 0})";


optiling::utils::OpRunInfo runInfo;

ASSERT_TRUE(optiling::EletwiseTiling(this->test_info_->name(), add_op, nlohmann::json::parse(compileInfo), runInfo));
}

TEST_F(EletwiseTilingV2, Eletwise_tiling11) {
  // dynamic_op_add_267.static_op_add_269
  using namespace optiling;

  std::vector<int64_t> inputA{1, 33, 0};
  std::vector<int64_t> inputB{1, 33, 0};
  std::vector<int64_t> output{1, 33, 0};

  TensorDesc tensor_inputA(ge::Shape(inputA), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_inputB(ge::Shape(inputB), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_inputA);
  x1.update_output_desc_y(tensor_inputA);
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(tensor_inputB);
  x2.update_output_desc_y(tensor_inputB);

  auto add_op = op::Add("Add_0");
  add_op.set_input_x1(x1).set_input_x2(x2);
  add_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1, x2};
  std::vector<Operator> outputs{add_op};
  ge::Graph graph("Eletwise_tiling11");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string compileInfo = R"({ "push_status": 0, "_pattern": "ElemWise", "_flag_info": [false, false, false, true, false, false, false], "_base_info": {"100": [32, 4, 32768, 16384]}, "_elewise_vars": { "210000000": [ 10000, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";

  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(optiling::EletwiseTiling(this->test_info_->name(), add_op, nlohmann::json::parse(compileInfo), runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "");
}

TEST_F(EletwiseTilingV2, Eletwise_tiling12) {
// dynamic_op_add_267.static_op_add_269
using namespace optiling;

std::vector<int64_t> inputA{6643712,};
std::vector<int64_t> inputB{6643712,};
std::vector<int64_t> output{6643712,};

TensorDesc tensor_inputA(ge::Shape(inputA), FORMAT_ND, DT_FLOAT16);
TensorDesc tensor_inputB(ge::Shape(inputB), FORMAT_ND, DT_FLOAT16);
TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT16);

auto x1 = op::Data("x1");
x1.update_input_desc_x(tensor_inputA);
x1.update_output_desc_y(tensor_inputA);
auto x2 = op::Data("x2");
x2.update_input_desc_x(tensor_inputB);
x2.update_output_desc_y(tensor_inputB);

auto add_op = op::Add("Add_0");
add_op.set_input_x1(x1).set_input_x2(x2);
add_op.update_output_desc_y(tensor_output);

std::vector<Operator> inputs{x1, x2};
std::vector<Operator> outputs{add_op};
ge::Graph graph("Eletwise_tiling12");
graph.SetInputs(inputs).SetOutputs(outputs);

ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);





std::string compileInfo = R"({ "_outs_uint1": false, "_flag_info": [true],"_base_info": {"000": [32, 2, 43680, 21840]}, "_broadcast_axis": [false], "_fusion_index": [[0], [1], [2]], "_pattern": "ElemWise", "push_status": 0})";


optiling::utils::OpRunInfo runInfo;

ASSERT_TRUE(optiling::EletwiseTiling(this->test_info_->name(), add_op, nlohmann::json::parse(compileInfo), runInfo));
}


TEST_F(EletwiseTilingV2, Eletwise_tiling13) {
// dynamic_op_add_267.static_op_add_269
using namespace optiling;

std::vector<int64_t> inputA{35, 45, 223};
std::vector<int64_t> inputB{45, 223};
std::vector<int64_t> output{35, 45, 223};

TensorDesc tensor_inputA(ge::Shape(inputA), FORMAT_ND, DT_FLOAT16);
TensorDesc tensor_inputB(ge::Shape(inputB), FORMAT_ND, DT_FLOAT16);
TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT16);

auto x1 = op::Data("x1");
x1.update_input_desc_x(tensor_inputA);
x1.update_output_desc_y(tensor_inputA);
auto x2 = op::Data("x2");
x2.update_input_desc_x(tensor_inputB);
x2.update_output_desc_y(tensor_inputB);

auto add_op = op::Add("Add_0");
add_op.set_input_x1(x1).set_input_x2(x2);
add_op.update_output_desc_y(tensor_output);

std::vector<Operator> inputs{x1, x2};
std::vector<Operator> outputs{add_op};
ge::Graph graph("Eletwise_tiling13");
graph.SetInputs(inputs).SetOutputs(outputs);

ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);





std::string compileInfo = R"({ "push_status": 0, "_pattern": "Broadcast", "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"210": [32, 4, 21832, 10912]}, "_elewise_vars": { "221000001": [10000, 10001, 10100, 20000, 30000], "221000002": [10000, 10001, 10100, 20000, 30001] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";


optiling::utils::OpRunInfo runInfo;

ASSERT_TRUE(optiling::EletwiseTiling(this->test_info_->name(), add_op, nlohmann::json::parse(compileInfo), runInfo));
}

namespace optiling {
    class VarAttrHelper {
    public:
        static void InitTeOpVarAttr(ge::OpDescPtr &op_desc, optiling::TeOpVarAttrArgs &attr);
    };
}

TEST_F(EletwiseTilingV2, Eletwise_tiling17) {
// dynamic_op_add_267.static_op_add_269
using namespace optiling;

std::vector<int64_t> inputA{28, 1, 35, 45, 223};
std::vector<int64_t> inputB{28, 5, 35, 1, 1};
std::vector<int64_t> output{28, 5, 35, 45, 223};

TensorDesc tensor_inputA(ge::Shape(inputA), FORMAT_ND, DT_FLOAT);
TensorDesc tensor_inputB(ge::Shape(inputB), FORMAT_ND, DT_FLOAT);
TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

auto x1 = op::Data("x1");
x1.update_input_desc_x(tensor_inputA);
x1.update_output_desc_y(tensor_inputA);
auto x2 = op::Data("x2");
x2.update_input_desc_x(tensor_inputB);
x2.update_output_desc_y(tensor_inputB);

auto add_op = op::Add("Add_0");
add_op.set_input_x1(x1).set_input_x2(x2);
add_op.update_output_desc_y(tensor_output);

std::vector<Operator> inputs{x1, x2};
std::vector<Operator> outputs{add_op};
ge::Graph graph("Eletwise_tiling17");
graph.SetInputs(inputs).SetOutputs(outputs);

ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);





std::string compileInfo = R"({"_fusion_index": [[0], [1], [2], [3], [4]], "_pattern": "Broadcast", "_outs_uint1": false, "_flag_info": [false, false, true, true, true, false, true], "_base_info": {"100": [32, 2, 43680, 21840], "120": [32, 2, 28656, 14320], "121": [32, 2, 30704, 15344], "210": [32, 2, 30704, 15344], "320": [32, 2, 42320, 21152], "230": [32, 2, 42320, 21152], "000": [32, 2, 30704, 15344], "999": [32, 2, 30704, 15344]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212000004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212010004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "212100005": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_1"], "212100006": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_2"], "212100009": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_2", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_1", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_2"], "4": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_3"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_4"], "7": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_1"], "8": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_3"], "10": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_4"], "13": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_2", "_ub_factor_2"], "14": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_2", "_ub_factor_3"], "15": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_2", "_ub_factor_4"], "19": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_3", "_ub_factor_3"], "20": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_3", "_ub_factor_4"], "25": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_4", "_ub_factor_4"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_1"], "299900003": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_2"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_3"], "299900006": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_1", "_ub_factor_1"], "299900007": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_1", "_ub_factor_2"], "299900008": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_1", "_ub_factor_3"], "299900011": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_2", "_ub_factor_2"], "299900012": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_2", "_ub_factor_3"], "299900016": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_3", "_ub_factor_3"]}, "_normal_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212000004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212010004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "212100005": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_1"], "212100006": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_2"], "212100009": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_2", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_1", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_2"], "4": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_3"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_4"], "7": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_1"], "8": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_3"], "10": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_4"], "13": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_2", "_ub_factor_2"], "14": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_2", "_ub_factor_3"], "15": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_2", "_ub_factor_4"], "19": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_3", "_ub_factor_3"], "20": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_3", "_ub_factor_4"], "25": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_4", "_ub_factor_4"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_1"], "299900003": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_2"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_3"], "299900006": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_1", "_ub_factor_1"], "299900007": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_1", "_ub_factor_2"], "299900008": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_1", "_ub_factor_3"], "299900011": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_2", "_ub_factor_2"], "299900012": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_2", "_ub_factor_3"], "299900016": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_3", "_ub_factor_3"]}, "_attr_vars": {"210000000": [], "210010000": [], "212000000": [], "212000001": [], "212000002": [], "212010002": [], "212000004": [], "212010004": [], "212100000": [], "212100001": [], "212100002": [], "212100003": [], "212100005": [], "212100006": [], "212100009": [], "221000000": [], "221000001": [], "221000002": [], "221000004": [], "232000000": [], "223000000": [], "0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "7": [], "8": [], "9": [], "10": [], "13": [], "14": [], "15": [], "19": [], "20": [], "25": [], "299900000": [], "299900001": [], "299900002": [], "299900003": [], "299900004": [], "299900006": [], "299900007": [], "299900008": [], "299900011": [], "299900012": [], "299900016": []}, "_custom_vars": {"210000000": [], "210010000": [], "212000000": [], "212000001": [], "212000002": [], "212010002": [], "212000004": [], "212010004": [], "212100000": [], "212100001": [], "212100002": [], "212100003": [], "212100005": [], "212100006": [], "212100009": [], "221000000": [], "221000001": [], "221000002": [], "221000004": [], "232000000": [], "223000000": [], "0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "7": [], "8": [], "9": [], "10": [], "13": [], "14": [], "15": [], "19": [], "20": [], "25": [], "299900000": [], "299900001": [], "299900002": [], "299900003": [], "299900004": [], "299900006": [], "299900007": [], "299900008": [], "299900011": [], "299900012": [], "299900016": []}, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "212000000": [10000, 10100, 10101], "212000001": [10000, 10100, 10101, 20000, 30000], "212000002": [10000, 10100, 10101, 20000, 30001], "212010002": [10000, 10100, 10101, 20000, 30001], "212000004": [10000, 10100, 10101, 20001, 30001], "212010004": [10000, 10100, 10101, 20001, 30001], "212100000": [10000, 10100, 10101, 10200], "212100001": [10000, 10100, 10101, 10200, 20000, 30000], "212100002": [10000, 10100, 10101, 10200, 20000, 30001], "212100003": [10000, 10100, 10101, 10200, 20000, 30002], "212100005": [10000, 10100, 10101, 10200, 20001, 30001], "212100006": [10000, 10100, 10101, 10200, 20001, 30002], "212100009": [10000, 10100, 10101, 10200, 20002, 30002], "221000000": [10000, 10001, 10100], "221000001": [10000, 10001, 10100, 20000, 30000], "221000002": [10000, 10001, 10100, 20000, 30001], "221000004": [10000, 10001, 10100, 20001, 30001], "232000000": [10001, 20000, 30000], "223000000": [10000, 20000, 30000], "0": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401], "1": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30000], "2": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30001], "3": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30002], "4": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30003], "5": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30004], "7": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20001, 30001], "8": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20001, 30002], "9": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20001, 30003], "10": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20001, 30004], "13": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20002, 30002], "14": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20002, 30003], "15": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20002, 30004], "19": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20003, 30003], "20": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20003, 30004], "25": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20004, 30004], "299900000": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301], "299900001": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30000], "299900002": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30001], "299900003": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30002], "299900004": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30003], "299900006": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20001, 30001], "299900007": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20001, 30002], "299900008": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20001, 30003], "299900011": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20002, 30002], "299900012": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20002, 30003], "299900016": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20003, 30003]}})";


optiling::utils::OpRunInfo runInfo;

ASSERT_TRUE(optiling::EletwiseTiling(this->test_info_->name(), add_op, nlohmann::json::parse(compileInfo), runInfo));
EXPECT_EQ(runInfo.GetBlockDim(), 28);
EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "28 28 1 5 35 35 10035 1 5 3 ");
}

TEST_F(EletwiseTilingV2, Eletwise_tiling18) {
// dynamic_op_add_267.static_op_add_269
using namespace optiling;

std::vector<int64_t> inputA{1, 1, 1, 112, 22};
std::vector<int64_t> inputB{32, 5, 25, 1, 22};
std::vector<int64_t> output{32, 5, 25, 112, 22};

TensorDesc tensor_inputA(ge::Shape(inputA), FORMAT_ND, DT_FLOAT16);
TensorDesc tensor_inputB(ge::Shape(inputB), FORMAT_ND, DT_FLOAT16);
TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT16);

auto x1 = op::Data("x1");
x1.update_input_desc_x(tensor_inputA);
x1.update_output_desc_y(tensor_inputA);
auto x2 = op::Data("x2");
x2.update_input_desc_x(tensor_inputB);
x2.update_output_desc_y(tensor_inputB);

auto add_op = op::Add("Add_0");
add_op.set_input_x1(x1).set_input_x2(x2);
add_op.update_output_desc_y(tensor_output);

std::vector<Operator> inputs{x1, x2};
std::vector<Operator> outputs{add_op};
ge::Graph graph("Eletwise_tiling18");
graph.SetInputs(inputs).SetOutputs(outputs);

ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);





std::string compileInfo = R"({"_fusion_index": [[0], [1], [2], [3], [4]], "_pattern": "Broadcast", "_outs_uint1": false, "_flag_info": [false, false, true, true, true, false, true], "_base_info": {"100": [32, 2, 43680, 21840], "120": [32, 2, 28656, 14320], "121": [32, 2, 30704, 15344], "210": [32, 2, 30704, 15344], "320": [32, 2, 42320, 21152], "230": [32, 2, 42320, 21152], "000": [32, 2, 30704, 15344], "999": [32, 2, 30704, 15344]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212000004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212010004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "212100005": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_1"], "212100006": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_2"], "212100009": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_2", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_1", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_2"], "4": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_3"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_4"], "7": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_1"], "8": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_3"], "10": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_4"], "13": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_2", "_ub_factor_2"], "14": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_2", "_ub_factor_3"], "15": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_2", "_ub_factor_4"], "19": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_3", "_ub_factor_3"], "20": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_3", "_ub_factor_4"], "25": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_4", "_ub_factor_4"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_1"], "299900003": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_2"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_3"], "299900006": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_1", "_ub_factor_1"], "299900007": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_1", "_ub_factor_2"], "299900008": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_1", "_ub_factor_3"], "299900011": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_2", "_ub_factor_2"], "299900012": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_2", "_ub_factor_3"], "299900016": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_3", "_ub_factor_3"]}, "_normal_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212000004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212010004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "212100005": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_1"], "212100006": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_2"], "212100009": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_2", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_1", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_2"], "4": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_3"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_4"], "7": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_1"], "8": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_3"], "10": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_4"], "13": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_2", "_ub_factor_2"], "14": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_2", "_ub_factor_3"], "15": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_2", "_ub_factor_4"], "19": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_3", "_ub_factor_3"], "20": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_3", "_ub_factor_4"], "25": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_4", "_ub_factor_4"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_1"], "299900003": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_2"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_3"], "299900006": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_1", "_ub_factor_1"], "299900007": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_1", "_ub_factor_2"], "299900008": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_1", "_ub_factor_3"], "299900011": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_2", "_ub_factor_2"], "299900012": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_2", "_ub_factor_3"], "299900016": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_3", "_ub_factor_3"]}, "_attr_vars": {"210000000": [], "210010000": [], "212000000": [], "212000001": [], "212000002": [], "212010002": [], "212000004": [], "212010004": [], "212100000": [], "212100001": [], "212100002": [], "212100003": [], "212100005": [], "212100006": [], "212100009": [], "221000000": [], "221000001": [], "221000002": [], "221000004": [], "232000000": [], "223000000": [], "0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "7": [], "8": [], "9": [], "10": [], "13": [], "14": [], "15": [], "19": [], "20": [], "25": [], "299900000": [], "299900001": [], "299900002": [], "299900003": [], "299900004": [], "299900006": [], "299900007": [], "299900008": [], "299900011": [], "299900012": [], "299900016": []}, "_custom_vars": {"210000000": [], "210010000": [], "212000000": [], "212000001": [], "212000002": [], "212010002": [], "212000004": [], "212010004": [], "212100000": [], "212100001": [], "212100002": [], "212100003": [], "212100005": [], "212100006": [], "212100009": [], "221000000": [], "221000001": [], "221000002": [], "221000004": [], "232000000": [], "223000000": [], "0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "7": [], "8": [], "9": [], "10": [], "13": [], "14": [], "15": [], "19": [], "20": [], "25": [], "299900000": [], "299900001": [], "299900002": [], "299900003": [], "299900004": [], "299900006": [], "299900007": [], "299900008": [], "299900011": [], "299900012": [], "299900016": []}, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "212000000": [10000, 10100, 10101], "212000001": [10000, 10100, 10101, 20000, 30000], "212000002": [10000, 10100, 10101, 20000, 30001], "212010002": [10000, 10100, 10101, 20000, 30001], "212000004": [10000, 10100, 10101, 20001, 30001], "212010004": [10000, 10100, 10101, 20001, 30001], "212100000": [10000, 10100, 10101, 10200], "212100001": [10000, 10100, 10101, 10200, 20000, 30000], "212100002": [10000, 10100, 10101, 10200, 20000, 30001], "212100003": [10000, 10100, 10101, 10200, 20000, 30002], "212100005": [10000, 10100, 10101, 10200, 20001, 30001], "212100006": [10000, 10100, 10101, 10200, 20001, 30002], "212100009": [10000, 10100, 10101, 10200, 20002, 30002], "221000000": [10000, 10001, 10100], "221000001": [10000, 10001, 10100, 20000, 30000], "221000002": [10000, 10001, 10100, 20000, 30001], "221000004": [10000, 10001, 10100, 20001, 30001], "232000000": [10001, 20000, 30000], "223000000": [10000, 20000, 30000], "0": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401], "1": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30000], "2": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30001], "3": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30002], "4": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30003], "5": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30004], "7": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20001, 30001], "8": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20001, 30002], "9": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20001, 30003], "10": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20001, 30004], "13": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20002, 30002], "14": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20002, 30003], "15": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20002, 30004], "19": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20003, 30003], "20": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20003, 30004], "25": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20004, 30004], "299900000": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301], "299900001": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30000], "299900002": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30001], "299900003": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30002], "299900004": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30003], "299900006": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20001, 30001], "299900007": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20001, 30002], "299900008": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20001, 30003], "299900011": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20002, 30002], "299900012": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20002, 30003], "299900016": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20003, 30003]}})";


optiling::utils::OpRunInfo runInfo;

ASSERT_TRUE(optiling::EletwiseTiling(this->test_info_->name(), add_op, nlohmann::json::parse(compileInfo), runInfo));
EXPECT_EQ(runInfo.GetBlockDim(), 32);
EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 4000 14 1 8 1 22 22 125 12 ");
}

TEST_F(EletwiseTilingV2, Eletwise_tiling19) {
  // dynamic_op_add_267.static_op_add_269
  using namespace optiling;

  std::vector<int64_t> inputA{1, 1, 5824};
  std::vector<int64_t> inputB{32, 100, 1};
  std::vector<int64_t> output{32, 100, 5824};

  TensorDesc tensor_inputA(ge::Shape(inputA), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_inputB(ge::Shape(inputB), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_inputA);
  x1.update_output_desc_y(tensor_inputA);
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(tensor_inputB);
  x2.update_output_desc_y(tensor_inputB);

  auto add_op = op::Add("Add_0");
  add_op.set_input_x1(x1).set_input_x2(x2);
  add_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1, x2};
  std::vector<Operator> outputs{add_op};
  ge::Graph graph("Eletwise_tiling1");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);




  std::string compileInfo = R"({"_fusion_index": [[0], [1], [2]], "_pattern": "Broadcast", "_outs_uint1": false, "_soc_version": "Ascend920", "_flag_info": [false, false, true, true, true, false, true], "_base_info": {"100": [48, 2, 32768, 16384], "120": [48, 2, 24576, 12288], "121": [48, 2, 19648, 9824], "210": [48, 2, 19648, 9824], "320": [48, 2, 32752, 16368], "230": [48, 2, 32752, 16368], "000": [48, 2, 19648, 9824], "999": [48, 2, 19648, 9824]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212000004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212010004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "212100005": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_1"], "212100006": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_2"], "212100009": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_2", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_1", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_2", "_ub_factor_2"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"]}, "_normal_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212000004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212010004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "212100005": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_1"], "212100006": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_2"], "212100009": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_2", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_1", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_2", "_ub_factor_2"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"]}, "_attr_vars": {"210000000": [], "210010000": [], "212000000": [], "212000001": [], "212000002": [], "212010002": [], "212000004": [], "212010004": [], "212100000": [], "212100001": [], "212100002": [], "212100003": [], "212100005": [], "212100006": [], "212100009": [], "221000000": [], "221000001": [], "221000002": [], "221000004": [], "232000000": [], "223000000": [], "0": [], "1": [], "2": [], "3": [], "5": [], "6": [], "9": [], "299900000": [], "299900001": [], "299900002": [], "299900004": []}, "_custom_vars": {"210000000": [], "210010000": [], "212000000": [], "212000001": [], "212000002": [], "212010002": [], "212000004": [], "212010004": [], "212100000": [], "212100001": [], "212100002": [], "212100003": [], "212100005": [], "212100006": [], "212100009": [], "221000000": [], "221000001": [], "221000002": [], "221000004": [], "232000000": [], "223000000": [], "0": [], "1": [], "2": [], "3": [], "5": [], "6": [], "9": [], "299900000": [], "299900001": [], "299900002": [], "299900004": []}, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "212000000": [10000, 10100, 10101], "212000001": [10000, 10100, 10101, 20000, 30000], "212000002": [10000, 10100, 10101, 20000, 30001], "212010002": [10000, 10100, 10101, 20000, 30001], "212000004": [10000, 10100, 10101, 20001, 30001], "212010004": [10000, 10100, 10101, 20001, 30001], "212100000": [10000, 10100, 10101, 10200], "212100001": [10000, 10100, 10101, 10200, 20000, 30000], "212100002": [10000, 10100, 10101, 10200, 20000, 30001], "212100003": [10000, 10100, 10101, 10200, 20000, 30002], "212100005": [10000, 10100, 10101, 10200, 20001, 30001], "212100006": [10000, 10100, 10101, 10200, 20001, 30002], "212100009": [10000, 10100, 10101, 10200, 20002, 30002], "221000000": [10000, 10001, 10100], "221000001": [10000, 10001, 10100, 20000, 30000], "221000002": [10000, 10001, 10100, 20000, 30001], "221000004": [10000, 10001, 10100, 20001, 30001], "232000000": [10001, 20000, 30000], "223000000": [10000, 20000, 30000], "0": [10000, 10001, 10100, 10101, 10200, 10201], "1": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30000], "2": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30001], "3": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30002], "5": [10000, 10001, 10100, 10101, 10200, 10201, 20001, 30001], "6": [10000, 10001, 10100, 10101, 10200, 10201, 20001, 30002], "9": [10000, 10001, 10100, 10101, 10200, 10201, 20002, 30002], "299900000": [10000, 10001, 10100, 10101], "299900001": [10000, 10001, 10100, 10101, 20000, 30000], "299900002": [10000, 10001, 10100, 10101, 20000, 30001], "299900004": [10000, 10001, 10100, 10101, 20001, 30001]}})";

  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(optiling::EletwiseTiling(this->test_info_->name(), add_op, nlohmann::json::parse(compileInfo), runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 48);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 3200 5824 1 67 3 ");
}

TEST_F(EletwiseTilingV2, Eletwise_tiling_mul3) {
  // dynamic_op_fusedmuladd_267.static_op_fusedmuladd_269
  using namespace optiling;

  std::vector<int64_t> inputA{1, 1, 5824};
  std::vector<int64_t> inputB{32, 100, 1};
  std::vector<int64_t> inputC{32, 100, 1};
  std::vector<int64_t> output{32, 100, 5824};

  TensorDesc tensor_inputA(ge::Shape(inputA), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_inputB(ge::Shape(inputB), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_inputC(ge::Shape(inputC), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_inputA);
  x1.update_output_desc_y(tensor_inputA);
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(tensor_inputB);
  x2.update_output_desc_y(tensor_inputB);
  auto x3 = op::Data("x3");
  x2.update_input_desc_x(tensor_inputC);
  x2.update_output_desc_y(tensor_inputC);

  auto fusedmuladd_op = op::FusedMulAdd("fusedmuladd_0");
  fusedmuladd_op.set_input_x1(x1).set_input_x2(x2).set_input_x3(x3);
  fusedmuladd_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1, x2, x3};
  std::vector<Operator> outputs{fusedmuladd_op};
  ge::Graph graph("Eletwise_tiling_mul3");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);




  std::string compileInfo = R"({"_fusion_index": [[0], [1], [2]], "_pattern": "Broadcast", "_outs_uint1": false, "_soc_version": "Ascend920", "_flag_info": [false, false, true, true, false, false, true], "_base_info": {"100": [48, 2, 32768, 16384], "120": [48, 2, 16384, 8192], "121": [48, 2, 16384, 8192], "210": [48, 2, 16384, 8192], "200": [48, 2, 24576, 12288], "000": [48, 2, 16384, 8192], "999": [48, 2, 16384, 8192]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212000004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212010004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "212100005": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_1"], "212100006": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_2"], "212100009": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_2", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_1", "_ub_factor_1"], "220000000": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_2", "_ub_factor_2"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"]}, "_normal_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212000004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212010004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "212100005": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_1"], "212100006": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_2"], "212100009": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_2", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_1", "_ub_factor_1"], "220000000": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_2", "_ub_factor_2"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"]}, "_attr_vars": {"210000000": [], "210010000": [], "212000000": [], "212000001": [], "212000002": [], "212010002": [], "212000004": [], "212010004": [], "212100000": [], "212100001": [], "212100002": [], "212100003": [], "212100005": [], "212100006": [], "212100009": [], "221000000": [], "221000001": [], "221000002": [], "221000004": [], "220000000": [], "0": [], "1": [], "2": [], "3": [], "5": [], "6": [], "9": [], "299900000": [], "299900001": [], "299900002": [], "299900004": []}, "_custom_vars": {"210000000": [], "210010000": [], "212000000": [], "212000001": [], "212000002": [], "212010002": [], "212000004": [], "212010004": [], "212100000": [], "212100001": [], "212100002": [], "212100003": [], "212100005": [], "212100006": [], "212100009": [], "221000000": [], "221000001": [], "221000002": [], "221000004": [], "220000000": [], "0": [], "1": [], "2": [], "3": [], "5": [], "6": [], "9": [], "299900000": [], "299900001": [], "299900002": [], "299900004": []}, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "212000000": [10000, 10100, 10101], "212000001": [10000, 10100, 10101, 20000, 30000], "212000002": [10000, 10100, 10101, 20000, 30001], "212010002": [10000, 10100, 10101, 20000, 30001], "212000004": [10000, 10100, 10101, 20001, 30001], "212010004": [10000, 10100, 10101, 20001, 30001], "212100000": [10000, 10100, 10101, 10200], "212100001": [10000, 10100, 10101, 10200, 20000, 30000], "212100002": [10000, 10100, 10101, 10200, 20000, 30001], "212100003": [10000, 10100, 10101, 10200, 20000, 30002], "212100005": [10000, 10100, 10101, 10200, 20001, 30001], "212100006": [10000, 10100, 10101, 10200, 20001, 30002], "212100009": [10000, 10100, 10101, 10200, 20002, 30002], "221000000": [10000, 10001, 10100], "221000001": [10000, 10001, 10100, 20000, 30000], "221000002": [10000, 10001, 10100, 20000, 30001], "221000004": [10000, 10001, 10100, 20001, 30001], "220000000": [10000, 10001, 20000, 30000], "0": [10000, 10001, 10100, 10101, 10200, 10201], "1": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30000], "2": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30001], "3": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30002], "5": [10000, 10001, 10100, 10101, 10200, 10201, 20001, 30001], "6": [10000, 10001, 10100, 10101, 10200, 10201, 20001, 30002], "9": [10000, 10001, 10100, 10101, 10200, 10201, 20002, 30002], "299900000": [10000, 10001, 10100, 10101], "299900001": [10000, 10001, 10100, 10101, 20000, 30000], "299900002": [10000, 10001, 10100, 10101, 20000, 30001], "299900004": [10000, 10001, 10100, 10101, 20001, 30001]}})";


  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(optiling::EletwiseTiling(this->test_info_->name(), fusedmuladd_op, nlohmann::json::parse(compileInfo), runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 48);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 3200 5824 1 67 2 ");
}

TEST_F(EletwiseTilingV2, Eletwise_tiling_relu_v2) {
// dynamic_op_add_267.static_op_add_269
using namespace optiling;

std::vector<int64_t> inputA{32, 128};
std::vector<int64_t> outputA{32, 128};
std::vector<int64_t> outputB{32, 16};

TensorDesc tensor_inputA(ge::Shape(inputA), FORMAT_ND, DT_FLOAT);
TensorDesc tensor_outputA(ge::Shape(outputA), FORMAT_ND, DT_FLOAT);
TensorDesc tensor_outputB(ge::Shape(outputA), FORMAT_ND, DT_UINT8);

auto x = op::Data("x");
x.update_input_desc_x(tensor_inputA);
x.update_output_desc_y(tensor_inputA);

auto reluV2_op = op::ReluV2("ReluV2_0");
reluV2_op.set_input_x(x);
reluV2_op.update_output_desc_y(tensor_outputA);
reluV2_op.update_output_desc_mask(tensor_outputB);

std::vector<Operator> inputs{x,};
std::vector<Operator> outputs{reluV2_op};
ge::Graph graph("Eletwise_tiling1");
graph.SetInputs(inputs).SetOutputs(outputs);

ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);




std::string compileInfo = R"({"_fusion_index": [[0], [1], [2]], "_pattern": "Broadcast", "_outs_uint1": true, "_soc_version": "Ascend910", "_flag_info": [false, false, false, true, false, false, false], "_base_info": {"100": [32, 4, 21832, 10912], "120": [48, 2, 24576, 12288], "121": [48, 2, 19648, 9824], "210": [48, 2, 19648, 9824], "320": [48, 2, 32752, 16368], "230": [48, 2, 32752, 16368], "000": [48, 2, 19648, 9824], "999": [48, 2, 19648, 9824]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212000004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212010004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "212100005": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_1"], "212100006": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_2"], "212100009": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_2", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_1", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_2", "_ub_factor_2"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"]}, "_normal_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212000004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212010004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "212100005": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_1"], "212100006": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_2"], "212100009": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_2", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_1", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_2", "_ub_factor_2"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"]}, "_attr_vars": {"210000000": [], "210010000": [], "212000000": [], "212000001": [], "212000002": [], "212010002": [], "212000004": [], "212010004": [], "212100000": [], "212100001": [], "212100002": [], "212100003": [], "212100005": [], "212100006": [], "212100009": [], "221000000": [], "221000001": [], "221000002": [], "221000004": [], "232000000": [], "223000000": [], "0": [], "1": [], "2": [], "3": [], "5": [], "6": [], "9": [], "299900000": [], "299900001": [], "299900002": [], "299900004": []}, "_custom_vars": {"210000000": [], "210010000": [], "212000000": [], "212000001": [], "212000002": [], "212010002": [], "212000004": [], "212010004": [], "212100000": [], "212100001": [], "212100002": [], "212100003": [], "212100005": [], "212100006": [], "212100009": [], "221000000": [], "221000001": [], "221000002": [], "221000004": [], "232000000": [], "223000000": [], "0": [], "1": [], "2": [], "3": [], "5": [], "6": [], "9": [], "299900000": [], "299900001": [], "299900002": [], "299900004": []}, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "212000000": [10000, 10100, 10101], "212000001": [10000, 10100, 10101, 20000, 30000], "212000002": [10000, 10100, 10101, 20000, 30001], "212010002": [10000, 10100, 10101, 20000, 30001], "212000004": [10000, 10100, 10101, 20001, 30001], "212010004": [10000, 10100, 10101, 20001, 30001], "212100000": [10000, 10100, 10101, 10200], "212100001": [10000, 10100, 10101, 10200, 20000, 30000], "212100002": [10000, 10100, 10101, 10200, 20000, 30001], "212100003": [10000, 10100, 10101, 10200, 20000, 30002], "212100005": [10000, 10100, 10101, 10200, 20001, 30001], "212100006": [10000, 10100, 10101, 10200, 20001, 30002], "212100009": [10000, 10100, 10101, 10200, 20002, 30002], "221000000": [10000, 10001, 10100], "221000001": [10000, 10001, 10100, 20000, 30000], "221000002": [10000, 10001, 10100, 20000, 30001], "221000004": [10000, 10001, 10100, 20001, 30001], "232000000": [10001, 20000, 30000], "223000000": [10000, 20000, 30000], "0": [10000, 10001, 10100, 10101, 10200, 10201], "1": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30000], "2": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30001], "3": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30002], "5": [10000, 10001, 10100, 10101, 10200, 10201, 20001, 30001], "6": [10000, 10001, 10100, 10101, 10200, 10201, 20001, 30002], "9": [10000, 10001, 10100, 10101, 10200, 10201, 20002, 30002], "299900000": [10000, 10001, 10100, 10101], "299900001": [10000, 10001, 10100, 10101, 20000, 30000], "299900002": [10000, 10001, 10100, 10101, 20000, 30001], "299900004": [10000, 10001, 10100, 10101, 20001, 30001]}})";

optiling::utils::OpRunInfo runInfo;

ASSERT_TRUE(optiling::EletwiseTiling("CustomOP", reluV2_op, nlohmann::json::parse(compileInfo), runInfo));
}

TEST_F(EletwiseTilingV2, Eletwise_tiling20) {
  using namespace optiling;

  std::vector<int64_t> inputA{1, 1, 5824};
  std::vector<int64_t> inputB{32, 100, 1};
  std::vector<int64_t> output{32, 100, 5824};

  TensorDesc tensor_inputA(ge::Shape(inputA), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_inputB(ge::Shape(inputB), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_inputA);
  x1.update_output_desc_y(tensor_inputA);
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(tensor_inputB);
  x2.update_output_desc_y(tensor_inputB);

  auto add_op = op::Add("Add_0");
  add_op.set_input_x1(x1).set_input_x2(x2);
  add_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1, x2};
  std::vector<Operator> outputs{add_op};
  ge::Graph graph("Eletwise_tiling20");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string compileInfo = R"({"_fusion_index": [[0], [1], [2]], "_pattern": "Broadcast", "_outs_uint1": false, "_soc_version": "Ascend920", "_flag_info": [false, false, true, true, true, false, true], "_base_info": {"100": [48, 2, 32768, 16384], "120": [48, 2, 24576, 12288], "121": [48, 2, 19648, 9824], "210": [48, 2, 19648, 9824], "320": [48, 2, 32752, 16368], "230": [48, 2, 32752, 16368], "000": [48, 2, 19648, 9824], "999": [48, 2, 19648, 9824]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212000004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212010004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "212100005": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_1"], "212100006": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_2"], "212100009": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_2", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_1", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_2", "_ub_factor_2"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"]}, "_normal_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212000004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212010004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "212100005": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_1"], "212100006": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_2"], "212100009": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_2", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_1", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_2", "_ub_factor_2"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"]}, "_attr_vars": {"210000000": [], "210010000": [], "212000000": [], "212000001": [], "212000002": [], "212010002": [], "212000004": [], "212010004": [], "212100000": [], "212100001": [], "212100002": [], "212100003": [], "212100005": [], "212100006": [], "212100009": [], "221000000": [], "221000001": [], "221000002": [], "221000004": [], "232000000": [], "223000000": [], "0": [], "1": [], "2": [], "3": [], "5": [], "6": [], "9": [], "299900000": [], "299900001": [], "299900002": [], "299900004": []}, "_custom_vars": {"210000000": [], "210010000": [], "212000000": [], "212000001": [], "212000002": [], "212010002": [], "212000004": [], "212010004": [], "212100000": [], "212100001": [], "212100002": [], "212100003": [], "212100005": [], "212100006": [], "212100009": [], "221000000": [], "221000001": [], "221000002": [], "221000004": [], "232000000": [], "223000000": [], "0": [], "1": [], "2": [], "3": [], "5": [], "6": [], "9": [], "299900000": [], "299900001": [], "299900002": [], "299900004": []}, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "212000000": [10000, 10100, 10101], "212000001": [10000, 10100, 10101, 20000, 30000], "212000002": [10000, 10100, 10101, 20000, 30001], "212010002": [10000, 10100, 10101, 20000, 30001], "212000004": [10000, 10100, 10101, 20001, 30001], "212010004": [10000, 10100, 10101, 20001, 30001], "212100000": [10000, 10100, 10101, 10200], "212100001": [10000, 10100, 10101, 10200, 20000, 30000], "212100002": [10000, 10100, 10101, 10200, 20000, 30001], "212100003": [10000, 10100, 10101, 10200, 20000, 30002], "212100005": [10000, 10100, 10101, 10200, 20001, 30001], "212100006": [10000, 10100, 10101, 10200, 20001, 30002], "212100009": [10000, 10100, 10101, 10200, 20002, 30002], "221000000": [10000, 10001, 10100], "221000001": [10000, 10001, 10100, 20000, 30000], "221000002": [10000, 10001, 10100, 20000, 30001], "221000004": [10000, 10001, 10100, 20001, 30001], "232000000": [10001, 20000, 30000], "223000000": [10000, 20000, 30000], "0": [10000, 10001, 10100, 10101, 10200, 10201], "1": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30000], "2": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30001], "3": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30002], "5": [10000, 10001, 10100, 10101, 10200, 10201, 20001, 30001], "6": [10000, 10001, 10100, 10101, 10200, 10201, 20001, 30002], "9": [10000, 10001, 10100, 10101, 10200, 10201, 20002, 30002], "299900000": [10000, 10001, 10100, 10101], "299900001": [10000, 10001, 10100, 10101, 20000, 30000], "299900002": [10000, 10001, 10100, 10101, 20000, 30001], "299900004": [10000, 10001, 10100, 10101, 20001, 30001]}})";
  nlohmann::json op_info = nlohmann::json::parse(compileInfo.c_str());
  std::vector<std::vector<int64_t>> input_shapes{inputA, inputB};
  optiling::OpInfo c_op_info(input_shapes, DT_FLOAT);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(optiling::EletwiseTiling("CustomOP", add_op, op_info, runInfo, c_op_info));\
  EXPECT_EQ(runInfo.GetBlockDim(), 48);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 3200 5824 1 67 3 ");
}

TEST_F(EletwiseTilingV2, Eletwise_tiling_empty_tensor) {
  using namespace optiling;

  std::vector<int64_t> input{2, 0, 2};
  std::vector<int64_t> output{2, 0, 2};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto exp_op = op::Exp("Exp_0");
  exp_op.set_input_x(x1);
  exp_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{exp_op};
  ge::Graph graph("Eletwise_tiling_empty_tensor");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string op_type = "AutoTiling";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_type);


  std::string compileInfo = R"({"_pattern": "ElemWise", "_outs_uint1": false, "_flag_info": [false, false, false, true, false, false], "_base_info": {}, "_elewise_vars":{}, "_vars": {"2147483647": []}, "_normal_vars": {"2147483647": []}, "_attr_vars": {"2147483647": []}, "_custom_vars": {"2147483647": []}})";
  nlohmann::json op_info = nlohmann::json::parse(compileInfo.c_str());
  std::vector<std::vector<int64_t>> input_shapes{input};
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(optiling::EletwiseTiling(op_type, exp_op, op_info, runInfo));\
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "");
}

TEST_F(EletwiseTilingV2, Eletwise_custom_empty_tensor) {
  using namespace optiling;

  std::vector<int64_t> inputA{2, 0, 2};
  std::vector<int64_t> inputB{2, 0, 2};
  std::vector<int64_t> output{2, 0, 2};

  TensorDesc tensor_inputA(ge::Shape(inputA), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_inputB(ge::Shape(inputB), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_inputA);
  x1.update_output_desc_y(tensor_inputA);
  auto x2 = op::Data("x2");
  x2.update_input_desc_x(tensor_inputB);
  x2.update_output_desc_y(tensor_inputB);

  auto add_op = op::Add("Add_0");
  add_op.set_input_x1(x1).set_input_x2(x2);
  add_op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1, x2};
  std::vector<Operator> outputs{add_op};
  ge::Graph graph("Eletwise_custom_empty_tensor");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string compileInfo = R"({"_fusion_index": [[0], [1], [2]], "_pattern": "ElemWise", "_outs_uint1": false, "_flag_info": [false, false, true, true, false, true], "_base_info": {"100": [32, 4, 21840, 10920]}, "_elewise_vars": {"210000000": [20000, 30000], "210010000": [20000, 30000]}, "_vars": {"210000000": ["_block_factor_0", "_ub_factor_0"], "210010000": ["_block_factor_0", "_ub_factor_0"], "2147483647": []}, "_normal_vars": {"210000000":["_block_factor_0", "ub_factor_0"], "210010000": ["_block_factor_0", "_ub_factor_0"], "2147483647": []}, "_attr_vars": {"210000000":[], "210010000": [], "2147483647": []}, "_custom_vars": {"210000000":[], "210010000": [], "2147483647": []}})";
  nlohmann::json op_info = nlohmann::json::parse(compileInfo.c_str());
  std::vector<std::vector<int64_t>> input_shapes{inputA, inputB};
  optiling::OpInfo c_op_info(input_shapes, DT_FLOAT);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(optiling::EletwiseTiling("CustomOP", add_op, op_info, runInfo, c_op_info));
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "");
}
