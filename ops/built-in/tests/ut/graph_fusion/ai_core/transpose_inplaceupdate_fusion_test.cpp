#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "fusion_pass_test_utils.h"
#include "array_ops.h"
#include "transformation_ops.h"
#include "selection_ops.h"


using namespace ge;
using namespace op;


class transpose_inplaceupdate_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "transpose_inplaceupdate_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "transpose_inplaceupdate_fusion_test TearDown" << std::endl;
  }
  void BuildGraph(ComputeGraphPtr &compute_graph_ptr, const vector<int64_t>& inputshape0, const vector<int64_t>& inputshape1, const vector<int64_t>& inputshape2, const vector<int64_t>& attrshape0, const vector<int64_t>& attrshape1) {
    ge::Graph graph("test_transpose_inplaceupdate");
    auto input0 = op::Data("input0");
    ge::Shape shape0(inputshape0);
    ge::TensorDesc tensorDescX0(shape0, FORMAT_NHWC, DT_FLOAT16);
    input0.update_input_desc_x(tensorDescX0);
    input0.update_output_desc_y(tensorDescX0);

    TensorDesc desc_add_1(ge::Shape(inputshape1), FORMAT_ND, DT_FLOAT16);
    Tensor const_tensor1(desc_add_1);
    auto const1 = op::Const("const1");
    const1.set_attr_value(const_tensor1);

    TensorDesc desc_add_2(ge::Shape(inputshape2), FORMAT_ND, DT_FLOAT16);
    Tensor const_tensor2(desc_add_2);
    auto const2 = op::Const("const2");
    const2.set_attr_value(const_tensor2);


    auto transpose_layer1 = op::TransposeD("transposeD1");
    if(attrshape0.empty()){
      transpose_layer1.set_input_x(input0);
    }else{
      transpose_layer1.set_input_x(input0)
                    .set_attr_perm(attrshape0);
    }
    auto inplaceUpdate_layer1 = op::InplaceUpdate("inplaceUpdate");
    inplaceUpdate_layer1.set_input_x(transpose_layer1)
                    .set_input_indices(const1)
                    .set_input_v(const2);
    auto transpose_layer2 = op::TransposeD("transposeD2");
    if(attrshape1.empty()){
      transpose_layer2.set_input_x(inplaceUpdate_layer1);
    }else{
      transpose_layer2.set_input_x(inplaceUpdate_layer1)
                    .set_attr_perm(attrshape1);
    }
    std::vector<Operator> inputs{input0,const1,const2};
    std::vector<Operator> outputs{transpose_layer2};
    graph.SetInputs(inputs).SetOutputs(outputs);
    compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);;
  }
};
TEST_F(transpose_inplaceupdate_fusion_test, transpose_inplaceupdate_fusion_test_1){
  ge::ComputeGraphPtr compute_graph_ptr;
  BuildGraph(compute_graph_ptr, {48,48,48,48}, {1}, {1,48,48,48}, {2,0,1,3}, {1,2,0,3});
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ZTransposeInplaceUpdateFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool InplaceUpdate_fusion = false;
  bool TransposeD_fusion = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "InplaceUpdate") {
        InplaceUpdate_fusion = true;
    }
    if (node->GetType() == "TransposeD") {
        TransposeD_fusion = true;
    }
  }
  EXPECT_EQ(InplaceUpdate_fusion, true);
  EXPECT_EQ(TransposeD_fusion, false);
}
// inplace_dims0.size() < 4
TEST_F(transpose_inplaceupdate_fusion_test, transpose_inplaceupdate_fusion_test_2){
  ge::ComputeGraphPtr compute_graph_ptr;
  BuildGraph(compute_graph_ptr, {48,48,48}, {1}, {1,48,48,48}, {2,0,1,3}, {1,2,0,3});
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ZTransposeInplaceUpdateFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool InplaceUpdate_fusion = false;
  bool TransposeD_fusion = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "InplaceUpdate") {
        InplaceUpdate_fusion = true;
    }
    if (node->GetType() == "TransposeD") {
        TransposeD_fusion = true;
    }
  }
  EXPECT_EQ(InplaceUpdate_fusion, true);
  EXPECT_EQ(TransposeD_fusion, true);
}
// (input_dims.size() == 4) && (perm0.size() == 4) && (perm1.size() == 4) && (perm0[3] != 3)
TEST_F(transpose_inplaceupdate_fusion_test, transpose_inplaceupdate_fusion_test_3){
  ge::ComputeGraphPtr compute_graph_ptr;
  BuildGraph(compute_graph_ptr, {48,48,48,48}, {1}, {1,48,48,48}, {2,0,1,2}, {1,2,0,3});
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ZTransposeInplaceUpdateFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool InplaceUpdate_fusion = false;
  bool TransposeD_fusion = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "InplaceUpdate") {
        InplaceUpdate_fusion = true;
    }
    if (node->GetType() == "TransposeD") {
        TransposeD_fusion = true;
    }
  }
  EXPECT_EQ(InplaceUpdate_fusion, true);
  EXPECT_EQ(TransposeD_fusion, true);
}
// (input_dims.size() == 4) && (perm0.size() == 4) && (perm1.size() == 4) && (perm1[3] != 3)
TEST_F(transpose_inplaceupdate_fusion_test, transpose_inplaceupdate_fusion_test_4){
  ge::ComputeGraphPtr compute_graph_ptr;
  BuildGraph(compute_graph_ptr, {48,48,48,48}, {1}, {1,48,48,48}, {2,0,1,3}, {1,2,0,2});
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ZTransposeInplaceUpdateFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool InplaceUpdate_fusion = false;
  bool TransposeD_fusion = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "InplaceUpdate") {
        InplaceUpdate_fusion = true;
    }
    if (node->GetType() == "TransposeD") {
        TransposeD_fusion = true;
    }
  }
  EXPECT_EQ(InplaceUpdate_fusion, true);
  EXPECT_EQ(TransposeD_fusion, true);
}
// perm1.size() != 4
TEST_F(transpose_inplaceupdate_fusion_test, transpose_inplaceupdate_fusion_test_5){
  ge::ComputeGraphPtr compute_graph_ptr;
  BuildGraph(compute_graph_ptr, {48,48,48,48}, {1}, {1,48,48,48}, {2,0,1,3}, {1,2,0});
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ZTransposeInplaceUpdateFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool InplaceUpdate_fusion = false;
  bool TransposeD_fusion = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "InplaceUpdate") {
        InplaceUpdate_fusion = true;
    }
    if (node->GetType() == "TransposeD") {
        TransposeD_fusion = true;
    }
  }
  EXPECT_EQ(InplaceUpdate_fusion, true);
  EXPECT_EQ(TransposeD_fusion, true);
}
// (inplace_dims2.size() != 4)
TEST_F(transpose_inplaceupdate_fusion_test, transpose_inplaceupdate_fusion_test_6){
  ge::ComputeGraphPtr compute_graph_ptr;
  BuildGraph(compute_graph_ptr, {48,48,48,48}, {1}, {1,48,48}, {2,0,1,3}, {1,2,0,3});
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ZTransposeInplaceUpdateFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool InplaceUpdate_fusion = false;
  bool TransposeD_fusion = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "InplaceUpdate") {
        InplaceUpdate_fusion = true;
    }
    if (node->GetType() == "TransposeD") {
        TransposeD_fusion = true;
    }
  }
  EXPECT_EQ(InplaceUpdate_fusion, true);
  EXPECT_EQ(TransposeD_fusion, true);
}
// (inplace_dims2[0] != 1)
TEST_F(transpose_inplaceupdate_fusion_test, transpose_inplaceupdate_fusion_test_7){
  ge::ComputeGraphPtr compute_graph_ptr;
  BuildGraph(compute_graph_ptr, {48,48,48,48}, {1}, {2,48,48,48}, {2,0,1,3}, {1,2,0,3});
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ZTransposeInplaceUpdateFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool InplaceUpdate_fusion = false;
  bool TransposeD_fusion = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "InplaceUpdate") {
        InplaceUpdate_fusion = true;
    }
    if (node->GetType() == "TransposeD") {
        TransposeD_fusion = true;
    }
  }
  EXPECT_EQ(InplaceUpdate_fusion, true);
  EXPECT_EQ(TransposeD_fusion, true);
}
// get attr perm0 failed
TEST_F(transpose_inplaceupdate_fusion_test, transpose_inplaceupdate_fusion_test_8){
  ge::ComputeGraphPtr compute_graph_ptr;
  BuildGraph(compute_graph_ptr, {48,48,48,48}, {1}, {1,48,48,48}, {}, {1,2,0,3});
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ZTransposeInplaceUpdateFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool InplaceUpdate_fusion = false;
  bool TransposeD_fusion = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "InplaceUpdate") {
        InplaceUpdate_fusion = true;
    }
    if (node->GetType() == "TransposeD") {
        TransposeD_fusion = true;
    }
  }
  EXPECT_EQ(InplaceUpdate_fusion, true);
  EXPECT_EQ(TransposeD_fusion, true);
}
// get attr perm1 failed
TEST_F(transpose_inplaceupdate_fusion_test, transpose_inplaceupdate_fusion_test_9){
  ge::ComputeGraphPtr compute_graph_ptr;
  BuildGraph(compute_graph_ptr, {48,48,48,48}, {1}, {1,48,48,48}, {2,0,1,3}, {});
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ZTransposeInplaceUpdateFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool InplaceUpdate_fusion = false;
  bool TransposeD_fusion = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "InplaceUpdate") {
        InplaceUpdate_fusion = true;
    }
    if (node->GetType() == "TransposeD") {
        TransposeD_fusion = true;
    }
  }
  EXPECT_EQ(InplaceUpdate_fusion, true);
  EXPECT_EQ(TransposeD_fusion, true);
}