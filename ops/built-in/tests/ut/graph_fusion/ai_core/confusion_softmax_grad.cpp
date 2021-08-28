#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "nn_norm_ops.h"
#include "reduce_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class confusion_softmax_grad_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "confusionsoftmaxgrad_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "confusionsoftmaxgrad_fusion_test TearDown" << std::endl;
  }
};

TEST_F(confusion_softmax_grad_fusion_test, confusion_softmax_grad_fusion_test_1) {
    ge::Graph graph("confusion_softmax_grad_fusion_test_1");

    auto in_input_x_data = op::Data("input_data");
    auto in_input_x_data1 = op::Data("input_data1");
    std::vector<int64_t> dims{-1, 8, 900, 3};
    ge::Shape shape(dims);
    std::vector<int64_t> dims1{1, 8, 900, 3};
    ge::Shape shape1(dims1);
    ge::TensorDesc tensorDesc0(shape, ge::FORMAT_ND, ge::DT_FLOAT);
    ge::TensorDesc tensorDesc1(shape1, ge::FORMAT_ND, ge::DT_FLOAT);
    in_input_x_data.update_input_desc_x(tensorDesc0);
    in_input_x_data.update_output_desc_y(tensorDesc0);
    
    in_input_x_data1.update_input_desc_x(tensorDesc1);
    in_input_x_data1.update_output_desc_y(tensorDesc1);

    auto mul_op = op::Mul("mul_0");
    mul_op.set_input_x1(in_input_x_data)
          .set_input_x2(in_input_x_data1);
        //.set_attr_axes({-1});
    auto reducesumd_op = op::ReduceSumD("reducesumd_0");
    reducesumd_op.set_input_x(mul_op)
                 .set_attr_axes({3})
                 .set_attr_keep_dims(true);

    auto sub_op = op::Sub("sub_op_0");
    sub_op.set_input_x1(in_input_x_data)
          .set_input_x2(reducesumd_op);
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(sub_op);

    std::vector<Operator> inputs{in_input_x_data,in_input_x_data1};
    std::vector<Operator> outputs{end_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZConfusionSoftmaxGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findmul = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Mul") {
            findmul = true;
        }
    }
    EXPECT_EQ(findmul, true);
}
TEST_F(confusion_softmax_grad_fusion_test, confusion_softmax_grad_fusion_test_2) {
    ge::Graph graph("confusion_softmax_grad_fusion_test_2");

    auto in_input_x_data = op::Data("input_data");
    auto in_input_x_data1 = op::Data("input_data1");
    std::vector<int64_t> dims{1, 8, 900, 3};
    ge::Shape shape(dims);
    std::vector<int64_t> dims1{-1, 8, 900, 3};
    ge::Shape shape1(dims1);
    ge::TensorDesc tensorDesc0(shape, ge::FORMAT_ND, ge::DT_FLOAT);
    ge::TensorDesc tensorDesc1(shape1, ge::FORMAT_ND, ge::DT_FLOAT);
    in_input_x_data.update_input_desc_x(tensorDesc0);
    in_input_x_data.update_output_desc_y(tensorDesc0);
    
    in_input_x_data1.update_input_desc_x(tensorDesc1);
    in_input_x_data1.update_output_desc_y(tensorDesc1);

    auto mul_op = op::Mul("mul_0");
    mul_op.set_input_x1(in_input_x_data)
          .set_input_x2(in_input_x_data1);
        //.set_attr_axes({-1});
    auto reducesumd_op = op::ReduceSumD("reducesumd_0");
    reducesumd_op.set_input_x(mul_op)
                 .set_attr_axes({3})
                 .set_attr_keep_dims(true);

    auto sub_op = op::Sub("sub_op_0");
    sub_op.set_input_x1(in_input_x_data)
          .set_input_x2(reducesumd_op);
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(sub_op);

    std::vector<Operator> inputs{in_input_x_data,in_input_x_data1};
    std::vector<Operator> outputs{end_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZConfusionSoftmaxGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findmul = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Mul") {
            findmul = true;
        }
    }
    EXPECT_EQ(findmul, true);
}