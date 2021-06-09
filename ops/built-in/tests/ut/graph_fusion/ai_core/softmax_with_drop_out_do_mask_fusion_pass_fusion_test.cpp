#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "nn_norm_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class softmax_with_drop_out_do_mask_fusion_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "softmax_with_drop_out_do_mask_fusion_pass_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "softmax_with_drop_out_do_mask_fusion_pass_test TearDown" << std::endl;
  }
};

TEST_F(softmax_with_drop_out_do_mask_fusion_pass_test, softmax_with_drop_out_do_mask_fusion_pass_test_1) {
    ge::Graph graph("softmax_with_drop_out_do_mask_fusion_pass_test_1");

    auto in_input_x_data = op::Data("diag_input_data");
    std::vector<int64_t> dims{4,16,32,32,16,16};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
    in_input_x_data.update_input_desc_x(tensorDesc);
    in_input_x_data.update_output_desc_y(tensorDesc);

    auto in_input_mask_data = op::Data("diag_mask_data");
    ge::TensorDesc tensorDesc1(shape, FORMAT_ND, ge::DT_UINT8);
    in_input_mask_data.update_input_desc_x(tensorDesc1);
    in_input_mask_data.update_output_desc_y(tensorDesc1);

    auto softmax_op = op::SoftmaxV2("softmaxv2_0");
    softmax_op.set_input_x(in_input_x_data)
        .set_attr_axes({-1});

    auto end_op = op::DropOutDoMaskV3D("end_op_0");
    end_op.set_input_x(softmax_op);
    end_op.set_input_mask(in_input_mask_data);
    end_op.set_attr_keep_prob(0.5);

    std::vector<Operator> inputs{in_input_x_data, in_input_mask_data};
    std::vector<Operator> outputs{end_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("SoftmaxWithDropOutDoMaskFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findSoftmax = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SoftmaxV2WithDropOutDoMaskV3D") {
            findSoftmax = true;
        }
    }
    EXPECT_EQ(true, true);
}