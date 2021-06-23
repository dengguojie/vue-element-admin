#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nonlinear_fuc_ops.h"
#include "elewise_calculation_ops.h"
#include "state_ops.h"
#include "nn_batch_norm_ops.h"
#include "reduce_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"

using namespace ge;
using namespace op;

class layernorm_training_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "layernorm_training_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "layernorm_training_fusion_test TearDown" << std::endl;
    }
};

TEST_F(layernorm_training_fusion_test, layernorm_training_fusion_test_1) {
    ge::Graph graph("layernorm_training_fusion_test");

    ge::TensorDesc add0_desc(ge::Shape({1}), FORMAT_ND, DT_FLOAT);
    int64_t add0_size = add0_desc.GetShape().GetShapeSize();
    float add0_data = 0.0000001;
    ge::Tensor add0_tensor(add0_desc, reinterpret_cast<uint8_t*>(&add0_data), add0_size);

    ge::TensorDesc mul1_desc(ge::Shape({224}), FORMAT_ND, DT_FLOAT);
    int64_t mul1_size = mul1_desc.GetShape().GetShapeSize();
    std::vector<int> mul1_data(224, 1);
    ge::Tensor mul1_tensor(mul1_desc, reinterpret_cast<uint8_t*>(mul1_data.data()), mul1_size);

    ge::TensorDesc add1_desc(ge::Shape({1}), FORMAT_ND, DT_FLOAT);
    int64_t add1_size = add1_desc.GetShape().GetShapeSize();
    float add1_data = 1.;
    ge::Tensor add1_tensor(add1_desc, reinterpret_cast<uint8_t*>(&add1_data), add1_size);

    auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
    auto mul1_const_op = op::Constant().set_attr_value(mul1_tensor);
    auto add1_const_op = op::Constant().set_attr_value(add1_tensor);
    add0_const_op.update_output_desc_y(add0_desc);
    mul1_const_op.update_output_desc_y(mul1_desc);
    add1_const_op.update_output_desc_y(add1_desc);

    std::vector<int64_t> axes;
    axes.push_back(-1);
    auto data0 = op::Data().set_attr_index(0);
    auto mean0 = op::ReduceMeanD("mean0").set_input_x(data0).set_attr_axes(axes).set_attr_keep_dims(true);
    auto squaredifference0 = op::SquaredDifference("SquaredDifference0").set_input_x1(data0).set_input_x2(mean0);
    auto mean1 = op::ReduceMeanD("mean1").set_input_x(squaredifference0).set_attr_axes(axes).set_attr_keep_dims(true);
    auto add0 = op::Add("add0").set_input_x1(mean1).set_input_x2(add0_const_op);
    auto rsqrt0 = op::Rsqrt("rsqrt0").set_input_x(add0);
    auto mul0 = op::Mul("mul0").set_input_x1(rsqrt0).set_input_x2(mul1_const_op);
    auto mul1 = op::Mul("mul1").set_input_x1(mean0).set_input_x2(mul0);
    auto sub0 = op::Sub("sub0").set_input_x1(add1_const_op).set_input_x2(mul1);
    auto mul2 = op::Mul("mul2").set_input_x1(data0).set_input_x2(mul0);
    auto add1 = op::Add("add1").set_input_x1(mul2).set_input_x2(sub0);
    auto add2 = op::Add("add2").set_input_x1(add1).set_input_x2(add0_const_op);

    ge::TensorDesc data0_desc(ge::Shape({3, 224, 224}), FORMAT_ND,  DT_FLOAT);
    ge::TensorDesc data1_desc(ge::Shape({224}), FORMAT_ND,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    mean0.update_input_desc_x(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{add1};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormTrainingFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{3, 224, 224};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "LayerNorm") {
            findOp = true;
            auto inputDesc = node->GetOpDesc()->GetInputDesc(0);
            std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
            if (dims == expectShape) {
                shapeMatch = true;
            }
        }
    }
    EXPECT_EQ(findOp, true);
    EXPECT_EQ(shapeMatch, true);
}
