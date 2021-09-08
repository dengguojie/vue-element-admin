#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "nonlinear_fuc_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class mul_maximum_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "mul_maximum_fusion SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "mul_maximum_fusion TearDown" << std::endl;
    }
};

TEST_F(mul_maximum_fusion_test, mul_maximum_fusion_test_1) {
    ge::Graph graph("mul_maximum_fusion_test_1");

    auto input_x = op::Data("input_x");

    std::vector<int64_t> dims_x{3, 32};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescMul(shape_x, ge::FORMAT_NHWC, ge::DT_FLOAT16);
    input_x.update_input_desc_x(tensorDescMul);
    input_x.update_output_desc_y(tensorDescMul);

    auto mul_const = op::Constant("mul_const");

    std::vector<int64_t> dims_scale ={};
    ge::Shape shape_scale(dims_scale);
    ge::TensorDesc tensorDescScale(shape_scale, FORMAT_NHWC,  ge::DT_FLOAT16);

    Tensor consttensor1;
    float * dataValue = new float[1];
    dataValue[0] = 0;

    consttensor1.SetTensorDesc(tensorDescScale);
    consttensor1.SetData((uint8_t*)dataValue, 4);
    mul_const.set_attr_value(consttensor1);

    auto mul_op = op::Mul("mul_0");
    mul_op.set_input_x1(input_x)
          .set_input_x2(mul_const);

    auto maximum_op = op::Maximum("maximum_0");
    maximum_op.set_input_x1(mul_op)
              .set_input_x2(input_x);

    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(maximum_op);

    std::vector<Operator> inputs{input_x, mul_const};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "mul_maximum_fusion_test_1_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("MulMaximumFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "mul_maximum_fusion_test_1_after");
    bool findLeakyRelu = false;
    vector<int64_t> expectShape{3, 32};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "LeakyRelu") {
            findLeakyRelu = true;
        }
    }
    EXPECT_EQ(findLeakyRelu, true);
}


