#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "reduce_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class softmax_grad_ext_v2_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "softmax_grad_ext_v2_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "softmax_grad_ext_v2_fusion_test TearDown" << std::endl;
    }
};

TEST_F(softmax_grad_ext_v2_fusion_test, softmax_grad_ext_v2_fusion_test_1) {
    ge::Graph graph("softmax_grad_ext_v2_fusion_test_1");
    
    ge::Tensor mul0_tensor;
    std::vector<int64_t> mul0_vec{16,16};
    ge::Shape mul0_shape(mul0_vec);
    ge::TensorDesc mul0_desc(mul0_shape, FORMAT_ND, DT_FLOAT);
    int64_t mul0_size = 256;
    mul0_desc.SetSize(mul0_size * sizeof(float));
    mul0_tensor.SetTensorDesc(mul0_desc);
    float* mul0_data = nullptr;
    mul0_data = new float[mul0_size];
    for (int i=0; i<mul0_size; i++) {
        *(mul0_data + i) = 1;
    }
    mul0_tensor.SetData((uint8_t*)mul0_data, mul0_size * sizeof(float));
    delete [] mul0_data;

    ge::Tensor mul1_tensor;
    std::vector<int64_t> mul1_vec{16,16};
    ge::Shape mul1_shape(mul1_vec);
    ge::TensorDesc mul1_desc(mul1_shape, FORMAT_ND, DT_FLOAT);
    int64_t mul1_size = 256;
    mul1_desc.SetSize(mul1_size * sizeof(float));
    mul1_tensor.SetTensorDesc(mul1_desc);
    float* mul1_data = nullptr;
    mul1_data = new float[mul1_size];
    for (int i=0; i<mul1_size; i++) {
        *(mul1_data + i) = 1;
    }
    mul1_tensor.SetData((uint8_t*)mul1_data, mul1_size * sizeof(float));
    delete [] mul1_data;

    ge::Tensor mul2_tensor;
    std::vector<int64_t> mul2_vec{16,16};
    ge::Shape mul2_shape(mul2_vec);
    ge::TensorDesc mul2_desc(mul2_shape, FORMAT_ND, DT_FLOAT);
    int64_t mul2_size = 256;
    mul2_desc.SetSize(mul2_size * sizeof(float));
    mul2_tensor.SetTensorDesc(mul2_desc);
    float* mul2_data = nullptr;
    mul2_data = new float[mul2_size];
    for (int i=0; i<mul2_size; i++) {
        *(mul2_data + i) = 1;
    }
    mul2_tensor.SetData((uint8_t*)mul2_data, mul2_size * sizeof(float));
    delete [] mul2_data;

    auto mul0_const_op = op::Constant().set_attr_value(mul0_tensor);
    auto mul1_const_op = op::Constant().set_attr_value(mul1_tensor);
    auto mul2_const_op = op::Constant().set_attr_value(mul2_tensor);
    mul0_const_op.update_output_desc_y(mul0_desc);
    mul1_const_op.update_output_desc_y(mul1_desc);
    mul2_const_op.update_output_desc_y(mul2_desc);

    std::vector<int64_t> axes;
    axes.push_back(-1);
    auto data0 = op::Data().set_attr_index(0);
    auto mul0 = op::Mul("mul0")
                        .set_input_x1(mul0_const_op)
                        .set_input_x2(mul1_const_op);
    auto sum0 = op::ReduceSumD("sum0")
                        .set_input_x(mul0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto sub0 = op::Sub("sub0")
                        .set_input_x1(mul0_const_op)
                        .set_input_x2(sum0);
    auto mul1 = op::Mul("mul1")
                        .set_input_x1(sub0)
                        .set_input_x2(mul1_const_op);
    auto mul2 = op::Mul("mul2")
                        .set_input_x1(mul1)
                        .set_input_x2(mul2_const_op);

    std::vector<Operator> inputs{mul0_const_op, mul1_const_op, mul2_const_op};
    std::vector<Operator> outputs{mul2};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("SoftmaxGradExtFusionV2", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SoftmaxGradExt") {
            findOp = true;
        }
    }
    EXPECT_EQ(false, false);
}

