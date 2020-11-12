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

class layer_norm_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "layer_norm_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "layer_norm_fusion_test TearDown" << std::endl;
    }
};

TEST_F(layer_norm_fusion_test, layer_norm_fusion_test_1) {
    ge::Graph graph("layer_norm_fusion_test_1");

    ge::Tensor add0_tensor;
    std::vector<int64_t> add0_vec{1};
    ge::Shape add0_shape(add0_vec);
    ge::TensorDesc add0_desc(add0_shape, FORMAT_ND, DT_FLOAT);
    int64_t add0_size = add0_desc.GetShape().GetShapeSize();
    add0_desc.SetSize(add0_size * sizeof(float));
    add0_tensor.SetTensorDesc(add0_desc);
    float* add0_data = nullptr;
    add0_data = new float[add0_size];
    *(add0_data + 0) = 1;
    add0_tensor.SetData((uint8_t*)add0_data, add0_size * sizeof(float));
    delete [] add0_data;

    ge::Tensor mul0_tensor;
    std::vector<int64_t> mul0_vec{224};
    ge::Shape mul0_shape(mul0_vec);
    ge::TensorDesc mul0_desc(mul0_shape, FORMAT_ND, DT_FLOAT);
    int64_t mul0_size = mul0_desc.GetShape().GetShapeSize();
    mul0_desc.SetSize(mul0_size * sizeof(float));
    mul0_tensor.SetTensorDesc(mul0_desc);
    float* mul0_data = nullptr;
    mul0_data = new float[mul0_size];
    for (int i=0; i<mul0_size; i++) {
        *(mul0_data + i) = 1;
    }
    mul0_tensor.SetData((uint8_t*)mul0_data, mul0_size * sizeof(float));
    delete [] mul0_data;

    ge::Tensor add1_tensor;
    std::vector<int64_t> add1_vec{224};
    ge::Shape add1_shape(add1_vec);
    ge::TensorDesc add1_desc(add1_shape, FORMAT_ND, DT_FLOAT);
    int64_t add1_size = add1_desc.GetShape().GetShapeSize();
    add1_desc.SetSize(add1_size * sizeof(float));
    add1_tensor.SetTensorDesc(add1_desc);
    float* add1_data = nullptr;
    add1_data = new float[add1_size];
    for (int i=0; i<add1_size; i++) {
        *(add1_data + i) = 1;
    }
    add1_tensor.SetData((uint8_t*)add1_data, add1_size * sizeof(float));
    delete [] add1_data;

    auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
    auto mul0_const_op = op::Constant().set_attr_value(mul0_tensor);
    auto add1_const_op = op::Constant().set_attr_value(add1_tensor);
    add0_const_op.update_output_desc_y(add0_desc);
    mul0_const_op.update_output_desc_y(mul0_desc);
    add1_const_op.update_output_desc_y(add1_desc);

    std::vector<int64_t> axes;
    axes.push_back(-1);
    auto data0 = op::Data().set_attr_index(0);
    auto mean0 = op::ReduceMeanD("mean0")
                        .set_input_x(data0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto squared0 = op::SquaredDifference("squared0")
                        .set_input_x1(mean0)
                        .set_input_x2(data0);
    auto sub0 = op::Sub("sub0")
                        .set_input_x1(data0)
                        .set_input_x2(mean0);
    auto mean1 = op::ReduceMeanD("mean1")
                        .set_input_x(squared0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto add0 = op::Add("add0")
                        .set_input_x1(add0_const_op)
                        .set_input_x2(mean1);
    auto rsqrt0 = op::Rsqrt("rsqrt0")
                        .set_input_x(add0);
    auto mul0 = op::Mul("mul0")
                        .set_input_x1(mul0_const_op)
                        .set_input_x2(rsqrt0);
    auto mul1 = op::Mul("mul1")
                        .set_input_x1(mul0)
                        .set_input_x2(sub0);
    auto add1 = op::Add("add1")
                        .set_input_x1(add1_const_op)
                        .set_input_x2(mul1);

    std::vector<int64_t> data0_vec{1, 3, 224, 224};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_ND,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    mean0.update_input_desc_x(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{add1};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 3, 224, 224};
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
TEST_F(layer_norm_fusion_test, layer_norm_fusion_test_2) {
    ge::Graph graph("layer_norm_fusion_test_2");

    ge::Tensor add0_tensor;
    std::vector<int64_t> add0_vec{1};
    ge::Shape add0_shape(add0_vec);
    ge::TensorDesc add0_desc(add0_shape, FORMAT_ND, DT_FLOAT);
    int64_t add0_size = add0_desc.GetShape().GetShapeSize();
    add0_desc.SetSize(add0_size * sizeof(float));
    add0_tensor.SetTensorDesc(add0_desc);
    float* add0_data = nullptr;
    add0_data = new float[add0_size];
    *(add0_data + 0) = 1;
    add0_tensor.SetData((uint8_t*)add0_data, add0_size * sizeof(float));
    delete [] add0_data;

    ge::Tensor mul0_tensor;
    std::vector<int64_t> mul0_vec{224};
    ge::Shape mul0_shape(mul0_vec);
    ge::TensorDesc mul0_desc(mul0_shape, FORMAT_ND, DT_FLOAT);
    int64_t mul0_size = mul0_desc.GetShape().GetShapeSize();
    mul0_desc.SetSize(mul0_size * sizeof(float));
    mul0_tensor.SetTensorDesc(mul0_desc);
    float* mul0_data = nullptr;
    mul0_data = new float[mul0_size];
    for (int i=0; i<mul0_size; i++) {
        *(mul0_data + i) = 1;
    }
    mul0_tensor.SetData((uint8_t*)mul0_data, mul0_size * sizeof(float));
    delete [] mul0_data;

    ge::Tensor add1_tensor;
    std::vector<int64_t> add1_vec{224};
    ge::Shape add1_shape(add1_vec);
    ge::TensorDesc add1_desc(add1_shape, FORMAT_ND, DT_FLOAT);
    int64_t add1_size = add1_desc.GetShape().GetShapeSize();
    add1_desc.SetSize(add1_size * sizeof(float));
    add1_tensor.SetTensorDesc(add1_desc);
    float* add1_data = nullptr;
    add1_data = new float[add1_size];
    for (int i=0; i<add1_size; i++) {
        *(add1_data + i) = 1;
    }
    add1_tensor.SetData((uint8_t*)add1_data, add1_size * sizeof(float));
    delete [] add1_data;

    auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
    auto mul0_const_op = op::Constant().set_attr_value(mul0_tensor);
    auto add1_const_op = op::Constant().set_attr_value(add1_tensor);
    add0_const_op.update_output_desc_y(add0_desc);
    mul0_const_op.update_output_desc_y(mul0_desc);
    add1_const_op.update_output_desc_y(add1_desc);

    std::vector<int64_t> axes;
    axes.push_back(-1);
    auto data0 = op::Data().set_attr_index(0);
    auto data1 = op::Data().set_attr_index(1);
    auto mean0 = op::ReduceMeanD("mean0")
                        .set_input_x(data0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto squared0 = op::SquaredDifference("squared0")
                        .set_input_x1(mean0)
                        .set_input_x2(data0);
    auto sub0 = op::Sub("sub0")
                        .set_input_x1(data1)
                        .set_input_x2(mean0);
    auto mean1 = op::ReduceMeanD("mean1")
                        .set_input_x(squared0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto add0 = op::Add("add0")
                        .set_input_x1(add0_const_op)
                        .set_input_x2(mean1);
    auto rsqrt0 = op::Rsqrt("rsqrt0")
                        .set_input_x(add0);
    auto mul0 = op::Mul("mul0")
                        .set_input_x1(mul0_const_op)
                        .set_input_x2(rsqrt0);
    auto mul1 = op::Mul("mul1")
                        .set_input_x1(mul0)
                        .set_input_x2(sub0);
    auto add1 = op::Add("add1")
                        .set_input_x1(add1_const_op)
                        .set_input_x2(mul1);

    std::vector<int64_t> data0_vec{1, 3, 224, 224};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_ND,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    data1.update_input_desc_x(data0_desc);
    data1.update_output_desc_y(data0_desc);
    mean0.update_input_desc_x(data0_desc);
    std::vector<Operator> inputs{data0, data1};
    std::vector<Operator> outputs{add1};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 3, 224, 224};
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
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}
TEST_F(layer_norm_fusion_test, layer_norm_fusion_test_3) {
    ge::Graph graph("layer_norm_fusion_test_3");

    ge::Tensor add0_tensor;
    std::vector<int64_t> add0_vec{1};
    ge::Shape add0_shape(add0_vec);
    ge::TensorDesc add0_desc(add0_shape, FORMAT_ND, DT_FLOAT);
    int64_t add0_size = add0_desc.GetShape().GetShapeSize();
    add0_desc.SetSize(add0_size * sizeof(float));
    add0_tensor.SetTensorDesc(add0_desc);
    float* add0_data = nullptr;
    add0_data = new float[add0_size];
    *(add0_data + 0) = 1;
    add0_tensor.SetData((uint8_t*)add0_data, add0_size * sizeof(float));
    delete [] add0_data;

    ge::Tensor mul0_tensor;
    std::vector<int64_t> mul0_vec{224};
    ge::Shape mul0_shape(mul0_vec);
    ge::TensorDesc mul0_desc(mul0_shape, FORMAT_ND, DT_FLOAT);
    int64_t mul0_size = mul0_desc.GetShape().GetShapeSize();
    mul0_desc.SetSize(mul0_size * sizeof(float));
    mul0_tensor.SetTensorDesc(mul0_desc);
    float* mul0_data = nullptr;
    mul0_data = new float[mul0_size];
    for (int i=0; i<mul0_size; i++) {
        *(mul0_data + i) = 1;
    }
    mul0_tensor.SetData((uint8_t*)mul0_data, mul0_size * sizeof(float));
    delete [] mul0_data;

    ge::Tensor add1_tensor;
    std::vector<int64_t> add1_vec{224};
    ge::Shape add1_shape(add1_vec);
    ge::TensorDesc add1_desc(add1_shape, FORMAT_ND, DT_FLOAT);
    int64_t add1_size = add1_desc.GetShape().GetShapeSize();
    add1_desc.SetSize(add1_size * sizeof(float));
    add1_tensor.SetTensorDesc(add1_desc);
    float* add1_data = nullptr;
    add1_data = new float[add1_size];
    for (int i=0; i<add1_size; i++) {
        *(add1_data + i) = 1;
    }
    add1_tensor.SetData((uint8_t*)add1_data, add1_size * sizeof(float));
    delete [] add1_data;

    auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
    auto mul0_const_op = op::Constant().set_attr_value(mul0_tensor);
    auto add1_const_op = op::Constant().set_attr_value(add1_tensor);
    add0_const_op.update_output_desc_y(add0_desc);
    mul0_const_op.update_output_desc_y(mul0_desc);
    add1_const_op.update_output_desc_y(add1_desc);

    std::vector<int64_t> axes;
    axes.push_back(-1);
    auto data0 = op::Data().set_attr_index(0);
    auto data1 = op::Data().set_attr_index(1);
    auto mean0 = op::ReduceMeanD("mean0")
                        .set_input_x(data0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto squared0 = op::SquaredDifference("squared0")
                        .set_input_x1(mean0)
                        .set_input_x2(data1);
    auto sub0 = op::Sub("sub0")
                        .set_input_x1(data0)
                        .set_input_x2(mean0);
    auto mean1 = op::ReduceMeanD("mean1")
                        .set_input_x(squared0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto add0 = op::Add("add0")
                        .set_input_x1(add0_const_op)
                        .set_input_x2(mean1);
    auto rsqrt0 = op::Rsqrt("rsqrt0")
                        .set_input_x(add0);
    auto mul0 = op::Mul("mul0")
                        .set_input_x1(mul0_const_op)
                        .set_input_x2(rsqrt0);
    auto mul1 = op::Mul("mul1")
                        .set_input_x1(mul0)
                        .set_input_x2(sub0);
    auto add1 = op::Add("add1")
                        .set_input_x1(add1_const_op)
                        .set_input_x2(mul1);

    std::vector<int64_t> data0_vec{1, 3, 224, 224};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_ND,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    data1.update_input_desc_x(data0_desc);
    data1.update_output_desc_y(data0_desc);
    mean0.update_input_desc_x(data0_desc);
    std::vector<Operator> inputs{data0, data1};
    std::vector<Operator> outputs{add1};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 3, 224, 224};
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
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}
TEST_F(layer_norm_fusion_test, layer_norm_fusion_test_4) {
    ge::Graph graph("layer_norm_fusion_test_4");

    ge::Tensor add0_tensor;
    std::vector<int64_t> add0_vec{1};
    ge::Shape add0_shape(add0_vec);
    ge::TensorDesc add0_desc(add0_shape, FORMAT_ND, DT_FLOAT);
    int64_t add0_size = add0_desc.GetShape().GetShapeSize();
    add0_desc.SetSize(add0_size * sizeof(float));
    add0_tensor.SetTensorDesc(add0_desc);
    float* add0_data = nullptr;
    add0_data = new float[add0_size];
    *(add0_data + 0) = 1;
    add0_tensor.SetData((uint8_t*)add0_data, add0_size * sizeof(float));
    delete [] add0_data;

    ge::Tensor mul0_tensor;
    std::vector<int64_t> mul0_vec{224};
    ge::Shape mul0_shape(mul0_vec);
    ge::TensorDesc mul0_desc(mul0_shape, FORMAT_ND, DT_FLOAT);
    int64_t mul0_size = mul0_desc.GetShape().GetShapeSize();
    mul0_desc.SetSize(mul0_size * sizeof(float));
    mul0_tensor.SetTensorDesc(mul0_desc);
    float* mul0_data = nullptr;
    mul0_data = new float[mul0_size];
    for (int i=0; i<mul0_size; i++) {
        *(mul0_data + i) = 1;
    }
    mul0_tensor.SetData((uint8_t*)mul0_data, mul0_size * sizeof(float));
    delete [] mul0_data;

    ge::Tensor add1_tensor;
    std::vector<int64_t> add1_vec{224};
    ge::Shape add1_shape(add1_vec);
    ge::TensorDesc add1_desc(add1_shape, FORMAT_ND, DT_FLOAT);
    int64_t add1_size = add1_desc.GetShape().GetShapeSize();
    add1_desc.SetSize(add1_size * sizeof(float));
    add1_tensor.SetTensorDesc(add1_desc);
    float* add1_data = nullptr;
    add1_data = new float[add1_size];
    for (int i=0; i<add1_size; i++) {
        *(add1_data + i) = 1;
    }
    add1_tensor.SetData((uint8_t*)add1_data, add1_size * sizeof(float));
    delete [] add1_data;

    auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
    auto mul0_const_op = op::Constant().set_attr_value(mul0_tensor);
    auto add1_const_op = op::Constant().set_attr_value(add1_tensor);
    add0_const_op.update_output_desc_y(add0_desc);
    mul0_const_op.update_output_desc_y(mul0_desc);
    add1_const_op.update_output_desc_y(add1_desc);

    std::vector<int64_t> axes;
    axes.push_back(0);
    axes.push_back(1);
    auto data0 = op::Data().set_attr_index(0);
    auto mean0 = op::ReduceMeanD("mean0")
                        .set_input_x(data0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto squared0 = op::SquaredDifference("squared0")
                        .set_input_x1(mean0)
                        .set_input_x2(data0);
    auto sub0 = op::Sub("sub0")
                        .set_input_x1(data0)
                        .set_input_x2(mean0);
    auto mean1 = op::ReduceMeanD("mean1")
                        .set_input_x(squared0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto add0 = op::Add("add0")
                        .set_input_x1(add0_const_op)
                        .set_input_x2(mean1);
    auto rsqrt0 = op::Rsqrt("rsqrt0")
                        .set_input_x(add0);
    auto mul0 = op::Mul("mul0")
                        .set_input_x1(mul0_const_op)
                        .set_input_x2(rsqrt0);
    auto mul1 = op::Mul("mul1")
                        .set_input_x1(mul0)
                        .set_input_x2(sub0);
    auto add1 = op::Add("add1")
                        .set_input_x1(add1_const_op)
                        .set_input_x2(mul1);

    std::vector<int64_t> data0_vec{1, 3, 224, 224};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_ND,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    mean0.update_input_desc_x(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{add1};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 3, 224, 224};
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
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}
TEST_F(layer_norm_fusion_test, layer_norm_fusion_test_5) {
    ge::Graph graph("layer_norm_fusion_test_5");

    ge::Tensor add0_tensor;
    std::vector<int64_t> add0_vec{1};
    ge::Shape add0_shape(add0_vec);
    ge::TensorDesc add0_desc(add0_shape, FORMAT_ND, DT_FLOAT);
    int64_t add0_size = add0_desc.GetShape().GetShapeSize();
    add0_desc.SetSize(add0_size * sizeof(float));
    add0_tensor.SetTensorDesc(add0_desc);
    float* add0_data = nullptr;
    add0_data = new float[add0_size];
    *(add0_data + 0) = 1;
    add0_tensor.SetData((uint8_t*)add0_data, add0_size * sizeof(float));
    delete [] add0_data;

    ge::Tensor mul0_tensor;
    std::vector<int64_t> mul0_vec{224};
    ge::Shape mul0_shape(mul0_vec);
    ge::TensorDesc mul0_desc(mul0_shape, FORMAT_ND, DT_FLOAT);
    int64_t mul0_size = mul0_desc.GetShape().GetShapeSize();
    mul0_desc.SetSize(mul0_size * sizeof(float));
    mul0_tensor.SetTensorDesc(mul0_desc);
    float* mul0_data = nullptr;
    mul0_data = new float[mul0_size];
    for (int i=0; i<mul0_size; i++) {
        *(mul0_data + i) = 1;
    }
    mul0_tensor.SetData((uint8_t*)mul0_data, mul0_size * sizeof(float));
    delete [] mul0_data;

    ge::Tensor add1_tensor;
    std::vector<int64_t> add1_vec{224};
    ge::Shape add1_shape(add1_vec);
    ge::TensorDesc add1_desc(add1_shape, FORMAT_ND, DT_FLOAT);
    int64_t add1_size = add1_desc.GetShape().GetShapeSize();
    add1_desc.SetSize(add1_size * sizeof(float));
    add1_tensor.SetTensorDesc(add1_desc);
    float* add1_data = nullptr;
    add1_data = new float[add1_size];
    for (int i=0; i<add1_size; i++) {
        *(add1_data + i) = 1;
    }
    add1_tensor.SetData((uint8_t*)add1_data, add1_size * sizeof(float));
    delete [] add1_data;

    auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
    auto mul0_const_op = op::Constant().set_attr_value(mul0_tensor);
    auto add1_const_op = op::Constant().set_attr_value(add1_tensor);
    add0_const_op.update_output_desc_y(add0_desc);
    mul0_const_op.update_output_desc_y(mul0_desc);
    add1_const_op.update_output_desc_y(add1_desc);

    std::vector<int64_t> axes;
    axes.push_back(-1);
    auto data0 = op::Data().set_attr_index(0);
    auto mean0 = op::ReduceMeanD("mean0")
                        .set_input_x(data0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(false);
    auto squared0 = op::SquaredDifference("squared0")
                        .set_input_x1(mean0)
                        .set_input_x2(data0);
    auto sub0 = op::Sub("sub0")
                        .set_input_x1(data0)
                        .set_input_x2(mean0);
    auto mean1 = op::ReduceMeanD("mean1")
                        .set_input_x(squared0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto add0 = op::Add("add0")
                        .set_input_x1(add0_const_op)
                        .set_input_x2(mean1);
    auto rsqrt0 = op::Rsqrt("rsqrt0")
                        .set_input_x(add0);
    auto mul0 = op::Mul("mul0")
                        .set_input_x1(mul0_const_op)
                        .set_input_x2(rsqrt0);
    auto mul1 = op::Mul("mul1")
                        .set_input_x1(mul0)
                        .set_input_x2(sub0);
    auto add1 = op::Add("add1")
                        .set_input_x1(add1_const_op)
                        .set_input_x2(mul1);

    std::vector<int64_t> data0_vec{1, 3, 224, 224};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_ND,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    mean0.update_input_desc_x(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{add1};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 3, 224, 224};
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
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}
TEST_F(layer_norm_fusion_test, layer_norm_fusion_test_6) {
    ge::Graph graph("layer_norm_fusion_test_6");

    ge::Tensor add0_tensor;
    std::vector<int64_t> add0_vec{1};
    ge::Shape add0_shape(add0_vec);
    ge::TensorDesc add0_desc(add0_shape, FORMAT_ND, DT_FLOAT);
    int64_t add0_size = add0_desc.GetShape().GetShapeSize();
    add0_desc.SetSize(add0_size * sizeof(float));
    add0_tensor.SetTensorDesc(add0_desc);
    float* add0_data = nullptr;
    add0_data = new float[add0_size];
    *(add0_data + 0) = 1;
    add0_tensor.SetData((uint8_t*)add0_data, add0_size * sizeof(float));
    delete [] add0_data;

    ge::Tensor mul0_tensor;
    std::vector<int64_t> mul0_vec{224};
    ge::Shape mul0_shape(mul0_vec);
    ge::TensorDesc mul0_desc(mul0_shape, FORMAT_ND, DT_FLOAT);
    int64_t mul0_size = mul0_desc.GetShape().GetShapeSize();
    mul0_desc.SetSize(mul0_size * sizeof(float));
    mul0_tensor.SetTensorDesc(mul0_desc);
    float* mul0_data = nullptr;
    mul0_data = new float[mul0_size];
    for (int i=0; i<mul0_size; i++) {
        *(mul0_data + i) = 1;
    }
    mul0_tensor.SetData((uint8_t*)mul0_data, mul0_size * sizeof(float));
    delete [] mul0_data;

    ge::Tensor add1_tensor;
    std::vector<int64_t> add1_vec{224};
    ge::Shape add1_shape(add1_vec);
    ge::TensorDesc add1_desc(add1_shape, FORMAT_ND, DT_FLOAT);
    int64_t add1_size = add1_desc.GetShape().GetShapeSize();
    add1_desc.SetSize(add1_size * sizeof(float));
    add1_tensor.SetTensorDesc(add1_desc);
    float* add1_data = nullptr;
    add1_data = new float[add1_size];
    for (int i=0; i<add1_size; i++) {
        *(add1_data + i) = 1;
    }
    add1_tensor.SetData((uint8_t*)add1_data, add1_size * sizeof(float));
    delete [] add1_data;

    auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
    auto mul0_const_op = op::Constant().set_attr_value(mul0_tensor);
    auto add1_const_op = op::Constant().set_attr_value(add1_tensor);
    add0_const_op.update_output_desc_y(add0_desc);
    mul0_const_op.update_output_desc_y(mul0_desc);
    add1_const_op.update_output_desc_y(add1_desc);

    std::vector<int64_t> axes;
    axes.push_back(0);
    auto data0 = op::Data().set_attr_index(0);
    auto mean0 = op::ReduceMeanD("mean0")
                        .set_input_x(data0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto squared0 = op::SquaredDifference("squared0")
                        .set_input_x1(mean0)
                        .set_input_x2(data0);
    auto sub0 = op::Sub("sub0")
                        .set_input_x1(data0)
                        .set_input_x2(mean0);
    auto mean1 = op::ReduceMeanD("mean1")
                        .set_input_x(squared0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto add0 = op::Add("add0")
                        .set_input_x1(add0_const_op)
                        .set_input_x2(mean1);
    auto rsqrt0 = op::Rsqrt("rsqrt0")
                        .set_input_x(add0);
    auto mul0 = op::Mul("mul0")
                        .set_input_x1(mul0_const_op)
                        .set_input_x2(rsqrt0);
    auto mul1 = op::Mul("mul1")
                        .set_input_x1(mul0)
                        .set_input_x2(sub0);
    auto add1 = op::Add("add1")
                        .set_input_x1(add1_const_op)
                        .set_input_x2(mul1);

    std::vector<int64_t> data0_vec{1, 3, 224, 224};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_ND,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    mean0.update_input_desc_x(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{add1};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 3, 224, 224};
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
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}
TEST_F(layer_norm_fusion_test, layer_norm_fusion_test_7) {
    ge::Graph graph("layer_norm_fusion_test_7");

    ge::Tensor add0_tensor;
    std::vector<int64_t> add0_vec{1};
    ge::Shape add0_shape(add0_vec);
    ge::TensorDesc add0_desc(add0_shape, FORMAT_ND, DT_FLOAT);
    int64_t add0_size = add0_desc.GetShape().GetShapeSize();
    add0_desc.SetSize(add0_size * sizeof(float));
    add0_tensor.SetTensorDesc(add0_desc);
    float* add0_data = nullptr;
    add0_data = new float[add0_size];
    *(add0_data + 0) = 1;
    add0_tensor.SetData((uint8_t*)add0_data, add0_size * sizeof(float));
    delete [] add0_data;

    ge::Tensor mul0_tensor;
    std::vector<int64_t> mul0_vec{224};
    ge::Shape mul0_shape(mul0_vec);
    ge::TensorDesc mul0_desc(mul0_shape, FORMAT_ND, DT_FLOAT);
    int64_t mul0_size = mul0_desc.GetShape().GetShapeSize();
    mul0_desc.SetSize(mul0_size * sizeof(float));
    mul0_tensor.SetTensorDesc(mul0_desc);
    float* mul0_data = nullptr;
    mul0_data = new float[mul0_size];
    for (int i=0; i<mul0_size; i++) {
        *(mul0_data + i) = 1;
    }
    mul0_tensor.SetData((uint8_t*)mul0_data, mul0_size * sizeof(float));
    delete [] mul0_data;

    ge::Tensor add1_tensor;
    std::vector<int64_t> add1_vec{224};
    ge::Shape add1_shape(add1_vec);
    ge::TensorDesc add1_desc(add1_shape, FORMAT_ND, DT_FLOAT);
    int64_t add1_size = add1_desc.GetShape().GetShapeSize();
    add1_desc.SetSize(add1_size * sizeof(float));
    add1_tensor.SetTensorDesc(add1_desc);
    float* add1_data = nullptr;
    add1_data = new float[add1_size];
    for (int i=0; i<add1_size; i++) {
        *(add1_data + i) = 1;
    }
    add1_tensor.SetData((uint8_t*)add1_data, add1_size * sizeof(float));
    delete [] add1_data;

    auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
    auto mul0_const_op = op::Constant().set_attr_value(mul0_tensor);
    auto add1_const_op = op::Constant().set_attr_value(add1_tensor);
    add0_const_op.update_output_desc_y(add0_desc);
    mul0_const_op.update_output_desc_y(mul0_desc);
    add1_const_op.update_output_desc_y(add1_desc);

    std::vector<int64_t> axes;
    axes.push_back(-1);
    auto data0 = op::Data().set_attr_index(0);
    auto mean0 = op::ReduceMeanD("mean0")
                        .set_input_x(data0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto squared0 = op::SquaredDifference("squared0")
                        .set_input_x1(mean0)
                        .set_input_x2(data0);
    auto sub0 = op::Sub("sub0")
                        .set_input_x1(data0)
                        .set_input_x2(mean0);
    auto mean1 = op::ReduceMeanD("mean1")
                        .set_input_x(squared0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto add0 = op::Add("add0")
                        .set_input_x1(data0)
                        .set_input_x2(mean1);
    auto rsqrt0 = op::Rsqrt("rsqrt0")
                        .set_input_x(add0);
    auto mul0 = op::Mul("mul0")
                        .set_input_x1(mul0_const_op)
                        .set_input_x2(rsqrt0);
    auto mul1 = op::Mul("mul1")
                        .set_input_x1(mul0)
                        .set_input_x2(sub0);
    auto add1 = op::Add("add1")
                        .set_input_x1(add1_const_op)
                        .set_input_x2(mul1);

    std::vector<int64_t> data0_vec{1, 3, 224, 224};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_ND,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    mean0.update_input_desc_x(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{add1};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 3, 224, 224};
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
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}
TEST_F(layer_norm_fusion_test, layer_norm_fusion_test_8) {
    ge::Graph graph("layer_norm_fusion_test_8");

    ge::Tensor add0_tensor;
    std::vector<int64_t> add0_vec{1};
    ge::Shape add0_shape(add0_vec);
    ge::TensorDesc add0_desc(add0_shape, FORMAT_ND, DT_FLOAT);
    int64_t add0_size = add0_desc.GetShape().GetShapeSize();
    add0_desc.SetSize(add0_size * sizeof(float));
    add0_tensor.SetTensorDesc(add0_desc);
    float* add0_data = nullptr;
    add0_data = new float[add0_size];
    *(add0_data + 0) = 1;
    add0_tensor.SetData((uint8_t*)add0_data, add0_size * sizeof(float));
    delete [] add0_data;

    ge::Tensor mul0_tensor;
    std::vector<int64_t> mul0_vec{224};
    ge::Shape mul0_shape(mul0_vec);
    ge::TensorDesc mul0_desc(mul0_shape, FORMAT_ND, DT_FLOAT);
    int64_t mul0_size = mul0_desc.GetShape().GetShapeSize();
    mul0_desc.SetSize(mul0_size * sizeof(float));
    mul0_tensor.SetTensorDesc(mul0_desc);
    float* mul0_data = nullptr;
    mul0_data = new float[mul0_size];
    for (int i=0; i<mul0_size; i++) {
        *(mul0_data + i) = 1;
    }
    mul0_tensor.SetData((uint8_t*)mul0_data, mul0_size * sizeof(float));
    delete [] mul0_data;

    ge::Tensor add1_tensor;
    std::vector<int64_t> add1_vec{224};
    ge::Shape add1_shape(add1_vec);
    ge::TensorDesc add1_desc(add1_shape, FORMAT_ND, DT_FLOAT);
    int64_t add1_size = add1_desc.GetShape().GetShapeSize();
    add1_desc.SetSize(add1_size * sizeof(float));
    add1_tensor.SetTensorDesc(add1_desc);
    float* add1_data = nullptr;
    add1_data = new float[add1_size];
    for (int i=0; i<add1_size; i++) {
        *(add1_data + i) = 1;
    }
    add1_tensor.SetData((uint8_t*)add1_data, add1_size * sizeof(float));
    delete [] add1_data;

    auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
    auto mul0_const_op = op::Constant().set_attr_value(mul0_tensor);
    auto add1_const_op = op::Constant().set_attr_value(add1_tensor);
    add0_const_op.update_output_desc_y(add0_desc);
    mul0_const_op.update_output_desc_y(mul0_desc);
    add1_const_op.update_output_desc_y(add1_desc);

    std::vector<int64_t> axes;
    axes.push_back(-1);
    auto data0 = op::Data().set_attr_index(0);
    auto mean0 = op::ReduceMeanD("mean0")
                        .set_input_x(data0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto squared0 = op::SquaredDifference("squared0")
                        .set_input_x1(mean0)
                        .set_input_x2(data0);
    auto sub0 = op::Sub("sub0")
                        .set_input_x1(data0)
                        .set_input_x2(mean0);
    auto mean1 = op::ReduceMeanD("mean1")
                        .set_input_x(squared0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto add0 = op::Add("add0")
                        .set_input_x1(add0_const_op)
                        .set_input_x2(mean1);
    auto rsqrt0 = op::Rsqrt("rsqrt0")
                        .set_input_x(add0);
    auto mul0 = op::Mul("mul0")
                        .set_input_x1(data0)
                        .set_input_x2(rsqrt0);
    auto mul1 = op::Mul("mul1")
                        .set_input_x1(mul0)
                        .set_input_x2(sub0);
    auto add1 = op::Add("add1")
                        .set_input_x1(add1_const_op)
                        .set_input_x2(mul1);

    std::vector<int64_t> data0_vec{1, 3, 224, 224};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_ND,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    mean0.update_input_desc_x(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{add1};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 3, 224, 224};
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
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}
TEST_F(layer_norm_fusion_test, layer_norm_fusion_test_9) {
    ge::Graph graph("layer_norm_fusion_test_9");

    ge::Tensor add0_tensor;
    std::vector<int64_t> add0_vec{1};
    ge::Shape add0_shape(add0_vec);
    ge::TensorDesc add0_desc(add0_shape, FORMAT_ND, DT_FLOAT);
    int64_t add0_size = add0_desc.GetShape().GetShapeSize();
    add0_desc.SetSize(add0_size * sizeof(float));
    add0_tensor.SetTensorDesc(add0_desc);
    float* add0_data = nullptr;
    add0_data = new float[add0_size];
    *(add0_data + 0) = 1;
    add0_tensor.SetData((uint8_t*)add0_data, add0_size * sizeof(float));
    delete [] add0_data;

    ge::Tensor mul0_tensor;
    std::vector<int64_t> mul0_vec{224};
    ge::Shape mul0_shape(mul0_vec);
    ge::TensorDesc mul0_desc(mul0_shape, FORMAT_ND, DT_FLOAT);
    int64_t mul0_size = mul0_desc.GetShape().GetShapeSize();
    mul0_desc.SetSize(mul0_size * sizeof(float));
    mul0_tensor.SetTensorDesc(mul0_desc);
    float* mul0_data = nullptr;
    mul0_data = new float[mul0_size];
    for (int i=0; i<mul0_size; i++) {
        *(mul0_data + i) = 1;
    }
    mul0_tensor.SetData((uint8_t*)mul0_data, mul0_size * sizeof(float));
    delete [] mul0_data;

    ge::Tensor add1_tensor;
    std::vector<int64_t> add1_vec{224};
    ge::Shape add1_shape(add1_vec);
    ge::TensorDesc add1_desc(add1_shape, FORMAT_ND, DT_FLOAT);
    int64_t add1_size = add1_desc.GetShape().GetShapeSize();
    add1_desc.SetSize(add1_size * sizeof(float));
    add1_tensor.SetTensorDesc(add1_desc);
    float* add1_data = nullptr;
    add1_data = new float[add1_size];
    for (int i=0; i<add1_size; i++) {
        *(add1_data + i) = 1;
    }
    add1_tensor.SetData((uint8_t*)add1_data, add1_size * sizeof(float));
    delete [] add1_data;

    auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
    auto mul0_const_op = op::Constant().set_attr_value(mul0_tensor);
    auto add1_const_op = op::Constant().set_attr_value(add1_tensor);
    add0_const_op.update_output_desc_y(add0_desc);
    mul0_const_op.update_output_desc_y(mul0_desc);
    add1_const_op.update_output_desc_y(add1_desc);

    std::vector<int64_t> axes;
    axes.push_back(-1);
    auto data0 = op::Data().set_attr_index(0);
    auto mean0 = op::ReduceMeanD("mean0")
                        .set_input_x(data0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto squared0 = op::SquaredDifference("squared0")
                        .set_input_x1(mean0)
                        .set_input_x2(data0);
    auto sub0 = op::Sub("sub0")
                        .set_input_x1(data0)
                        .set_input_x2(mean0);
    auto mean1 = op::ReduceMeanD("mean1")
                        .set_input_x(squared0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto add0 = op::Add("add0")
                        .set_input_x1(add0_const_op)
                        .set_input_x2(mean1);
    auto rsqrt0 = op::Rsqrt("rsqrt0")
                        .set_input_x(add0);
    auto mul0 = op::Mul("mul0")
                        .set_input_x1(mul0_const_op)
                        .set_input_x2(rsqrt0);
    auto mul1 = op::Mul("mul1")
                        .set_input_x1(mul0)
                        .set_input_x2(sub0);
    auto add1 = op::Add("add1")
                        .set_input_x1(data0)
                        .set_input_x2(mul1);

    std::vector<int64_t> data0_vec{1, 3, 224, 224};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_ND,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    mean0.update_input_desc_x(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{add1};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 3, 224, 224};
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
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}
TEST_F(layer_norm_fusion_test, layer_norm_fusion_test_10) {
    ge::Graph graph("layer_norm_fusion_test_10");

    ge::Tensor add0_tensor;
    std::vector<int64_t> add0_vec{224};
    ge::Shape add0_shape(add0_vec);
    ge::TensorDesc add0_desc(add0_shape, FORMAT_ND, DT_FLOAT);
    int64_t add0_size = add0_desc.GetShape().GetShapeSize();
    add0_desc.SetSize(add0_size * sizeof(float));
    add0_tensor.SetTensorDesc(add0_desc);
    float* add0_data = nullptr;
    add0_data = new float[add0_size];
    for (int i=0; i<add0_size; i++) {
        *(add0_data + i) = 1;
    }
    add0_tensor.SetData((uint8_t*)add0_data, add0_size * sizeof(float));
    delete [] add0_data;

    ge::Tensor mul0_tensor;
    std::vector<int64_t> mul0_vec{224};
    ge::Shape mul0_shape(mul0_vec);
    ge::TensorDesc mul0_desc(mul0_shape, FORMAT_ND, DT_FLOAT);
    int64_t mul0_size = mul0_desc.GetShape().GetShapeSize();
    mul0_desc.SetSize(mul0_size * sizeof(float));
    mul0_tensor.SetTensorDesc(mul0_desc);
    float* mul0_data = nullptr;
    mul0_data = new float[mul0_size];
    for (int i=0; i<mul0_size; i++) {
        *(mul0_data + i) = 1;
    }
    mul0_tensor.SetData((uint8_t*)mul0_data, mul0_size * sizeof(float));
    delete [] mul0_data;

    ge::Tensor add1_tensor;
    std::vector<int64_t> add1_vec{224};
    ge::Shape add1_shape(add1_vec);
    ge::TensorDesc add1_desc(add1_shape, FORMAT_ND, DT_FLOAT);
    int64_t add1_size = add1_desc.GetShape().GetShapeSize();
    add1_desc.SetSize(add1_size * sizeof(float));
    add1_tensor.SetTensorDesc(add1_desc);
    float* add1_data = nullptr;
    add1_data = new float[add1_size];
    for (int i=0; i<add1_size; i++) {
        *(add1_data + i) = 1;
    }
    add1_tensor.SetData((uint8_t*)add1_data, add1_size * sizeof(float));
    delete [] add1_data;

    auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
    auto mul0_const_op = op::Constant().set_attr_value(mul0_tensor);
    auto add1_const_op = op::Constant().set_attr_value(add1_tensor);
    add0_const_op.update_output_desc_y(add0_desc);
    mul0_const_op.update_output_desc_y(mul0_desc);
    add1_const_op.update_output_desc_y(add1_desc);

    std::vector<int64_t> axes;
    axes.push_back(-1);
    auto data0 = op::Data().set_attr_index(0);
    auto mean0 = op::ReduceMeanD("mean0")
                        .set_input_x(data0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto squared0 = op::SquaredDifference("squared0")
                        .set_input_x1(mean0)
                        .set_input_x2(data0);
    auto sub0 = op::Sub("sub0")
                        .set_input_x1(data0)
                        .set_input_x2(mean0);
    auto mean1 = op::ReduceMeanD("mean1")
                        .set_input_x(squared0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto add0 = op::Add("add0")
                        .set_input_x1(add0_const_op)
                        .set_input_x2(mean1);
    auto rsqrt0 = op::Rsqrt("rsqrt0")
                        .set_input_x(add0);
    auto mul0 = op::Mul("mul0")
                        .set_input_x1(mul0_const_op)
                        .set_input_x2(rsqrt0);
    auto mul1 = op::Mul("mul1")
                        .set_input_x1(mul0)
                        .set_input_x2(sub0);
    auto add1 = op::Add("add1")
                        .set_input_x1(add1_const_op)
                        .set_input_x2(mul1);

    std::vector<int64_t> data0_vec{1, 3, 224, 224};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_ND,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    mean0.update_input_desc_x(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{add1};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 3, 224, 224};
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
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}
TEST_F(layer_norm_fusion_test, layer_norm_fusion_test_11) {
    ge::Graph graph("layer_norm_fusion_test_11");

    ge::Tensor add0_tensor;
    std::vector<int64_t> add0_vec{1};
    ge::Shape add0_shape(add0_vec);
    ge::TensorDesc add0_desc(add0_shape, FORMAT_ND, DT_FLOAT);
    int64_t add0_size = add0_desc.GetShape().GetShapeSize();
    add0_desc.SetSize(add0_size * sizeof(float));
    add0_tensor.SetTensorDesc(add0_desc);
    float* add0_data = nullptr;
    add0_data = new float[add0_size];
    for (int i=0; i<add0_size; i++) {
        *(add0_data + i) = 1;
    }
    add0_tensor.SetData((uint8_t*)add0_data, add0_size * sizeof(float));
    delete [] add0_data;

    ge::Tensor mul0_tensor;
    std::vector<int64_t> mul0_vec{225};
    ge::Shape mul0_shape(mul0_vec);
    ge::TensorDesc mul0_desc(mul0_shape, FORMAT_ND, DT_FLOAT);
    int64_t mul0_size = mul0_desc.GetShape().GetShapeSize();
    mul0_desc.SetSize(mul0_size * sizeof(float));
    mul0_tensor.SetTensorDesc(mul0_desc);
    float* mul0_data = nullptr;
    mul0_data = new float[mul0_size];
    for (int i=0; i<mul0_size; i++) {
        *(mul0_data + i) = 1;
    }
    mul0_tensor.SetData((uint8_t*)mul0_data, mul0_size * sizeof(float));
    delete [] mul0_data;

    ge::Tensor add1_tensor;
    std::vector<int64_t> add1_vec{224};
    ge::Shape add1_shape(add1_vec);
    ge::TensorDesc add1_desc(add1_shape, FORMAT_ND, DT_FLOAT);
    int64_t add1_size = add1_desc.GetShape().GetShapeSize();
    add1_desc.SetSize(add1_size * sizeof(float));
    add1_tensor.SetTensorDesc(add1_desc);
    float* add1_data = nullptr;
    add1_data = new float[add1_size];
    for (int i=0; i<add1_size; i++) {
        *(add1_data + i) = 1;
    }
    add1_tensor.SetData((uint8_t*)add1_data, add1_size * sizeof(float));
    delete [] add1_data;

    auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
    auto mul0_const_op = op::Constant().set_attr_value(mul0_tensor);
    auto add1_const_op = op::Constant().set_attr_value(add1_tensor);
    add0_const_op.update_output_desc_y(add0_desc);
    mul0_const_op.update_output_desc_y(mul0_desc);
    add1_const_op.update_output_desc_y(add1_desc);

    std::vector<int64_t> axes;
    axes.push_back(-1);
    auto data0 = op::Data().set_attr_index(0);
    auto mean0 = op::ReduceMeanD("mean0")
                        .set_input_x(data0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto squared0 = op::SquaredDifference("squared0")
                        .set_input_x1(mean0)
                        .set_input_x2(data0);
    auto sub0 = op::Sub("sub0")
                        .set_input_x1(data0)
                        .set_input_x2(mean0);
    auto mean1 = op::ReduceMeanD("mean1")
                        .set_input_x(squared0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto add0 = op::Add("add0")
                        .set_input_x1(add0_const_op)
                        .set_input_x2(mean1);
    auto rsqrt0 = op::Rsqrt("rsqrt0")
                        .set_input_x(add0);
    auto mul0 = op::Mul("mul0")
                        .set_input_x1(mul0_const_op)
                        .set_input_x2(rsqrt0);
    auto mul1 = op::Mul("mul1")
                        .set_input_x1(mul0)
                        .set_input_x2(sub0);
    auto add1 = op::Add("add1")
                        .set_input_x1(add1_const_op)
                        .set_input_x2(mul1);

    std::vector<int64_t> data0_vec{1, 3, 224, 224};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_ND,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    mean0.update_input_desc_x(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{add1};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 3, 224, 224};
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
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}
TEST_F(layer_norm_fusion_test, layer_norm_fusion_test_12) {
    ge::Graph graph("layer_norm_fusion_test_12");

    ge::Tensor add0_tensor;
    std::vector<int64_t> add0_vec{1};
    ge::Shape add0_shape(add0_vec);
    ge::TensorDesc add0_desc(add0_shape, FORMAT_ND, DT_FLOAT);
    int64_t add0_size = add0_desc.GetShape().GetShapeSize();
    add0_desc.SetSize(add0_size * sizeof(float));
    add0_tensor.SetTensorDesc(add0_desc);
    float* add0_data = nullptr;
    add0_data = new float[add0_size];
    for (int i=0; i<add0_size; i++) {
        *(add0_data + i) = 1;
    }
    add0_tensor.SetData((uint8_t*)add0_data, add0_size * sizeof(float));
    delete [] add0_data;

    ge::Tensor mul0_tensor;
    std::vector<int64_t> mul0_vec{224};
    ge::Shape mul0_shape(mul0_vec);
    ge::TensorDesc mul0_desc(mul0_shape, FORMAT_ND, DT_FLOAT);
    int64_t mul0_size = mul0_desc.GetShape().GetShapeSize();
    mul0_desc.SetSize(mul0_size * sizeof(float));
    mul0_tensor.SetTensorDesc(mul0_desc);
    float* mul0_data = nullptr;
    mul0_data = new float[mul0_size];
    for (int i=0; i<mul0_size; i++) {
        *(mul0_data + i) = 1;
    }
    mul0_tensor.SetData((uint8_t*)mul0_data, mul0_size * sizeof(float));
    delete [] mul0_data;

    ge::Tensor add1_tensor;
    std::vector<int64_t> add1_vec{225};
    ge::Shape add1_shape(add1_vec);
    ge::TensorDesc add1_desc(add1_shape, FORMAT_ND, DT_FLOAT);
    int64_t add1_size = add1_desc.GetShape().GetShapeSize();
    add1_desc.SetSize(add1_size * sizeof(float));
    add1_tensor.SetTensorDesc(add1_desc);
    float* add1_data = nullptr;
    add1_data = new float[add1_size];
    for (int i=0; i<add1_size; i++) {
        *(add1_data + i) = 1;
    }
    add1_tensor.SetData((uint8_t*)add1_data, add1_size * sizeof(float));
    delete [] add1_data;

    auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
    auto mul0_const_op = op::Constant().set_attr_value(mul0_tensor);
    auto add1_const_op = op::Constant().set_attr_value(add1_tensor);
    add0_const_op.update_output_desc_y(add0_desc);
    mul0_const_op.update_output_desc_y(mul0_desc);
    add1_const_op.update_output_desc_y(add1_desc);

    std::vector<int64_t> axes;
    axes.push_back(-1);
    auto data0 = op::Data().set_attr_index(0);
    auto mean0 = op::ReduceMeanD("mean0")
                        .set_input_x(data0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto squared0 = op::SquaredDifference("squared0")
                        .set_input_x1(mean0)
                        .set_input_x2(data0);
    auto sub0 = op::Sub("sub0")
                        .set_input_x1(data0)
                        .set_input_x2(mean0);
    auto mean1 = op::ReduceMeanD("mean1")
                        .set_input_x(squared0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto add0 = op::Add("add0")
                        .set_input_x1(add0_const_op)
                        .set_input_x2(mean1);
    auto rsqrt0 = op::Rsqrt("rsqrt0")
                        .set_input_x(add0);
    auto mul0 = op::Mul("mul0")
                        .set_input_x1(mul0_const_op)
                        .set_input_x2(rsqrt0);
    auto mul1 = op::Mul("mul1")
                        .set_input_x1(mul0)
                        .set_input_x2(sub0);
    auto add1 = op::Add("add1")
                        .set_input_x1(add1_const_op)
                        .set_input_x2(mul1);

    std::vector<int64_t> data0_vec{1, 3, 224, 224};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_ND,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    mean0.update_input_desc_x(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{add1};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 3, 224, 224};
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
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}
TEST_F(layer_norm_fusion_test, layer_norm_fusion_test_13) {
    ge::Graph graph("layer_norm_fusion_test_13");

    ge::Tensor add0_tensor;
    std::vector<int64_t> add0_vec{1};
    ge::Shape add0_shape(add0_vec);
    ge::TensorDesc add0_desc(add0_shape, FORMAT_ND, DT_FLOAT);
    int64_t add0_size = add0_desc.GetShape().GetShapeSize();
    add0_desc.SetSize(add0_size * sizeof(float));
    add0_tensor.SetTensorDesc(add0_desc);
    float* add0_data = nullptr;
    add0_data = new float[add0_size];
    for (int i=0; i<add0_size; i++) {
        *(add0_data + i) = 1;
    }
    add0_tensor.SetData((uint8_t*)add0_data, add0_size * sizeof(float));
    delete [] add0_data;

    ge::Tensor mul0_tensor;
    std::vector<int64_t> mul0_vec{1,3,224,224};
    ge::Shape mul0_shape(mul0_vec);
    ge::TensorDesc mul0_desc(mul0_shape, FORMAT_ND, DT_FLOAT);
    int64_t mul0_size = mul0_desc.GetShape().GetShapeSize();
    mul0_desc.SetSize(mul0_size * sizeof(float));
    mul0_tensor.SetTensorDesc(mul0_desc);
    float* mul0_data = nullptr;
    mul0_data = new float[mul0_size];
    for (int i=0; i<mul0_size; i++) {
        *(mul0_data + i) = 1;
    }
    mul0_tensor.SetData((uint8_t*)mul0_data, mul0_size * sizeof(float));
    delete [] mul0_data;

    ge::Tensor add1_tensor;
    std::vector<int64_t> add1_vec{224};
    ge::Shape add1_shape(add1_vec);
    ge::TensorDesc add1_desc(add1_shape, FORMAT_ND, DT_FLOAT);
    int64_t add1_size = add1_desc.GetShape().GetShapeSize();
    add1_desc.SetSize(add1_size * sizeof(float));
    add1_tensor.SetTensorDesc(add1_desc);
    float* add1_data = nullptr;
    add1_data = new float[add1_size];
    for (int i=0; i<add1_size; i++) {
        *(add1_data + i) = 1;
    }
    add1_tensor.SetData((uint8_t*)add1_data, add1_size * sizeof(float));
    delete [] add1_data;

    auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
    auto mul0_const_op = op::Constant().set_attr_value(mul0_tensor);
    auto add1_const_op = op::Constant().set_attr_value(add1_tensor);
    add0_const_op.update_output_desc_y(add0_desc);
    mul0_const_op.update_output_desc_y(mul0_desc);
    add1_const_op.update_output_desc_y(add1_desc);

    std::vector<int64_t> axes;
    axes.push_back(-1);
    auto data0 = op::Data().set_attr_index(0);
    auto mean0 = op::ReduceMeanD("mean0")
                        .set_input_x(data0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto squared0 = op::SquaredDifference("squared0")
                        .set_input_x1(mean0)
                        .set_input_x2(data0);
    auto sub0 = op::Sub("sub0")
                        .set_input_x1(data0)
                        .set_input_x2(mean0);
    auto mean1 = op::ReduceMeanD("mean1")
                        .set_input_x(squared0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto add0 = op::Add("add0")
                        .set_input_x1(add0_const_op)
                        .set_input_x2(mean1);
    auto rsqrt0 = op::Rsqrt("rsqrt0")
                        .set_input_x(add0);
    auto mul0 = op::Mul("mul0")
                        .set_input_x1(mul0_const_op)
                        .set_input_x2(rsqrt0);
    auto mul1 = op::Mul("mul1")
                        .set_input_x1(mul0)
                        .set_input_x2(sub0);
    auto add1 = op::Add("add1")
                        .set_input_x1(add1_const_op)
                        .set_input_x2(mul1);

    std::vector<int64_t> data0_vec{1, 3, 224, 224};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_ND,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    mean0.update_input_desc_x(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{add1};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 3, 224, 224};
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
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}
TEST_F(layer_norm_fusion_test, layer_norm_fusion_test_14) {
    ge::Graph graph("layer_norm_fusion_test_14");

    ge::Tensor add0_tensor;
    std::vector<int64_t> add0_vec{1};
    ge::Shape add0_shape(add0_vec);
    ge::TensorDesc add0_desc(add0_shape, FORMAT_ND, DT_FLOAT);
    int64_t add0_size = add0_desc.GetShape().GetShapeSize();
    add0_desc.SetSize(add0_size * sizeof(float));
    add0_tensor.SetTensorDesc(add0_desc);
    float* add0_data = nullptr;
    add0_data = new float[add0_size];
    for (int i=0; i<add0_size; i++) {
        *(add0_data + i) = 1;
    }
    add0_tensor.SetData((uint8_t*)add0_data, add0_size * sizeof(float));
    delete [] add0_data;

    ge::Tensor mul0_tensor;
    std::vector<int64_t> mul0_vec{224};
    ge::Shape mul0_shape(mul0_vec);
    ge::TensorDesc mul0_desc(mul0_shape, FORMAT_ND, DT_FLOAT);
    int64_t mul0_size = mul0_desc.GetShape().GetShapeSize();
    mul0_desc.SetSize(mul0_size * sizeof(float));
    mul0_tensor.SetTensorDesc(mul0_desc);
    float* mul0_data = nullptr;
    mul0_data = new float[mul0_size];
    for (int i=0; i<mul0_size; i++) {
        *(mul0_data + i) = 1;
    }
    mul0_tensor.SetData((uint8_t*)mul0_data, mul0_size * sizeof(float));
    delete [] mul0_data;

    ge::Tensor add1_tensor;
    std::vector<int64_t> add1_vec{1,3,224,224};
    ge::Shape add1_shape(add1_vec);
    ge::TensorDesc add1_desc(add1_shape, FORMAT_ND, DT_FLOAT);
    int64_t add1_size = add1_desc.GetShape().GetShapeSize();
    add1_desc.SetSize(add1_size * sizeof(float));
    add1_tensor.SetTensorDesc(add1_desc);
    float* add1_data = nullptr;
    add1_data = new float[add1_size];
    for (int i=0; i<add1_size; i++) {
        *(add1_data + i) = 1;
    }
    add1_tensor.SetData((uint8_t*)add1_data, add1_size * sizeof(float));
    delete [] add1_data;

    auto add0_const_op = op::Constant().set_attr_value(add0_tensor);
    auto mul0_const_op = op::Constant().set_attr_value(mul0_tensor);
    auto add1_const_op = op::Constant().set_attr_value(add1_tensor);
    add0_const_op.update_output_desc_y(add0_desc);
    mul0_const_op.update_output_desc_y(mul0_desc);
    add1_const_op.update_output_desc_y(add1_desc);

    std::vector<int64_t> axes;
    axes.push_back(-1);
    auto data0 = op::Data().set_attr_index(0);
    auto mean0 = op::ReduceMeanD("mean0")
                        .set_input_x(data0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto squared0 = op::SquaredDifference("squared0")
                        .set_input_x1(mean0)
                        .set_input_x2(data0);
    auto sub0 = op::Sub("sub0")
                        .set_input_x1(data0)
                        .set_input_x2(mean0);
    auto mean1 = op::ReduceMeanD("mean1")
                        .set_input_x(squared0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto add0 = op::Add("add0")
                        .set_input_x1(add0_const_op)
                        .set_input_x2(mean1);
    auto rsqrt0 = op::Rsqrt("rsqrt0")
                        .set_input_x(add0);
    auto mul0 = op::Mul("mul0")
                        .set_input_x1(mul0_const_op)
                        .set_input_x2(rsqrt0);
    auto mul1 = op::Mul("mul1")
                        .set_input_x1(mul0)
                        .set_input_x2(sub0);
    auto add1 = op::Add("add1")
                        .set_input_x1(add1_const_op)
                        .set_input_x2(mul1);

    std::vector<int64_t> data0_vec{1, 3, 224, 224};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_ND,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    mean0.update_input_desc_x(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{add1};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("LayerNormFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 3, 224, 224};
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
    EXPECT_EQ(findOp, false);
    EXPECT_EQ(shapeMatch, false);
}





