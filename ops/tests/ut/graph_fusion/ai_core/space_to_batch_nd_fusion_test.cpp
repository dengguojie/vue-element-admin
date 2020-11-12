#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "transformation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class space_to_batch_nd_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "space_to_batch_nd_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "space_to_batch_nd_fusion_test TearDown" << std::endl;
    }
};

TEST_F(space_to_batch_nd_fusion_test, space_to_batch_nd_fusion_test_1) {
    ge::Graph graph("space_to_batch_nd_fusion_test_1");

    ge::Tensor block_tensor;
    std::vector<int64_t> block_vec{2};
    ge::Shape block_shape(block_vec);
    ge::TensorDesc block_desc(block_shape, FORMAT_NHWC, DT_INT64);
    int64_t block_size = block_desc.GetShape().GetShapeSize();
    block_desc.SetSize(block_size * sizeof(int64_t));
    block_tensor.SetTensorDesc(block_desc);
    int64_t* block_data = nullptr;
    block_data = new int64_t[block_size];
    *(block_data + 0) = 1;
    *(block_data + 1) = 1;
    block_tensor.SetData((uint8_t*)block_data, block_size * sizeof(int64_t));
    delete [] block_data;

    ge::Tensor paddings_tensor;
    std::vector<int64_t> paddings_vec{4};
    ge::Shape paddings_shape(paddings_vec);
    ge::TensorDesc paddings_desc(paddings_shape, FORMAT_NHWC, DT_INT32);
    int64_t paddings_size = paddings_desc.GetShape().GetShapeSize();
    paddings_desc.SetSize(paddings_size * sizeof(int32_t));
    paddings_tensor.SetTensorDesc(paddings_desc);
    int32_t* paddings_data = nullptr;
    paddings_data = new int32_t[paddings_size];
    *(paddings_data + 0) = 0;
    *(paddings_data + 1) = 0;
    *(paddings_data + 2) = 0;
    *(paddings_data + 3) = 0;
    paddings_tensor.SetData((uint8_t*)paddings_data, paddings_size * sizeof(int32_t));
    delete [] paddings_data;

    auto block = op::Constant().set_attr_value(block_tensor);
    auto paddings = op::Constant().set_attr_value(paddings_tensor);
    auto data0 = op::Data().set_attr_index(0);
    auto space_op = op::SpaceToBatchND("space_op")
                        .set_input_x(data0)
                        .set_input_block_shape(block)
                        .set_input_paddings(paddings);

    block.update_output_desc_y(block_desc);
    paddings.update_output_desc_y(paddings_desc);

    std::vector<int64_t> data0_vec{1, 2, 3, 4};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NHWC,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    space_op.update_input_desc_x(data0_desc);
    space_op.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{space_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "space_to_batch_nd_fusion_test_1_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrSpaceToBatchNdFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "space_to_batch_nd_fusion_test_1_after");

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 2, 3, 4};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SpaceToBatchNDD") {
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
TEST_F(space_to_batch_nd_fusion_test, space_to_batch_nd_fusion_test_2) {
    ge::Graph graph("space_to_batch_nd_fusion_test_2");

    ge::Tensor block_tensor;
    std::vector<int64_t> block_vec{3};
    ge::Shape block_shape(block_vec);
    ge::TensorDesc block_desc(block_shape, FORMAT_NCHW, DT_INT32);
    int64_t block_size = block_desc.GetShape().GetShapeSize();
    block_desc.SetSize(block_size * sizeof(int32_t));
    block_tensor.SetTensorDesc(block_desc);
    int32_t* block_data = nullptr;
    block_data = new int32_t[block_size];
    *(block_data + 0) = 1;
    *(block_data + 1) = 1;
    *(block_data + 2) = 1;
    block_tensor.SetData((uint8_t*)block_data, block_size * sizeof(int32_t));
    delete [] block_data;

    ge::Tensor paddings_tensor;
    std::vector<int64_t> paddings_vec{6};
    ge::Shape paddings_shape(paddings_vec);
    ge::TensorDesc paddings_desc(paddings_shape, FORMAT_NCHW, DT_INT32);
    int64_t paddings_size = paddings_desc.GetShape().GetShapeSize();
    paddings_desc.SetSize(paddings_size * sizeof(int32_t));
    paddings_tensor.SetTensorDesc(paddings_desc);
    int32_t* paddings_data = nullptr;
    paddings_data = new int32_t[paddings_size];
    *(paddings_data + 0) = 0;
    *(paddings_data + 1) = 0;
    *(paddings_data + 2) = 0;
    *(paddings_data + 3) = 0;
    *(paddings_data + 4) = 0;
    *(paddings_data + 5) = 0;
    paddings_tensor.SetData((uint8_t*)paddings_data, paddings_size * sizeof(int32_t));
    delete [] paddings_data;

    auto block = op::Constant().set_attr_value(block_tensor);
    auto paddings = op::Constant().set_attr_value(paddings_tensor);
    auto data0 = op::Data().set_attr_index(0);
    auto space_op = op::SpaceToBatchND("space_op")
                        .set_input_x(data0)
                        .set_input_block_shape(block)
                        .set_input_paddings(paddings);

    block.update_output_desc_y(block_desc);
    paddings.update_output_desc_y(paddings_desc);

    std::vector<int64_t> data0_vec{1, 2, 3, 4};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NCHW,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    space_op.update_input_desc_x(data0_desc);
    space_op.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{space_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "space_to_batch_nd_fusion_test_2_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrSpaceToBatchNdFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "space_to_batch_nd_fusion_test_2_after");

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 2, 3, 4};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SpaceToBatchNDD") {
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
TEST_F(space_to_batch_nd_fusion_test, space_to_batch_nd_fusion_test_3) {
    ge::Graph graph("space_to_batch_nd_fusion_test_3");

    ge::Tensor block_tensor;
    std::vector<int64_t> block_vec{2};
    ge::Shape block_shape(block_vec);
    ge::TensorDesc block_desc(block_shape, FORMAT_NHWC, DT_INT32);
    int64_t block_size = block_desc.GetShape().GetShapeSize();
    block_desc.SetSize(block_size * sizeof(int32_t));
    block_tensor.SetTensorDesc(block_desc);
    int32_t* block_data = nullptr;
    block_data = new int32_t[block_size];
    *(block_data + 0) = 1;
    *(block_data + 1) = 1;
    block_tensor.SetData((uint8_t*)block_data, block_size * sizeof(int32_t));
    delete [] block_data;

    ge::Tensor paddings_tensor;
    std::vector<int64_t> paddings_vec{4};
    ge::Shape paddings_shape(paddings_vec);
    ge::TensorDesc paddings_desc(paddings_shape, FORMAT_NHWC, DT_INT32);
    int64_t paddings_size = paddings_desc.GetShape().GetShapeSize();
    paddings_desc.SetSize(paddings_size * sizeof(int32_t));
    paddings_tensor.SetTensorDesc(paddings_desc);
    int32_t* paddings_data = nullptr;
    paddings_data = new int32_t[paddings_size];
    *(paddings_data + 0) = 0;
    *(paddings_data + 1) = 0;
    *(paddings_data + 2) = 0;
    *(paddings_data + 3) = 0;
    paddings_tensor.SetData((uint8_t*)paddings_data, paddings_size * sizeof(int32_t));
    delete [] paddings_data;

    auto block = op::Constant().set_attr_value(block_tensor);
    auto paddings = op::Constant().set_attr_value(paddings_tensor);
    auto data0 = op::Data().set_attr_index(0);
    auto space_op = op::SpaceToBatchND("space_op")
                        .set_input_x(data0)
                        .set_input_block_shape(block)
                        .set_input_paddings(paddings);

    block.update_output_desc_y(block_desc);
    paddings.update_output_desc_y(paddings_desc);

    std::vector<int64_t> data0_vec{1, 2, 3};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NHWC,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    space_op.update_input_desc_x(data0_desc);
    space_op.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{space_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "space_to_batch_nd_fusion_test_3_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrSpaceToBatchNdFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "space_to_batch_nd_fusion_test_3_after");

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 2, 3};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SpaceToBatchNDD") {
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
TEST_F(space_to_batch_nd_fusion_test, space_to_batch_nd_fusion_test_4) {
    ge::Graph graph("space_to_batch_nd_fusion_test_4");

    ge::Tensor block_tensor;
    std::vector<int64_t> block_vec{3};
    ge::Shape block_shape(block_vec);
    ge::TensorDesc block_desc(block_shape, FORMAT_NHWC, DT_INT32);
    int64_t block_size = block_desc.GetShape().GetShapeSize();
    block_desc.SetSize(block_size * sizeof(int32_t));
    block_tensor.SetTensorDesc(block_desc);
    int32_t* block_data = nullptr;
    block_data = new int32_t[block_size];
    *(block_data + 0) = 1;
    *(block_data + 1) = 1;
    *(block_data + 2) = 1;
    block_tensor.SetData((uint8_t*)block_data, block_size * sizeof(int32_t));
    delete [] block_data;

    ge::Tensor paddings_tensor;
    std::vector<int64_t> paddings_vec{4};
    ge::Shape paddings_shape(paddings_vec);
    ge::TensorDesc paddings_desc(paddings_shape, FORMAT_NHWC, DT_INT32);
    int64_t paddings_size = paddings_desc.GetShape().GetShapeSize();
    paddings_desc.SetSize(paddings_size * sizeof(int32_t));
    paddings_tensor.SetTensorDesc(paddings_desc);
    int32_t* paddings_data = nullptr;
    paddings_data = new int32_t[paddings_size];
    *(paddings_data + 0) = 0;
    *(paddings_data + 1) = 0;
    *(paddings_data + 2) = 0;
    *(paddings_data + 3) = 0;
    paddings_tensor.SetData((uint8_t*)paddings_data, paddings_size * sizeof(int32_t));
    delete [] paddings_data;

    auto block = op::Constant().set_attr_value(block_tensor);
    auto paddings = op::Constant().set_attr_value(paddings_tensor);
    auto data0 = op::Data().set_attr_index(0);
    auto space_op = op::SpaceToBatchND("space_op")
                        .set_input_x(data0)
                        .set_input_block_shape(block)
                        .set_input_paddings(paddings);

    block.update_output_desc_y(block_desc);
    paddings.update_output_desc_y(paddings_desc);

    std::vector<int64_t> data0_vec{1, 2, 3, 4};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NHWC,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    space_op.update_input_desc_x(data0_desc);
    space_op.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{space_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "space_to_batch_nd_fusion_test_1_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrSpaceToBatchNdFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "space_to_batch_nd_fusion_test_1_after");

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 2, 3, 4};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SpaceToBatchNDD") {
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
TEST_F(space_to_batch_nd_fusion_test, space_to_batch_nd_fusion_test_5) {
    ge::Graph graph("space_to_batch_nd_fusion_test_5");

    ge::Tensor block_tensor;
    std::vector<int64_t> block_vec{2};
    ge::Shape block_shape(block_vec);
    ge::TensorDesc block_desc(block_shape, FORMAT_NHWC, DT_INT32);
    int64_t block_size = block_desc.GetShape().GetShapeSize();
    block_desc.SetSize(block_size * sizeof(int32_t));
    block_tensor.SetTensorDesc(block_desc);
    int32_t* block_data = nullptr;
    block_data = new int32_t[block_size];
    *(block_data + 0) = 1;
    *(block_data + 1) = 1;
    block_tensor.SetData((uint8_t*)block_data, block_size * sizeof(int32_t));
    delete [] block_data;

    ge::Tensor paddings_tensor;
    std::vector<int64_t> paddings_vec{6};
    ge::Shape paddings_shape(paddings_vec);
    ge::TensorDesc paddings_desc(paddings_shape, FORMAT_NHWC, DT_INT32);
    int64_t paddings_size = paddings_desc.GetShape().GetShapeSize();
    paddings_desc.SetSize(paddings_size * sizeof(int32_t));
    paddings_tensor.SetTensorDesc(paddings_desc);
    int32_t* paddings_data = nullptr;
    paddings_data = new int32_t[paddings_size];
    *(paddings_data + 0) = 0;
    *(paddings_data + 1) = 0;
    *(paddings_data + 2) = 0;
    *(paddings_data + 3) = 0;
    *(paddings_data + 4) = 0;
    *(paddings_data + 5) = 0;
    paddings_tensor.SetData((uint8_t*)paddings_data, paddings_size * sizeof(int32_t));
    delete [] paddings_data;

    auto block = op::Constant().set_attr_value(block_tensor);
    auto paddings = op::Constant().set_attr_value(paddings_tensor);
    auto data0 = op::Data().set_attr_index(0);
    auto space_op = op::SpaceToBatchND("space_op")
                        .set_input_x(data0)
                        .set_input_block_shape(block)
                        .set_input_paddings(paddings);

    block.update_output_desc_y(block_desc);
    paddings.update_output_desc_y(paddings_desc);

    std::vector<int64_t> data0_vec{1, 2, 3, 4};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NHWC,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    space_op.update_input_desc_x(data0_desc);
    space_op.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{space_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "space_to_batch_nd_fusion_test_5_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrSpaceToBatchNdFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "space_to_batch_nd_fusion_test_5_after");

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 2, 3, 4};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SpaceToBatchNDD") {
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
TEST_F(space_to_batch_nd_fusion_test, space_to_batch_nd_fusion_test_6) {
    ge::Graph graph("space_to_batch_nd_fusion_test_6");

    ge::Tensor block_tensor;
    std::vector<int64_t> block_vec{2};
    ge::Shape block_shape(block_vec);
    ge::TensorDesc block_desc(block_shape, FORMAT_NCHW, DT_INT32);
    int64_t block_size = block_desc.GetShape().GetShapeSize();
    block_desc.SetSize(block_size * sizeof(int32_t));
    block_tensor.SetTensorDesc(block_desc);
    int32_t* block_data = nullptr;
    block_data = new int32_t[block_size];
    *(block_data + 0) = 1;
    *(block_data + 1) = 1;
    block_tensor.SetData((uint8_t*)block_data, block_size * sizeof(int32_t));
    delete [] block_data;

    ge::Tensor paddings_tensor;
    std::vector<int64_t> paddings_vec{6};
    ge::Shape paddings_shape(paddings_vec);
    ge::TensorDesc paddings_desc(paddings_shape, FORMAT_NCHW, DT_INT32);
    int64_t paddings_size = paddings_desc.GetShape().GetShapeSize();
    paddings_desc.SetSize(paddings_size * sizeof(int32_t));
    paddings_tensor.SetTensorDesc(paddings_desc);
    int32_t* paddings_data = nullptr;
    paddings_data = new int32_t[paddings_size];
    *(paddings_data + 0) = 0;
    *(paddings_data + 1) = 0;
    *(paddings_data + 2) = 0;
    *(paddings_data + 3) = 0;
    *(paddings_data + 4) = 0;
    *(paddings_data + 5) = 0;
    paddings_tensor.SetData((uint8_t*)paddings_data, paddings_size * sizeof(int32_t));
    delete [] paddings_data;

    auto block = op::Constant().set_attr_value(block_tensor);
    auto paddings = op::Constant().set_attr_value(paddings_tensor);
    auto data0 = op::Data().set_attr_index(0);
    auto space_op = op::SpaceToBatchND("space_op")
                        .set_input_x(data0)
                        .set_input_block_shape(block)
                        .set_input_paddings(paddings);

    block.update_output_desc_y(block_desc);
    paddings.update_output_desc_y(paddings_desc);

    std::vector<int64_t> data0_vec{1, 2, 3, 4};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NCHW,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    space_op.update_input_desc_x(data0_desc);
    space_op.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{space_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "space_to_batch_nd_fusion_test_6_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrSpaceToBatchNdFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "space_to_batch_nd_fusion_test_6_after");

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 2, 3, 4};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SpaceToBatchNDD") {
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
TEST_F(space_to_batch_nd_fusion_test, space_to_batch_nd_fusion_test_7) {
    ge::Graph graph("space_to_batch_nd_fusion_test_7");

    ge::Tensor block_tensor;
    std::vector<int64_t> block_vec{3};
    ge::Shape block_shape(block_vec);
    ge::TensorDesc block_desc(block_shape, FORMAT_NCHW, DT_INT32);
    int64_t block_size = block_desc.GetShape().GetShapeSize();
    block_desc.SetSize(block_size * sizeof(int32_t));
    block_tensor.SetTensorDesc(block_desc);
    int32_t* block_data = nullptr;
    block_data = new int32_t[block_size];
    *(block_data + 0) = 1;
    *(block_data + 1) = 1;
    *(block_data + 2) = 1;
    block_tensor.SetData((uint8_t*)block_data, block_size * sizeof(int32_t));
    delete [] block_data;

    ge::Tensor paddings_tensor;
    std::vector<int64_t> paddings_vec{4};
    ge::Shape paddings_shape(paddings_vec);
    ge::TensorDesc paddings_desc(paddings_shape, FORMAT_NCHW, DT_INT32);
    int64_t paddings_size = paddings_desc.GetShape().GetShapeSize();
    paddings_desc.SetSize(paddings_size * sizeof(int32_t));
    paddings_tensor.SetTensorDesc(paddings_desc);
    int32_t* paddings_data = nullptr;
    paddings_data = new int32_t[paddings_size];
    *(paddings_data + 0) = 0;
    *(paddings_data + 1) = 0;
    *(paddings_data + 2) = 0;
    *(paddings_data + 3) = 0;
    paddings_tensor.SetData((uint8_t*)paddings_data, paddings_size * sizeof(int32_t));
    delete [] paddings_data;

    auto block = op::Constant().set_attr_value(block_tensor);
    auto paddings = op::Constant().set_attr_value(paddings_tensor);
    auto data0 = op::Data().set_attr_index(0);
    auto space_op = op::SpaceToBatchND("space_op")
                        .set_input_x(data0)
                        .set_input_block_shape(block)
                        .set_input_paddings(paddings);

    block.update_output_desc_y(block_desc);
    paddings.update_output_desc_y(paddings_desc);

    std::vector<int64_t> data0_vec{1, 2, 3, 4};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NCHW,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    space_op.update_input_desc_x(data0_desc);
    space_op.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{space_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "space_to_batch_nd_fusion_test_7_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrSpaceToBatchNdFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "space_to_batch_nd_fusion_test_7_after");

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 2, 3, 4};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SpaceToBatchNDD") {
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
TEST_F(space_to_batch_nd_fusion_test, space_to_batch_nd_fusion_test_8) {
    ge::Graph graph("space_to_batch_nd_fusion_test_8");

    ge::Tensor block_tensor;
    std::vector<int64_t> block_vec{3};
    ge::Shape block_shape(block_vec);
    ge::TensorDesc block_desc(block_shape, FORMAT_NCHW, DT_INT32);
    int64_t block_size = block_desc.GetShape().GetShapeSize();
    block_desc.SetSize(block_size * sizeof(int32_t));
    block_tensor.SetTensorDesc(block_desc);
    int32_t* block_data = nullptr;
    block_data = new int32_t[block_size];
    *(block_data + 0) = 2;
    *(block_data + 1) = 1;
    *(block_data + 2) = 1;
    block_tensor.SetData((uint8_t*)block_data, block_size * sizeof(int32_t));
    delete [] block_data;

    ge::Tensor paddings_tensor;
    std::vector<int64_t> paddings_vec{6};
    ge::Shape paddings_shape(paddings_vec);
    ge::TensorDesc paddings_desc(paddings_shape, FORMAT_NCHW, DT_INT32);
    int64_t paddings_size = paddings_desc.GetShape().GetShapeSize();
    paddings_desc.SetSize(paddings_size * sizeof(int32_t));
    paddings_tensor.SetTensorDesc(paddings_desc);
    int32_t* paddings_data = nullptr;
    paddings_data = new int32_t[paddings_size];
    *(paddings_data + 0) = 0;
    *(paddings_data + 1) = 0;
    *(paddings_data + 2) = 0;
    *(paddings_data + 3) = 0;
    *(paddings_data + 4) = 0;
    *(paddings_data + 5) = 0;
    paddings_tensor.SetData((uint8_t*)paddings_data, paddings_size * sizeof(int32_t));
    delete [] paddings_data;

    auto block = op::Constant().set_attr_value(block_tensor);
    auto paddings = op::Constant().set_attr_value(paddings_tensor);
    auto data0 = op::Data().set_attr_index(0);
    auto space_op = op::SpaceToBatchND("space_op")
                        .set_input_x(data0)
                        .set_input_block_shape(block)
                        .set_input_paddings(paddings);

    block.update_output_desc_y(block_desc);
    paddings.update_output_desc_y(paddings_desc);

    std::vector<int64_t> data0_vec{1, 2, 3, 4};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NCHW,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    space_op.update_input_desc_x(data0_desc);
    space_op.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{space_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "space_to_batch_nd_fusion_test_8_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrSpaceToBatchNdFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "space_to_batch_nd_fusion_test_8_after");

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 2, 3, 4};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SpaceToBatchNDD") {
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
TEST_F(space_to_batch_nd_fusion_test, space_to_batch_nd_fusion_test_9) {
    ge::Graph graph("space_to_batch_nd_fusion_test_9");

    ge::Tensor block_tensor;
    std::vector<int64_t> block_vec{3};
    ge::Shape block_shape(block_vec);
    ge::TensorDesc block_desc(block_shape, FORMAT_NCHW, DT_INT32);
    int64_t block_size = block_desc.GetShape().GetShapeSize();
    block_desc.SetSize(block_size * sizeof(int32_t));
    block_tensor.SetTensorDesc(block_desc);
    int32_t* block_data = nullptr;
    block_data = new int32_t[block_size];
    *(block_data + 0) = 1;
    *(block_data + 1) = 1;
    *(block_data + 2) = 1;
    block_tensor.SetData((uint8_t*)block_data, block_size * sizeof(int32_t));
    delete [] block_data;

    ge::Tensor paddings_tensor;
    std::vector<int64_t> paddings_vec{6};
    ge::Shape paddings_shape(paddings_vec);
    ge::TensorDesc paddings_desc(paddings_shape, FORMAT_NCHW, DT_INT32);
    int64_t paddings_size = paddings_desc.GetShape().GetShapeSize();
    paddings_desc.SetSize(paddings_size * sizeof(int32_t));
    paddings_tensor.SetTensorDesc(paddings_desc);
    int32_t* paddings_data = nullptr;
    paddings_data = new int32_t[paddings_size];
    *(paddings_data + 0) = 1;
    *(paddings_data + 1) = 1;
    *(paddings_data + 2) = 0;
    *(paddings_data + 3) = 0;
    *(paddings_data + 4) = 0;
    *(paddings_data + 5) = 0;
    paddings_tensor.SetData((uint8_t*)paddings_data, paddings_size * sizeof(int32_t));
    delete [] paddings_data;

    auto block = op::Constant().set_attr_value(block_tensor);
    auto paddings = op::Constant().set_attr_value(paddings_tensor);
    auto data0 = op::Data().set_attr_index(0);
    auto space_op = op::SpaceToBatchND("space_op")
                        .set_input_x(data0)
                        .set_input_block_shape(block)
                        .set_input_paddings(paddings);

    block.update_output_desc_y(block_desc);
    paddings.update_output_desc_y(paddings_desc);

    std::vector<int64_t> data0_vec{1, 2, 3, 4};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NCHW,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    space_op.update_input_desc_x(data0_desc);
    space_op.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{space_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "space_to_batch_nd_fusion_test_9_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrSpaceToBatchNdFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "space_to_batch_nd_fusion_test_9_after");

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 2, 3, 4};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SpaceToBatchNDD") {
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