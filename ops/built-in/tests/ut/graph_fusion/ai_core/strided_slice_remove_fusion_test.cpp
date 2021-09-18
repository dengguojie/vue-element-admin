#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "selection_ops.h"
#include "gtest/gtest.h"

using namespace ge;
using namespace op;

class strided_slice_remove_fusion_test : public testing::Test {
  protected:
    static void SetUpTestCase()
    {
        std::cout << "strided_slice remove SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "strided_slice remove TearDown" << std::endl;
    }
};
TEST_F(strided_slice_remove_fusion_test, strided_slice_remove_fusion_test_1) {
    ge::Graph graph("strided_slice_remove_fusion_test_1");
    ge::Tensor begin_tensor;
    std::vector<int64_t> begin_vec{4};
    ge::Shape begin_shape(begin_vec);
    ge::TensorDesc begin_desc(begin_shape, FORMAT_ND, DT_INT32);
    int32_t begin_size = begin_desc.GetShape().GetShapeSize();
    begin_desc.SetSize(begin_size * sizeof(int32_t));
    begin_tensor.SetTensorDesc(begin_desc);
    int32_t *begin_data = nullptr;
    begin_data = new int32_t[begin_size];
    *(begin_data + 0) = 0;
    *(begin_data + 1) = 0;
    *(begin_data + 2) = 0;
    *(begin_data + 3) = 0;
    begin_tensor.SetData((uint8_t *)begin_data, begin_size * sizeof(int32_t));
    delete[] begin_data;

    ge::Tensor end_tensor;
    std::vector<int64_t> end_vec{4};
    ge::Shape end_shape(end_vec);
    ge::TensorDesc end_desc(end_shape, FORMAT_ND, DT_INT32);
    int32_t end_size = end_desc.GetShape().GetShapeSize();
    end_desc.SetSize(end_size * sizeof(int32_t));
    end_tensor.SetTensorDesc(end_desc);
    int32_t *end_data = nullptr;
    end_data = new int32_t[end_size];
    *(end_data + 0) = 1;
    *(end_data + 1) = 112;
    *(end_data + 2) = 60;
    *(end_data + 3) = 60;
    end_tensor.SetData((uint8_t *)end_data, end_size * sizeof(int32_t));
    delete[] end_data;

    ge::Tensor strides_tensor;
    std::vector<int64_t> strides_vec{4};
    ge::Shape strides_shape(strides_vec);
    ge::TensorDesc strides_desc(strides_shape, FORMAT_ND, DT_INT32);
    int32_t strides_size = strides_desc.GetShape().GetShapeSize();
    strides_desc.SetSize(strides_size * sizeof(int32_t));
    strides_tensor.SetTensorDesc(strides_desc);
    int32_t *strides_data = nullptr;
    strides_data = new int32_t[strides_size];
    *(strides_data + 0) = 1;
    *(strides_data + 1) = 1;
    *(strides_data + 2) = 1;
    *(strides_data + 3) = 1;
    strides_tensor.SetData((uint8_t *)strides_data, strides_size * sizeof(int32_t));
    delete[] strides_data;

    auto begin = op::Constant().set_attr_value(begin_tensor);
    auto end = op::Constant().set_attr_value(end_tensor);
    auto strides = op::Constant().set_attr_value(strides_tensor);
    auto data_x = op::Data().set_attr_index(0);
    auto strided_slice_op = op::StridedSlice("strided_slice_op")
                                .set_input_x(data_x)
                                .set_input_begin(begin)
                                .set_input_end(end)
                                .set_input_strides(strides)
                                .set_attr_begin_mask(0)
                                .set_attr_end_mask(0)
                                .set_attr_ellipsis_mask(0)
                                .set_attr_new_axis_mask(0)
                                .set_attr_shrink_axis_mask(0);

    begin.update_output_desc_y(begin_desc);
    end.update_output_desc_y(end_desc);
    strides.update_output_desc_y(strides_desc);

    std::vector<int64_t> data_x_vec{1, 112, 60, 60};
    ge::Shape data_x_shape(data_x_vec);
    ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT);
    data_x.update_input_desc_x(data_x_desc);
    data_x.update_output_desc_y(data_x_desc);
    std::vector<int64_t> output_vec{1, 112, 60, 60};
    ge::Shape output_shape(output_vec);
    ge::TensorDesc output_desc(output_shape, FORMAT_ND, DT_FLOAT);
    strided_slice_op.update_input_desc_x(output_desc);
    strided_slice_op.update_output_desc_y(output_desc);
    std::vector<Operator> inputs{data_x, begin, end, strides};
    std::vector<Operator> outputs{strided_slice_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    // GE_DUMP(compute_graph_ptr, "strided_slice_fusion_test_1_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("StridedSliceRemovePass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "StridedSlice") {
            findOp = true;
        }
    }
    EXPECT_EQ(findOp, false);
}

TEST_F(strided_slice_remove_fusion_test, strided_slice_remove_fusion_test_2) {
    ge::Graph graph("strided_slice_fusion_test_2");

    auto data_x = op::Data().set_attr_index(0);
    auto strided_slice_d_op = op::StridedSliceD("strided_sliced_op")
                                  .set_input_x(data_x)
                                  .set_attr_begin({0, 0, 0, 0})
                                  .set_attr_end({1, 112, 60, 60})
                                  .set_attr_strides({1, 1, 1, 1})
                                  .set_attr_begin_mask(0)
                                  .set_attr_end_mask(0)
                                  .set_attr_ellipsis_mask(0)
                                  .set_attr_new_axis_mask(0)
                                  .set_attr_shrink_axis_mask(0);

    std::vector<int64_t> data_x_vec{1, 112, 60, 60};
    ge::Shape data_x_shape(data_x_vec);
    ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT16);
    data_x.update_input_desc_x(data_x_desc);
    data_x.update_output_desc_y(data_x_desc);
    std::vector<int64_t> output_vec{1, 112, 60, 60};
    ge::Shape output_shape(output_vec);
    ge::TensorDesc output_desc(output_shape, FORMAT_ND, DT_FLOAT16);
    strided_slice_d_op.update_input_desc_x(output_desc);
    strided_slice_d_op.update_output_desc_y(output_desc);
    std::vector<Operator> inputs{data_x};
    std::vector<Operator> outputs{strided_slice_d_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    // GE_DUMP(compute_graph_ptr, "strided_slice_remove_fusion_test_2_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("StridedSliceRemovePass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "StridedSliceD") {
            findOp = true;
        }
    }
    EXPECT_EQ(findOp, false);
}

TEST_F(strided_slice_remove_fusion_test, strided_slice_remove_fusion_test_3) {
    ge::Graph graph("strided_slice_remove_fusion_test_3");

    auto data_x = op::Data().set_attr_index(0);
    auto strided_slice_d_op = op::StridedSliceD("strided_sliced_op")
                                  .set_input_x(data_x)
                                  .set_attr_begin({0, 0, 0, 0})
                                  .set_attr_end({1, 112, 20, 20})
                                  .set_attr_strides({1, 1, 1, 1})
                                  .set_attr_begin_mask(0)
                                  .set_attr_end_mask(0)
                                  .set_attr_ellipsis_mask(0)
                                  .set_attr_new_axis_mask(0)
                                  .set_attr_shrink_axis_mask(0);

    std::vector<int64_t> data_x_vec{1, 112, 60, 60};
    ge::Shape data_x_shape(data_x_vec);
    ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT16);
    data_x.update_input_desc_x(data_x_desc);
    data_x.update_output_desc_y(data_x_desc);
    std::vector<int64_t> output_vec{1, 112, 60, 60};
    ge::Shape output_shape(output_vec);
    ge::TensorDesc output_desc(output_shape, FORMAT_ND, DT_FLOAT16);
    strided_slice_d_op.update_input_desc_x(output_desc);
    strided_slice_d_op.update_output_desc_y(output_desc);
    std::vector<Operator> inputs{data_x};
    std::vector<Operator> outputs{strided_slice_d_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    // GE_DUMP(compute_graph_ptr, "strided_slice_remove_fusion_test_2_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("StridedSliceRemovePass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "StridedSliceD") {
            findOp = true;
        }
    }
    EXPECT_EQ(findOp, true);
}

TEST_F(strided_slice_remove_fusion_test, strided_slice_remove_fusion_test_4) {
    ge::Graph graph("strided_slice_remove_fusion_test_4");

    auto data_x = op::Data().set_attr_index(0);
    auto strided_slice_d_op = op::StridedSliceD("strided_sliced_op")
                                  .set_input_x(data_x)
                                  .set_attr_begin({0, 0, 0, 0})
                                  .set_attr_end({1, 112, 60, 60})
                                  .set_attr_strides({1, 1, 1, 1})
                                  .set_attr_begin_mask(0)
                                  .set_attr_end_mask(0)
                                  .set_attr_ellipsis_mask(0)
                                  .set_attr_new_axis_mask(0)
                                  .set_attr_shrink_axis_mask(0);

    std::vector<int64_t> data_x_vec{-1, 112, 60, 60};
    ge::Shape data_x_shape(data_x_vec);
    ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_FLOAT16);
    data_x.update_input_desc_x(data_x_desc);
    data_x.update_output_desc_y(data_x_desc);
    std::vector<int64_t> output_vec{-1, 112, 60, 60};
    ge::Shape output_shape(output_vec);
    ge::TensorDesc output_desc(output_shape, FORMAT_ND, DT_FLOAT16);
    strided_slice_d_op.update_input_desc_x(output_desc);
    strided_slice_d_op.update_output_desc_y(output_desc);
    std::vector<Operator> inputs{data_x};
    std::vector<Operator> outputs{strided_slice_d_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    // GE_DUMP(compute_graph_ptr, "strided_slice_remove_fusion_test_4_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("StridedSliceRemovePass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "StridedSliceD") {
            findOp = true;
        }
    }
    EXPECT_EQ(findOp, true);
}

TEST_F(strided_slice_remove_fusion_test, strided_slice_remove_fusion_test_5) {
    OpDescPtr data = std::make_shared<OpDesc>("data0", "Data");
    OpDescPtr strideslice_op = std::make_shared<OpDesc>("strideslice", "StridedSliceD");
    OpDescPtr exp_op = std::make_shared<OpDesc>("exp", "Exp");

    vector<int64_t> input_dim = {1, 112, 60, 60};
    GeShape input_shape(input_dim);
    GeTensorDesc input_tenosr_desc(input_shape, FORMAT_NCHW, DT_FLOAT);
    input_tenosr_desc.SetOriginFormat(FORMAT_NCHW);
    input_tenosr_desc.SetOriginDataType(DT_FLOAT);
    input_tenosr_desc.SetOriginShape(input_shape);

    vector<int64_t> output_dim = {1, 112, 60, 60};
    GeShape output_shape(output_dim);
    GeTensorDesc output_tenosr_desc(output_shape, FORMAT_NCHW, DT_FLOAT);
    output_tenosr_desc.SetOriginFormat(FORMAT_NCHW);
    output_tenosr_desc.SetOriginDataType(DT_FLOAT);
    output_tenosr_desc.SetOriginShape(output_shape);

    data->AddOutputDesc(input_tenosr_desc);

    strideslice_op->AddInputDesc("x", input_tenosr_desc);
    strideslice_op->AddOutputDesc(output_tenosr_desc);
    exp_op->AddInputDesc("x", input_tenosr_desc);
    exp_op->AddOutputDesc(output_tenosr_desc);

    ge::AttrUtils::SetListInt(strideslice_op, "begin", {0, 0, 0, 0});
    ge::AttrUtils::SetListInt(strideslice_op, "end", {1, 112, 60, 60});
    ge::AttrUtils::SetListInt(strideslice_op, "strides", {1, 1, 1, 1});

    ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ComputeGraph>("control anchor test");
    NodePtr data_node = compute_graph_ptr->AddNode(data);
    NodePtr strideslice_node = compute_graph_ptr->AddNode(strideslice_op);
    NodePtr exp_node = compute_graph_ptr->AddNode(exp_op);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), strideslice_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(strideslice_node->GetOutDataAnchor(0), exp_node->GetInDataAnchor(0));

    GraphUtils::AddEdge(data_node->GetOutControlAnchor(), strideslice_node->GetInControlAnchor());

    fe::FusionPassTestUtils::RunGraphFusionPass("StridedSliceRemovePass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findFusionCast = false;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "StridedSliceD") {
            findFusionCast = true;
        }
    }
    EXPECT_EQ(findFusionCast, false);
}