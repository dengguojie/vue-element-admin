#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "array_ops.h"
#include "nn_pooling_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class avg_pool3d_grad_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "avg_pool3d_grad_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "avg_pool3d_grad_test TearDown" << std::endl;
    }
};

TEST_F(avg_pool3d_grad_test, avg_pool3d_grad_test_1)
{
    ge::Graph graph("avg_pool3d_grad_test_1");

    ge::Tensor crops_tensor;
    std::vector<int64_t> crops_vec{5};
    ge::Shape crops_shape(crops_vec);
    ge::TensorDesc crops_desc(crops_shape, FORMAT_NDHWC, DT_INT32);
    int64_t crops_size = crops_desc.GetShape().GetShapeSize();
    crops_desc.SetSize(crops_size * sizeof(int32_t));
    crops_tensor.SetTensorDesc(crops_desc);
    int32_t* crops_data = nullptr;
    crops_data = new int32_t[crops_size];
    *(crops_data + 0) = 9;
    *(crops_data + 1) = 6;
    *(crops_data + 2) = 28;
    *(crops_data + 3) = 28;
    *(crops_data + 4) = 48;
    crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int32_t));
    delete [] crops_data;

    auto crops = op::Constant().set_attr_value(crops_tensor);
    auto data0 = op::Data().set_attr_index(0);
    auto avp_pool3d_grad_op = op::AvgPool3DGrad("AvgPool3DGrad")
                               .set_input_orig_input_shape(crops)
                               .set_input_grads(data0)
                               .set_attr_ksize({1,1,2,2,1})
                               .set_attr_strides({1,1,9,2,1})
                               .set_attr_pads({0, 0, 0, 0, 0, 0})
                               .set_attr_ceil_mode(false)
                               .set_attr_count_include_pad(true)
                               .set_attr_divisor_override(0)
                               .set_attr_data_format("NDHWC");

    crops.update_output_desc_y(crops_desc);

    std::vector<int64_t> data0_vec{9, 6, 4, 14, 48};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NDHWC,  DT_FLOAT16);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    avp_pool3d_grad_op.update_input_desc_grads(data0_desc);
    avp_pool3d_grad_op.update_output_desc_output(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{avp_pool3d_grad_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    // GE_DUMP(compute_graph_ptr, "avg_pool3d_grad_test_1_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPool3DGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    // GE_DUMP(compute_graph_ptr, "avg_pool3d_grad_test_1_after");

    bool findD = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPool3DGrad") {
            findD = true;
            break;
        }
    }

    EXPECT_EQ(findD, false);
}

TEST_F(avg_pool3d_grad_test, avg_pool3d_grad_test_2)
{
    ge::Graph graph("avg_pool3d_grad_test_2");

    ge::Tensor crops_tensor;
    std::vector<int64_t> crops_vec{5};
    ge::Shape crops_shape(crops_vec);
    ge::TensorDesc crops_desc(crops_shape, FORMAT_NDHWC, DT_INT32);
    int64_t crops_size = crops_desc.GetShape().GetShapeSize();
    crops_desc.SetSize(crops_size * sizeof(int32_t));
    crops_tensor.SetTensorDesc(crops_desc);
    int32_t* crops_data = nullptr;
    crops_data = new int32_t[crops_size];
    *(crops_data + 0) = 9;
    *(crops_data + 1) = 6;
    *(crops_data + 2) = 28;
    *(crops_data + 3) = 28;
    *(crops_data + 4) = 48;
    crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int32_t));
    delete [] crops_data;

    auto crops = op::Constant().set_attr_value(crops_tensor);
    auto data0 = op::Data().set_attr_index(0);
    auto avp_pool3d_grad_op = op::AvgPool3DGrad("AvgPool3DGrad")
                               .set_input_orig_input_shape(crops)
                               .set_input_grads(data0)
                               .set_attr_ksize({1,1,2,2,1})
                               .set_attr_strides({1,1,9,2,1})
                               .set_attr_pads({0, 0, 0, 0, 0, 0})
                               .set_attr_ceil_mode(true)
                               .set_attr_count_include_pad(false)
                               .set_attr_divisor_override(0)
                               .set_attr_data_format("NDHWC");

    crops.update_output_desc_y(crops_desc);

    std::vector<int64_t> data0_vec{-1, 6, 4, 14, 48};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NDHWC,  DT_FLOAT16);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    avp_pool3d_grad_op.update_input_desc_grads(data0_desc);
    avp_pool3d_grad_op.update_output_desc_output(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{avp_pool3d_grad_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    // GE_DUMP(compute_graph_ptr, "avg_pool3d_grad_test_2_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPool3DGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    // GE_DUMP(compute_graph_ptr, "avg_pool3d_grad_test_2_after");

    bool findD = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPool3DGrad") {
            findD = true;
            break;
        }
    }

    EXPECT_EQ(findD, false);
}

TEST_F(avg_pool3d_grad_test, avg_pool3d_grad_test_3)
{
    ge::Graph graph("avg_pool3d_grad_test_3");

    ge::Tensor crops_tensor;
    std::vector<int64_t> crops_vec{5};
    ge::Shape crops_shape(crops_vec);
    ge::TensorDesc crops_desc(crops_shape, FORMAT_NDHWC, DT_INT32);
    int64_t crops_size = crops_desc.GetShape().GetShapeSize();
    crops_desc.SetSize(crops_size * sizeof(int32_t));
    crops_tensor.SetTensorDesc(crops_desc);
    int32_t* crops_data = nullptr;
    crops_data = new int32_t[crops_size];
    *(crops_data + 0) = 1;
    *(crops_data + 1) = 5;
    *(crops_data + 2) = 5;
    *(crops_data + 3) = 5;
    *(crops_data + 4) = 1;
    crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int32_t));
    delete [] crops_data;

    auto crops = op::Constant().set_attr_value(crops_tensor);
    auto data0 = op::Data().set_attr_index(0);
    auto avp_pool3d_grad_op = op::AvgPool3DGrad("AvgPool3DGrad")
                               .set_input_orig_input_shape(crops)
                               .set_input_grads(data0)
                               .set_attr_ksize({1,3,3,3,1})
                               .set_attr_strides({1,1,1,1,1})
                               .set_attr_pads({1, 1, 1, 1, 1, 1})
                               .set_attr_ceil_mode(false)
                               .set_attr_count_include_pad(true)
                               .set_attr_divisor_override(0)
                               .set_attr_data_format("NDHWC");

    crops.update_output_desc_y(crops_desc);

    std::vector<int64_t> data0_vec{1, 5, 5, 5, 1};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NDHWC,  DT_FLOAT16);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    avp_pool3d_grad_op.update_input_desc_grads(data0_desc);
    avp_pool3d_grad_op.update_output_desc_output(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{avp_pool3d_grad_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    // GE_DUMP(compute_graph_ptr, "avg_pool3d_grad_test_1_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPool3DGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    // GE_DUMP(compute_graph_ptr, "avg_pool3d_grad_test_1_after");

    bool findD = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPool3DGrad") {
            findD = true;
            break;
        }
    }

    EXPECT_EQ(findD, false);
}

TEST_F(avg_pool3d_grad_test, avg_pool3d_grad_test_global)
{
    ge::Graph graph("avg_pool3d_grad_test_4");

    ge::Tensor crops_tensor;
    std::vector<int64_t> crops_vec{5};
    ge::Shape crops_shape(crops_vec);
    ge::TensorDesc crops_desc(crops_shape, FORMAT_NDHWC, DT_INT32);
    int64_t crops_size = crops_desc.GetShape().GetShapeSize();
    crops_desc.SetSize(crops_size * sizeof(int32_t));
    crops_tensor.SetTensorDesc(crops_desc);
    int32_t* crops_data = nullptr;
    crops_data = new int32_t[crops_size];
    *(crops_data + 0) = 1;
    *(crops_data + 1) = 5;
    *(crops_data + 2) = 5;
    *(crops_data + 3) = 5;
    *(crops_data + 4) = 1;
    crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int32_t));
    delete [] crops_data;

    auto crops = op::Constant().set_attr_value(crops_tensor);
    auto data0 = op::Data().set_attr_index(0);
    auto avp_pool3d_grad_op = op::AvgPool3DGrad("AvgPool3DGrad")
                               .set_input_orig_input_shape(crops)
                               .set_input_grads(data0)
                               .set_attr_ksize({1,5,5,5,1})
                               .set_attr_strides({1,1,1,1,1})
                               .set_attr_pads({0,0,0,0,0,0})
                               .set_attr_ceil_mode(false)
                               .set_attr_count_include_pad(true)
                               .set_attr_divisor_override(0)
                               .set_attr_data_format("NDHWC");

    crops.update_output_desc_y(crops_desc);

    std::vector<int64_t> data0_vec{1, 1, 1, 1, 1};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NDHWC,  DT_FLOAT16);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    avp_pool3d_grad_op.update_input_desc_grads(data0_desc);
    avp_pool3d_grad_op.update_output_desc_output(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{avp_pool3d_grad_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    // GE_DUMP(compute_graph_ptr, "avg_pool3d_grad_test_1_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPool3DGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    // GE_DUMP(compute_graph_ptr, "avg_pool3d_grad_test_1_after");

    bool findD = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPool3DGrad") {
            findD = true;
            break;
        }
    }

    EXPECT_EQ(findD, false);
}

TEST_F(avg_pool3d_grad_test, avg_pool3d_grad_test_divisor_override)
{
    ge::Graph graph("avg_pool3d_grad_test_5");

    ge::Tensor crops_tensor;
    std::vector<int64_t> crops_vec{5};
    ge::Shape crops_shape(crops_vec);
    ge::TensorDesc crops_desc(crops_shape, FORMAT_NDHWC, DT_INT32);
    int64_t crops_size = crops_desc.GetShape().GetShapeSize();
    crops_desc.SetSize(crops_size * sizeof(int32_t));
    crops_tensor.SetTensorDesc(crops_desc);
    int32_t* crops_data = nullptr;
    crops_data = new int32_t[crops_size];
    *(crops_data + 0) = 1;
    *(crops_data + 1) = 5;
    *(crops_data + 2) = 5;
    *(crops_data + 3) = 5;
    *(crops_data + 4) = 1;
    crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int32_t));
    delete [] crops_data;

    auto crops = op::Constant().set_attr_value(crops_tensor);
    auto data0 = op::Data().set_attr_index(0);
    auto avp_pool3d_grad_op = op::AvgPool3DGrad("AvgPool3DGrad")
                               .set_input_orig_input_shape(crops)
                               .set_input_grads(data0)
                               .set_attr_ksize({1,3,3,3,1})
                               .set_attr_strides({1,1,1,1,1})
                               .set_attr_pads({0,0,0,0,0,0})
                               .set_attr_ceil_mode(false)
                               .set_attr_count_include_pad(true)
                               .set_attr_divisor_override(1)
                               .set_attr_data_format("NDHWC");

    crops.update_output_desc_y(crops_desc);

    std::vector<int64_t> data0_vec{1, 3, 3, 3, 1};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NDHWC,  DT_FLOAT16);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    avp_pool3d_grad_op.update_input_desc_grads(data0_desc);
    avp_pool3d_grad_op.update_output_desc_output(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{avp_pool3d_grad_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    // GE_DUMP(compute_graph_ptr, "avg_pool3d_grad_test_1_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPool3DGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    // GE_DUMP(compute_graph_ptr, "avg_pool3d_grad_test_1_after");

    bool findD = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPool3DGrad") {
            findD = true;
            break;
        }
    }

    EXPECT_EQ(findD, false);
}
