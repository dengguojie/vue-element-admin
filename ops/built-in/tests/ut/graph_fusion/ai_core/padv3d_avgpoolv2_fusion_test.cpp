#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "all_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class padv3d_avgpoolv2_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "padv3d_avgpoolv2_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "padv3d_avgpoolv2_fusion_test TearDown" << std::endl;
    }
};

TEST_F(padv3d_avgpoolv2_fusion_test, padv3d_avgpoolv2_fusion_test_1) {
    ge::Graph graph("padv3d_avgpoolv2_fusion_test");
    std::vector<int64_t> ksize = {1, 1, 1, 1};
    std::vector<int64_t> strides = {1, 1, 1, 1};
    std::vector<int64_t> pads = {0, 0, 0, 0};
    std::string padding = "VALID";
    std::string data_format = "NCHW";
    auto data0 = op::Data().set_attr_index(0);
    std::vector<std::vector<int64_t>> pad_v3_paddings(4, std::vector<int64_t>(2, 0));
    auto pad = op::PadV3D("padd")
                        .set_input_x(data0)
                        .set_attr_paddings(pad_v3_paddings)
                        .set_attr_paddings_contiguous(false);

    auto avgpool = op::AvgPoolV2("AvgPoolV2")
                        .set_input_x(pad)
                        .set_attr_ksize(ksize)
                        .set_attr_strides(strides)
                        .set_attr_padding_mode(padding)
                        .set_attr_pads(pads)
                        .set_attr_data_format(data_format);

    std::vector<int64_t> data0_vec{1, 112, 112, 3};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NCHW,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    
    pad.update_input_desc_x(data0_desc);
    pad.update_output_desc_y(data0_desc);
    avgpool.update_input_desc_x(data0_desc);
    avgpool.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{avgpool};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

    bool findOp = false;

    fe::FusionPassTestUtils::RunGraphFusionPass("Padv3dAvgpoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    findOp = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPoolV2") {
            findOp = true;
        }
    }
    EXPECT_EQ(findOp, true);
}

TEST_F(padv3d_avgpoolv2_fusion_test, padv3d_avgpoolv2_fusion_test_2) {
    ge::Graph graph("padv3d_avgpoolv2_fusion_test");
    std::vector<int64_t> ksize = {1, 1, 1, 1};
    std::vector<int64_t> strides = {1, 1, 1, 1};
    std::vector<int64_t> pads = {0, 0, 0, 0};
    std::string padding = "VALID";
    std::string data_format = "NCHW";
    auto data0 = op::Data().set_attr_index(0);
    std::vector<std::vector<int64_t>> pad_v3_paddings(4, std::vector<int64_t>(2, 0));
    auto pad = op::PadV3D("padd")
                        .set_input_x(data0)
                        .set_attr_paddings(pad_v3_paddings)
                        .set_attr_paddings_contiguous(true);

    auto avgpool = op::AvgPoolV2("AvgPoolV2")
                        .set_input_x(pad)
                        .set_attr_ksize(ksize)
                        .set_attr_strides(strides)
                        .set_attr_padding_mode(padding)
                        .set_attr_pads(pads)
                        .set_attr_data_format(data_format);

    std::vector<int64_t> data0_vec{1, 112, 112, 3};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NCHW,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    
    pad.update_input_desc_x(data0_desc);
    pad.update_output_desc_y(data0_desc);
    avgpool.update_input_desc_x(data0_desc);
    avgpool.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{avgpool};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

    bool findOp = false;

    fe::FusionPassTestUtils::RunGraphFusionPass("Padv3dAvgpoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    findOp = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPoolV2") {
            findOp = true;
        }
    }
    EXPECT_EQ(findOp, true);
}

TEST_F(padv3d_avgpoolv2_fusion_test, padv3d_avgpoolv2_fusion_test_3) {
    ge::Graph graph("padv3d_avgpoolv2_fusion_test");
    std::vector<int64_t> ksize = {1, 1, 1, 1};
    std::vector<int64_t> strides = {1, 1, 1, 1};
    std::vector<int64_t> pads = {0, 0, 0, 0};
    std::string padding = "VALID";
    std::string data_format = "NCHW";
    auto data0 = op::Data().set_attr_index(0);
    
    std::vector<std::vector<int64_t>> pad_v3_paddings(4, std::vector<int64_t>(2, 0));
    auto pad = op::PadV3D("padd")
                        .set_input_x(data0)
                        .set_attr_paddings(pad_v3_paddings)
                        .set_attr_paddings_contiguous(true)
                        .set_attr_constant_values(0)
                        .set_attr_mode("constant");

    auto avgpool = op::AvgPoolV2("AvgPoolV2")
                        .set_input_x(pad)
                        .set_attr_ksize(ksize)
                        .set_attr_strides(strides)
                        .set_attr_padding_mode(padding)
                        .set_attr_pads(pads)
                        .set_attr_data_format(data_format);

    std::vector<int64_t> data0_vec{1, 112, 112, 3};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NHWC,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    
    pad.update_input_desc_x(data0_desc);
    pad.update_output_desc_y(data0_desc);
    avgpool.update_input_desc_x(data0_desc);
    avgpool.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{avgpool};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

    fe::FusionPassTestUtils::RunGraphFusionPass("Padv3dAvgpoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    fe::FusionPassTestUtils::RunGraphFusionPass("Padv3dAvgpoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    findOp = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPoolV2") {
            findOp = true;
        }
    }
    EXPECT_EQ(findOp, true);
}

TEST_F(padv3d_avgpoolv2_fusion_test, padv3d_avgpoolv2_fusion_test_4) {
    ge::Graph graph("padv3d_avgpoolv2_fusion_test");
    std::vector<int64_t> ksize = {1, 1, 1, 1};
    std::vector<int64_t> strides = {1, 1, 1, 1};
    std::vector<int64_t> pads = {0, 0, 0, 0};
    std::string padding = "VALID";
    std::string data_format = "NCHW";
    auto data0 = op::Data().set_attr_index(0);
    std::vector<std::vector<int64_t>> pad_v3_paddings(5, std::vector<int64_t>(2, 0));

    auto pad = op::PadV3D("padd")
                        .set_input_x(data0)
                        .set_attr_paddings(pad_v3_paddings)
                        .set_attr_paddings_contiguous(false)
                        .set_attr_constant_values(0)
                        .set_attr_mode("constant");

    auto avgpool = op::AvgPool3DD("AvgPool3DD")
                        .set_input_x(pad)
                        .set_attr_ksize(ksize)
                        .set_attr_strides(strides)
                        .set_attr_pads(pads)
                        .set_attr_data_format(data_format);

    std::vector<int64_t> data0_vec{1, 112, 112, 3, 112};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NCDHW,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    
    pad.update_input_desc_x(data0_desc);
    pad.update_output_desc_y(data0_desc);
    avgpool.update_input_desc_x(data0_desc);
    avgpool.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{avgpool};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

    bool findOp = false;
    fe::FusionPassTestUtils::RunGraphFusionPass("Padv3dAvgpoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    findOp = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPool3DD") {
            findOp = true;
        }
    }
    EXPECT_EQ(findOp, true);
}

TEST_F(padv3d_avgpoolv2_fusion_test, padv3d_avgpoolv2_fusion_test_5) {
    ge::Graph graph("padv3d_avgpoolv2_fusion_test");
    std::vector<int64_t> ksize = {1, 1, 1, 1};
    std::vector<int64_t> strides = {1, 1, 1, 1};
    std::vector<int64_t> pads = {0, 0, 0, 0};
    std::string padding = "VALID";
    std::string data_format = "NCHW";
    auto data0 = op::Data().set_attr_index(0);
    std::vector<std::vector<int64_t>> pad_v3_paddings(5, std::vector<int64_t>(2, 0));

    auto pad = op::PadV3D("padd")
                        .set_input_x(data0)
                        .set_attr_paddings(pad_v3_paddings)
                        .set_attr_paddings_contiguous(true)
                        .set_attr_constant_values(0)
                        .set_attr_mode("constant");

    auto avgpool = op::AvgPool3DD("AvgPool3DD")
                        .set_input_x(pad)
                        .set_attr_ksize(ksize)
                        .set_attr_strides(strides)
                        .set_attr_pads(pads)
                        .set_attr_data_format(data_format);

    std::vector<int64_t> data0_vec{1, 112, 112, 3, 112};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NCDHW,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    
    pad.update_input_desc_x(data0_desc);
    pad.update_output_desc_y(data0_desc);
    avgpool.update_input_desc_x(data0_desc);
    avgpool.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{avgpool};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

    bool findOp = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPool3DD") {
            findOp = true;
        }
    }
    EXPECT_EQ(findOp, true);
}

TEST_F(padv3d_avgpoolv2_fusion_test, padv3d_avgpoolv2_fusion_test_6) {
    ge::Graph graph("padv3d_avgpoolv2_fusion_test");
    std::vector<int64_t> ksize = {1, 1, 1, 1};
    std::vector<int64_t> strides = {1, 1, 1, 1};
    std::vector<int64_t> pads = {0, 0, 0, 0};
    std::string padding = "VALID";
    std::string data_format = "NCHW";
    auto data0 = op::Data().set_attr_index(0);
    
    std::vector<std::vector<int64_t>> pad_v3_paddings(4, std::vector<int64_t>(2, 0));
    auto pad = op::PadV3D("padd")
                        .set_input_x(data0)
                        .set_attr_paddings(pad_v3_paddings)
                        .set_attr_paddings_contiguous(false)
                        .set_attr_constant_values(0)
                        .set_attr_mode("constant");

    auto avgpool = op::AvgPoolV2("AvgPoolV2")
                        .set_input_x(pad)
                        .set_attr_ksize(ksize)
                        .set_attr_strides(strides)
                        .set_attr_padding_mode(padding)
                        .set_attr_pads(pads)
                        .set_attr_data_format(data_format);

    std::vector<int64_t> data0_vec{1, 112, 112, 3};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NHWC,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    
    pad.update_input_desc_x(data0_desc);
    pad.update_output_desc_y(data0_desc);
    avgpool.update_input_desc_x(data0_desc);
    avgpool.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{avgpool};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

    fe::FusionPassTestUtils::RunGraphFusionPass("Padv3dAvgpoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;

    fe::FusionPassTestUtils::RunGraphFusionPass("Padv3dAvgpoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    findOp = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPoolV2") {
            findOp = true;
        }
    }
    EXPECT_EQ(findOp, true);
}

TEST_F(padv3d_avgpoolv2_fusion_test, padv3d_avgpoolv2_fusion_test_7) {
    ge::Graph graph("padv3d_avgpoolv2_fusion_test");
    std::vector<int64_t> ksize = {-1, -1, -1, -1};
    std::vector<int64_t> strides = {1, 1, 1, 1};
    std::vector<int64_t> pads = {0, 0, 0, 0};
    std::string padding = "VALID";
    std::string data_format = "NCHW";
    auto data0 = op::Data().set_attr_index(0);
    
    std::vector<std::vector<int64_t>> pad_v3_paddings(4, std::vector<int64_t>(2, 0));
    pad_v3_paddings[1][0] = 1;
    pad_v3_paddings[1][1] = 1;
    auto pad = op::PadV3D("padd")
                        .set_input_x(data0)
                        .set_attr_paddings(pad_v3_paddings)
                        .set_attr_paddings_contiguous(false)
                        .set_attr_constant_values(0)
                        .set_attr_mode("constant");

    auto avgpool = op::AvgPoolV2("AvgPoolV2")
                        .set_input_x(pad)
                        .set_attr_ksize(ksize)
                        .set_attr_strides(strides)
                        .set_attr_padding_mode(padding)
                        .set_attr_pads(pads)
                        .set_attr_data_format(data_format);

    std::vector<int64_t> data0_vec{1, 112, 112, 3};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NHWC,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    
    pad.update_input_desc_x(data0_desc);
    pad.update_output_desc_y(data0_desc);
    avgpool.update_input_desc_x(data0_desc);
    avgpool.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{avgpool};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

    fe::FusionPassTestUtils::RunGraphFusionPass("Padv3dAvgpoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;

    fe::FusionPassTestUtils::RunGraphFusionPass("Padv3dAvgpoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    findOp = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPoolV2") {
            findOp = true;
        }
    }
    EXPECT_EQ(findOp, true);
}

TEST_F(padv3d_avgpoolv2_fusion_test, padv3d_avgpoolv2_fusion_test_8) {
    ge::Graph graph("padv3d_avgpoolv2_fusion_test");
    std::vector<int64_t> ksize = {1, 1, 1};
    std::vector<int64_t> strides = {1, 1, 1, 1};
    std::vector<int64_t> pads = {0, 0, 0, 0};
    std::string padding = "VALID";
    std::string data_format = "NCHW";
    auto data0 = op::Data().set_attr_index(0);
    
    std::vector<std::vector<int64_t>> pad_v3_paddings(4, std::vector<int64_t>(2, 0));
    pad_v3_paddings[1][0] = 1;
    pad_v3_paddings[1][1] = 1;
    auto pad = op::PadV3D("padd")
                        .set_input_x(data0)
                        .set_attr_paddings(pad_v3_paddings)
                        .set_attr_paddings_contiguous(false)
                        .set_attr_constant_values(0)
                        .set_attr_mode("constant");

    auto avgpool = op::AvgPoolV2("AvgPoolV2")
                        .set_input_x(pad)
                        .set_attr_ksize(ksize)
                        .set_attr_strides(strides)
                        .set_attr_padding_mode(padding)
                        .set_attr_pads(pads)
                        .set_attr_data_format(data_format);

    std::vector<int64_t> data0_vec{1, 112, 112, 3};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NHWC,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    
    pad.update_input_desc_x(data0_desc);
    pad.update_output_desc_y(data0_desc);
    avgpool.update_input_desc_x(data0_desc);
    avgpool.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{avgpool};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

    fe::FusionPassTestUtils::RunGraphFusionPass("Padv3dAvgpoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;

    fe::FusionPassTestUtils::RunGraphFusionPass("Padv3dAvgpoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    findOp = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPoolV2") {
            findOp = true;
        }
    }
    EXPECT_EQ(findOp, true);
}

TEST_F(padv3d_avgpoolv2_fusion_test, padv3d_avgpoolv2_fusion_test_9) {
    ge::Graph graph("padv3d_avgpoolv2_fusion_test");
    std::vector<int64_t> ksize = {1, 1, 1, 1};
    std::vector<int64_t> strides = {1, 1, 1, 1};
    std::vector<int64_t> pads = {0, 0, 0, 0};
    std::string padding = "VALID";
    std::string data_format = "NCHW";
    auto data0 = op::Data().set_attr_index(0);
    std::vector<int32_t> pad_v3_paddings(8, 0);
    std::vector<int64_t> dims = {8};
    ge::Shape shape(dims);
    ge::TensorDesc desc(shape, ge::FORMAT_ND, ge::DT_INT32);
    ge::Tensor tensor(desc, reinterpret_cast<uint8_t*>(pad_v3_paddings.data()), pad_v3_paddings.size() * sizeof(int32_t));
    auto const_op = op::Const("cost").set_attr_value(tensor);
    auto pad = op::PadV3("padd")
                        .set_input_x(data0)
                        .set_input_paddings(const_op)
                        .set_attr_paddings_contiguous(false);
    auto input_desc = pad.GetInputDesc("paddings");
    input_desc.SetDataType(ge::DT_INT32);
    pad.UpdateInputDesc("paddings", input_desc);

    auto avgpool = op::AvgPoolV2("AvgPoolV2")
                        .set_input_x(pad)
                        .set_attr_ksize(ksize)
                        .set_attr_strides(strides)
                        .set_attr_padding_mode(padding)
                        .set_attr_pads(pads)
                        .set_attr_data_format(data_format);

    std::vector<int64_t> data0_vec{1, 112, 112, 3};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NCHW,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    
    pad.update_input_desc_x(data0_desc);
    pad.update_output_desc_y(data0_desc);
    avgpool.update_input_desc_x(data0_desc);
    avgpool.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{avgpool};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

    bool findOp = false;

    fe::FusionPassTestUtils::RunGraphFusionPass("Padv3dAvgpoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    findOp = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPoolV2") {
            findOp = true;
        }
    }
    EXPECT_EQ(findOp, true);
}

TEST_F(padv3d_avgpoolv2_fusion_test, padv3d_avgpoolv2_fusion_test_10) {
    ge::Graph graph("padv3d_avgpoolv2_fusion_test");
    std::vector<int64_t> ksize = {1, 1, 1, 1};
    std::vector<int64_t> strides = {1, 1, 1, 1};
    std::vector<int64_t> pads = {0, 0, 0, 0};
    std::string padding = "VALID";
    std::string data_format = "NCHW";
    auto data0 = op::Data().set_attr_index(0);
    std::vector<int64_t> pad_v3_paddings(8, 0);
    std::vector<int64_t> dims = {8};
    ge::Shape shape(dims);
    ge::TensorDesc desc(shape, ge::FORMAT_ND, ge::DT_INT64);
    ge::Tensor tensor(desc, reinterpret_cast<uint8_t*>(pad_v3_paddings.data()), pad_v3_paddings.size() * sizeof(int64_t));
    auto const_op = op::Const("cost").set_attr_value(tensor);
    auto pad = op::PadV3("padd")
                        .set_input_x(data0)
                        .set_input_paddings(const_op)
                        .set_attr_paddings_contiguous(false);
    auto input_desc = pad.GetInputDesc("paddings");
    input_desc.SetDataType(ge::DT_INT64);
    pad.UpdateInputDesc("paddings", input_desc);

    auto avgpool = op::AvgPoolV2("AvgPoolV2")
                        .set_input_x(pad)
                        .set_attr_ksize(ksize)
                        .set_attr_strides(strides)
                        .set_attr_padding_mode(padding)
                        .set_attr_pads(pads)
                        .set_attr_data_format(data_format);

    std::vector<int64_t> data0_vec{1, 112, 112, 3};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NCHW,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    
    pad.update_input_desc_x(data0_desc);
    pad.update_output_desc_y(data0_desc);
    avgpool.update_input_desc_x(data0_desc);
    avgpool.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{avgpool};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

    bool findOp = false;

    fe::FusionPassTestUtils::RunGraphFusionPass("Padv3dAvgpoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    findOp = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPoolV2") {
            findOp = true;
        }
    }
    EXPECT_EQ(findOp, true);
}