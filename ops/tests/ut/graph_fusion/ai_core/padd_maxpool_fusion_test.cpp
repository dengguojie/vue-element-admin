#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "all_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class padd_maxpool_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "padd_maxpool_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "padd_maxpool_fusion_test TearDown" << std::endl;
    }
};

TEST_F(padd_maxpool_fusion_test, padd_maxpool_fusion_test_1) {
    ge::Graph graph("padd_maxpool_fusion_test_1");

    std::vector<vector<int64_t>> paddings;
    for(int i=0;i<4;i++){
        std::vector<int64_t> tmp;
        for(int j=0;j<2;j++) {
            tmp.push_back(0);
        }
        paddings.push_back(tmp);
    }
    std::vector<int64_t> ksize;
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(1);
    std::vector<int64_t> strides;
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    std::string padding = "VALID";
    std::string data_format = "NHWC";
    auto data0 = op::Data().set_attr_index(0);
    auto pad = op::PadD("padd")
                        .set_input_x(data0)
                        .set_attr_paddings(paddings);
    auto maxpool = op::MaxPool("maxpool")
                        .set_input_x(pad)
                        .set_attr_ksize(ksize)
                        .set_attr_strides(strides)
                        .set_attr_padding(padding)
                        .set_attr_data_format(data_format);

    std::vector<int64_t> data0_vec{1, 112, 112, 3};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NHWC,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    pad.update_input_desc_x(data0_desc);
    pad.update_output_desc_y(data0_desc);
    maxpool.update_input_desc_x(data0_desc);
    maxpool.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{maxpool};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("PaddMaxPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 112, 112, 3};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Pooling") {
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
TEST_F(padd_maxpool_fusion_test, padd_maxpool_fusion_test_2) {
    ge::Graph graph("padd_maxpool_fusion_test_2");

    std::vector<vector<int64_t>> paddings;
    for(int i=0;i<4;i++){
        std::vector<int64_t> tmp;
        for(int j=0;j<2;j++) {
            tmp.push_back(0);
        }
        paddings.push_back(tmp);
    }
    std::vector<int64_t> ksize;
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(1);
    std::vector<int64_t> strides;
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    std::string padding = "VALID";
    std::string data_format = "NCHW";
    auto data0 = op::Data().set_attr_index(0);
    auto pad = op::PadD("padd")
                        .set_input_x(data0)
                        .set_attr_paddings(paddings);
    auto maxpool = op::MaxPool("maxpool")
                        .set_input_x(pad)
                        .set_attr_ksize(ksize)
                        .set_attr_strides(strides)
                        .set_attr_padding(padding)
                        .set_attr_data_format(data_format);

    std::vector<int64_t> data0_vec{1, 3, 112, 112};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NCHW,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    pad.update_input_desc_x(data0_desc);
    pad.update_output_desc_y(data0_desc);
    maxpool.update_input_desc_x(data0_desc);
    maxpool.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{maxpool};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("PaddMaxPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 3, 112, 112};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Pooling") {
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
TEST_F(padd_maxpool_fusion_test, padd_maxpool_fusion_test_3) {
    ge::Graph graph("padd_maxpool_fusion_test_3");

    std::vector<vector<int64_t>> paddings;
    for(int i=0;i<4;i++){
        std::vector<int64_t> tmp;
        for(int j=0;j<2;j++) {
            tmp.push_back(0);
        }
        paddings.push_back(tmp);
    }
    std::vector<int64_t> ksize;
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(1);
    std::vector<int64_t> strides;
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    std::string padding = "VALID";
    std::string data_format = "NHWC";
    auto data0 = op::Data().set_attr_index(0);
    auto pad = op::PadD("padd")
                        .set_input_x(data0)
                        .set_attr_paddings(paddings);
    auto maxpool = op::MaxPool("maxpool")
                        .set_input_x(pad)
                        .set_attr_ksize(ksize)
                        .set_attr_strides(strides)
                        .set_attr_padding(padding)
                        .set_attr_data_format(data_format);

    std::vector<int64_t> data0_vec{1, 112, 112, 3};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_ND,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    pad.update_input_desc_x(data0_desc);
    pad.update_output_desc_y(data0_desc);
    maxpool.update_input_desc_x(data0_desc);
    maxpool.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{maxpool};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("PaddMaxPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 112, 112, 3};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Pooling") {
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
TEST_F(padd_maxpool_fusion_test, padd_maxpool_fusion_test_4) {
    ge::Graph graph("padd_maxpool_fusion_test_4");

    std::vector<vector<int64_t>> paddings;
    for(int i=0;i<5;i++){
        std::vector<int64_t> tmp;
        for(int j=0;j<2;j++) {
            tmp.push_back(0);
        }
        paddings.push_back(tmp);
    }
    std::vector<int64_t> ksize;
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(1);
    std::vector<int64_t> strides;
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    std::string padding = "VALID";
    std::string data_format = "NHWC";
    auto data0 = op::Data().set_attr_index(0);
    auto pad = op::PadD("padd")
                        .set_input_x(data0)
                        .set_attr_paddings(paddings);
    auto maxpool = op::MaxPool("maxpool")
                        .set_input_x(pad)
                        .set_attr_ksize(ksize)
                        .set_attr_strides(strides)
                        .set_attr_padding(padding)
                        .set_attr_data_format(data_format);

    std::vector<int64_t> data0_vec{1, 112, 112, 3};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NHWC,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    pad.update_input_desc_x(data0_desc);
    pad.update_output_desc_y(data0_desc);
    maxpool.update_input_desc_x(data0_desc);
    maxpool.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{maxpool};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("PaddMaxPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 112, 112, 3};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Pooling") {
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
TEST_F(padd_maxpool_fusion_test, padd_maxpool_fusion_test_5) {
    ge::Graph graph("padd_maxpool_fusion_test_5");

    std::vector<vector<int64_t>> paddings;
    for(int i=0;i<4;i++){
        std::vector<int64_t> tmp;
        for(int j=0;j<2;j++) {
            tmp.push_back(0);
        }
        paddings.push_back(tmp);
    }
    std::vector<int64_t> ksize;
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(1);
    std::vector<int64_t> strides;
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    std::string padding = "VALID";
    std::string data_format = "NHWC";
    auto data0 = op::Data().set_attr_index(0);
    auto pad = op::PadD("padd")
                        .set_input_x(data0)
                        .set_attr_paddings(paddings);
    auto maxpool = op::MaxPool("maxpool")
                        .set_input_x(pad)
                        .set_attr_ksize(ksize)
                        .set_attr_strides(strides)
                        .set_attr_padding(padding)
                        .set_attr_data_format(data_format);

    std::vector<int64_t> data0_vec{1, 112, 112, 3};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NHWC,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    pad.update_input_desc_x(data0_desc);
    pad.update_output_desc_y(data0_desc);
    maxpool.update_input_desc_x(data0_desc);
    maxpool.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{maxpool};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("PaddMaxPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 112, 112, 3};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Pooling") {
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
TEST_F(padd_maxpool_fusion_test, padd_maxpool_fusion_test_6) {
    ge::Graph graph("padd_maxpool_fusion_test_6");

    std::vector<vector<int64_t>> paddings;
    for(int i=0;i<4;i++){
        std::vector<int64_t> tmp;
        for(int j=0;j<2;j++) {
            tmp.push_back(0);
        }
        paddings.push_back(tmp);
    }
    std::vector<int64_t> ksize;
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(1);
    std::vector<int64_t> strides;
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    std::string padding = "VALID";
    std::string data_format = "NHWC";
    auto data0 = op::Data().set_attr_index(0);
    auto pad = op::PadD("padd")
                        .set_input_x(data0)
                        .set_attr_paddings(paddings);
    auto maxpool = op::MaxPool("maxpool")
                        .set_input_x(pad)
                        .set_attr_ksize(ksize)
                        .set_attr_strides(strides)
                        .set_attr_padding(padding)
                        .set_attr_data_format(data_format);

    std::vector<int64_t> data0_vec{1, 112, 112, 3};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NHWC,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    pad.update_input_desc_x(data0_desc);
    pad.update_output_desc_y(data0_desc);
    maxpool.update_input_desc_x(data0_desc);
    maxpool.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{maxpool};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("PaddMaxPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 112, 112, 3};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Pooling") {
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
TEST_F(padd_maxpool_fusion_test, padd_maxpool_fusion_test_7) {
    ge::Graph graph("padd_maxpool_fusion_test_7");

    std::vector<vector<int64_t>> paddings;
    for(int i=0;i<4;i++){
        std::vector<int64_t> tmp;
        for(int j=0;j<2;j++) {
            tmp.push_back(1);
        }
        paddings.push_back(tmp);
    }
    std::vector<int64_t> ksize;
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(1);
    std::vector<int64_t> strides;
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    std::string padding = "VALID";
    std::string data_format = "NHWC";
    auto data0 = op::Data().set_attr_index(0);
    auto pad = op::PadD("padd")
                        .set_input_x(data0)
                        .set_attr_paddings(paddings);
    auto maxpool = op::MaxPool("maxpool")
                        .set_input_x(pad)
                        .set_attr_ksize(ksize)
                        .set_attr_strides(strides)
                        .set_attr_padding(padding)
                        .set_attr_data_format(data_format);

    std::vector<int64_t> data0_vec{1, 112, 112, 3};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NHWC,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    pad.update_input_desc_x(data0_desc);
    pad.update_output_desc_y(data0_desc);
    maxpool.update_input_desc_x(data0_desc);
    maxpool.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{maxpool};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("PaddMaxPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 112, 112, 3};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Pooling") {
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
TEST_F(padd_maxpool_fusion_test, padd_maxpool_fusion_test_8) {
    ge::Graph graph("padd_maxpool_fusion_test_8");

    std::vector<vector<int64_t>> paddings;
    for(int i=0;i<4;i++){
        std::vector<int64_t> tmp;
        for(int j=0;j<2;j++) {
            tmp.push_back(1);
        }
        paddings.push_back(tmp);
    }
    std::vector<int64_t> ksize;
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(1);
    std::vector<int64_t> strides;
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    std::string padding = "VALID";
    std::string data_format = "NCHW";
    auto data0 = op::Data().set_attr_index(0);
    auto pad = op::PadD("padd")
                        .set_input_x(data0)
                        .set_attr_paddings(paddings);
    auto maxpool = op::MaxPool("maxpool")
                        .set_input_x(pad)
                        .set_attr_ksize(ksize)
                        .set_attr_strides(strides)
                        .set_attr_padding(padding)
                        .set_attr_data_format(data_format);

    std::vector<int64_t> data0_vec{1, 3, 112, 112};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NCHW,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    pad.update_input_desc_x(data0_desc);
    pad.update_output_desc_y(data0_desc);
    maxpool.update_input_desc_x(data0_desc);
    maxpool.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{maxpool};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("PaddMaxPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 3, 112, 112};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Pooling") {
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
TEST_F(padd_maxpool_fusion_test, padd_maxpool_fusion_test_9) {
    ge::Graph graph("padd_maxpool_fusion_test_9");

    std::vector<vector<int64_t>> paddings;
    for(int i=0;i<4;i++){
        std::vector<int64_t> tmp;
        for(int j=0;j<2;j++) {
            tmp.push_back(0);
        }
        paddings.push_back(tmp);
    }
    std::vector<int64_t> ksize;
    ksize.push_back(2);
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(1);
    std::vector<int64_t> strides;
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    std::string padding = "VALID";
    std::string data_format = "NHWC";
    auto data0 = op::Data().set_attr_index(0);
    auto pad = op::PadD("padd")
                        .set_input_x(data0)
                        .set_attr_paddings(paddings);
    auto maxpool = op::MaxPool("maxpool")
                        .set_input_x(pad)
                        .set_attr_ksize(ksize)
                        .set_attr_strides(strides)
                        .set_attr_padding(padding)
                        .set_attr_data_format(data_format);

    std::vector<int64_t> data0_vec{1, 112, 112, 3};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NHWC,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    pad.update_input_desc_x(data0_desc);
    pad.update_output_desc_y(data0_desc);
    maxpool.update_input_desc_x(data0_desc);
    maxpool.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{maxpool};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("PaddMaxPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 112, 112, 3};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Pooling") {
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
TEST_F(padd_maxpool_fusion_test, padd_maxpool_fusion_test_10) {
    ge::Graph graph("padd_maxpool_fusion_test_10");

    std::vector<vector<int64_t>> paddings;
    for(int i=0;i<4;i++){
        std::vector<int64_t> tmp;
        for(int j=0;j<2;j++) {
            tmp.push_back(0);
        }
        paddings.push_back(tmp);
    }
    std::vector<int64_t> ksize;
    ksize.push_back(2);
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(1);
    std::vector<int64_t> strides;
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    std::string padding = "VALID";
    std::string data_format = "NCHW";
    auto data0 = op::Data().set_attr_index(0);
    auto pad = op::PadD("padd")
                        .set_input_x(data0)
                        .set_attr_paddings(paddings);
    auto maxpool = op::MaxPool("maxpool")
                        .set_input_x(pad)
                        .set_attr_ksize(ksize)
                        .set_attr_strides(strides)
                        .set_attr_padding(padding)
                        .set_attr_data_format(data_format);

    std::vector<int64_t> data0_vec{1, 3, 112, 112};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NCHW,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    pad.update_input_desc_x(data0_desc);
    pad.update_output_desc_y(data0_desc);
    maxpool.update_input_desc_x(data0_desc);
    maxpool.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{maxpool};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("PaddMaxPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 3, 112, 112};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Pooling") {
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
TEST_F(padd_maxpool_fusion_test, padd_maxpool_fusion_test_11) {
    ge::Graph graph("padd_maxpool_fusion_test_11");

    std::vector<vector<int64_t>> paddings;
    for(int i=0;i<4;i++){
        std::vector<int64_t> tmp;
        for(int j=0;j<2;j++) {
            tmp.push_back(0);
        }
        paddings.push_back(tmp);
    }
    std::vector<int64_t> ksize;
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(1);
    std::vector<int64_t> strides;
    strides.push_back(2);
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    std::string padding = "VALID";
    std::string data_format = "NHWC";
    auto data0 = op::Data().set_attr_index(0);
    auto pad = op::PadD("padd")
                        .set_input_x(data0)
                        .set_attr_paddings(paddings);
    auto maxpool = op::MaxPool("maxpool")
                        .set_input_x(pad)
                        .set_attr_ksize(ksize)
                        .set_attr_strides(strides)
                        .set_attr_padding(padding)
                        .set_attr_data_format(data_format);

    std::vector<int64_t> data0_vec{1, 112, 112, 3};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NHWC,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    pad.update_input_desc_x(data0_desc);
    pad.update_output_desc_y(data0_desc);
    maxpool.update_input_desc_x(data0_desc);
    maxpool.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{maxpool};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("PaddMaxPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 112, 112, 3};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Pooling") {
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
TEST_F(padd_maxpool_fusion_test, padd_maxpool_fusion_test_12) {
    ge::Graph graph("padd_maxpool_fusion_test_12");

    std::vector<vector<int64_t>> paddings;
    for(int i=0;i<4;i++){
        std::vector<int64_t> tmp;
        for(int j=0;j<2;j++) {
            tmp.push_back(0);
        }
        paddings.push_back(tmp);
    }
    std::vector<int64_t> ksize;
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(1);
    std::vector<int64_t> strides;
    strides.push_back(2);
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    std::string padding = "VALID";
    std::string data_format = "NCHW";
    auto data0 = op::Data().set_attr_index(0);
    auto pad = op::PadD("padd")
                        .set_input_x(data0)
                        .set_attr_paddings(paddings);
    auto maxpool = op::MaxPool("maxpool")
                        .set_input_x(pad)
                        .set_attr_ksize(ksize)
                        .set_attr_strides(strides)
                        .set_attr_padding(padding)
                        .set_attr_data_format(data_format);

    std::vector<int64_t> data0_vec{1, 3, 112, 112};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NCHW,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    pad.update_input_desc_x(data0_desc);
    pad.update_output_desc_y(data0_desc);
    maxpool.update_input_desc_x(data0_desc);
    maxpool.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{maxpool};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("PaddMaxPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 3, 112, 112};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Pooling") {
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
TEST_F(padd_maxpool_fusion_test, padd_maxpool_fusion_test_13) {
    ge::Graph graph("padd_maxpool_fusion_test_13");

    std::vector<vector<int64_t>> paddings;
    for(int i=0;i<4;i++){
        std::vector<int64_t> tmp;
        for(int j=0;j<2;j++) {
            tmp.push_back(0);
        }
        paddings.push_back(tmp);
    }
    std::vector<int64_t> ksize;
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(1);
    ksize.push_back(1);
    std::vector<int64_t> strides;
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    strides.push_back(1);
    std::string padding = "SAME";
    std::string data_format = "NHWC";
    auto data0 = op::Data().set_attr_index(0);
    auto pad = op::PadD("padd")
                        .set_input_x(data0)
                        .set_attr_paddings(paddings);
    auto maxpool = op::MaxPool("maxpool")
                        .set_input_x(pad)
                        .set_attr_ksize(ksize)
                        .set_attr_strides(strides)
                        .set_attr_padding(padding)
                        .set_attr_data_format(data_format);

    std::vector<int64_t> data0_vec{1, 112, 112, 3};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NHWC,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    pad.update_input_desc_x(data0_desc);
    pad.update_output_desc_y(data0_desc);
    maxpool.update_input_desc_x(data0_desc);
    maxpool.update_output_desc_y(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{maxpool};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("PaddMaxPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 112, 112, 3};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Pooling") {
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
