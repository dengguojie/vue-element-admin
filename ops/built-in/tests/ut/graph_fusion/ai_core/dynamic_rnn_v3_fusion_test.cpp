#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "rnn.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class dynamic_rnn_v3_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "dynamic_rnn_v3_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "dynamic_rnn_v3_fusion_test TearDown" << std::endl;
    }
};

TEST_F(dynamic_rnn_v3_fusion_test, dynamic_rnn_v3_fusion_test_1) {
    ge::Graph graph("dynamic_rnn_fusion_test_1");

    auto xData = op::Data("xData");
    std::vector<int64_t> dims_x{1, 32, 1024};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC,  DT_FLOAT16);
    xData.update_input_desc_x(tensorDescX);
    xData.update_output_desc_y(tensorDescX);

    auto wData = op::Data("wData");
    std::vector<int64_t> dims_w{1056, 64};
    ge::Shape shape_w(dims_w);
    ge::TensorDesc tensorDescW(shape_w, FORMAT_HWCN,  DT_FLOAT16);
    wData.update_input_desc_x(tensorDescW);
    wData.update_output_desc_y(tensorDescW);

    auto bData = op::Data("bData");
    std::vector<int64_t> dims_b{64,};
    ge::Shape shape_b(dims_b);
    ge::TensorDesc tensorDescB(shape_b, FORMAT_NHWC,  DT_FLOAT16);
    bData.update_input_desc_x(tensorDescB);
    bData.update_output_desc_y(tensorDescB);

    auto initHData = op::Data("initHData");
    std::vector<int64_t> dims_init_h{32, 32};
    ge::Shape shape_init_h(dims_init_h);
    ge::TensorDesc tensorDescInitH(shape_init_h, FORMAT_NHWC,  DT_FLOAT16);
    initHData.update_input_desc_x(tensorDescInitH);
    initHData.update_output_desc_y(tensorDescInitH);

    auto initCData = op::Data("initCData");
    std::vector<int64_t> dims_init_c{32, 16};
    ge::Shape shape_init_c(dims_init_c);
    ge::TensorDesc tensorDescInitC(shape_init_c, FORMAT_NHWC,  DT_FLOAT16);
    initCData.update_input_desc_x(tensorDescInitC);
    initCData.update_output_desc_y(tensorDescInitC);

    auto wciData = op::Data("wciData");
    std::vector<int64_t> dims_wci{32, 16};
    ge::Shape shape_wci(dims_wci);
    ge::TensorDesc tensorDescWci(shape_wci, FORMAT_NHWC,  DT_FLOAT16);
    wciData.update_input_desc_x(tensorDescWci);
    wciData.update_output_desc_y(tensorDescWci);

    auto wcfData = op::Data("wcfData");
    std::vector<int64_t> dims_wcf{32, 16};
    ge::Shape shape_wcf(dims_wcf);
    ge::TensorDesc tensorDescWcf(shape_wcf, FORMAT_NHWC,  DT_FLOAT16);
    wcfData.update_input_desc_x(tensorDescWcf);
    wcfData.update_output_desc_y(tensorDescWcf);

    auto wcoData = op::Data("wcoData");
    std::vector<int64_t> dims_wco{32, 16};
    ge::Shape shape_wco(dims_wco);
    ge::TensorDesc tensorDescWco(shape_wco, FORMAT_NHWC,  DT_FLOAT16);
    wcoData.update_input_desc_x(tensorDescWco);
    wcoData.update_output_desc_y(tensorDescWco);

    auto maskData = op::Data("maskData");
    std::vector<int64_t> dims_mask{32, 1};
    ge::Shape shape_mask(dims_mask);
    ge::TensorDesc tensorDescMask(shape_mask, FORMAT_NHWC,  DT_FLOAT16);
    maskData.update_input_desc_x(tensorDescMask);
    maskData.update_output_desc_y(tensorDescMask);

    auto projectData = op::Data("projectData");
    std::vector<int64_t> dims_project{16, 32};
    ge::Shape shape_project(dims_project);
    ge::TensorDesc tensorDescProject(shape_project, FORMAT_NHWC,  DT_FLOAT16);
    projectData.update_input_desc_x(tensorDescProject);
    projectData.update_output_desc_y(tensorDescProject);

    auto dynamicRNNOp = op::DynamicRNNV3("DynamicRNNV3_1");
    dynamicRNNOp.set_input_x(xData)
         .set_input_w(wData)
         .set_input_b(bData)
         .set_input_init_h(initHData)
         .set_input_init_c(initCData)
         .set_input_wci(wciData)
         .set_input_wcf(wcfData)
         .set_input_wco(wcoData)
         .set_input_real_mask(maskData)
         .set_input_project(projectData)
         .set_attr_time_major(false);

    std::vector<Operator> inputs{xData, wData, bData, initHData, initCData, wciData, wcfData, wcoData, maskData, projectData};
    std::vector<Operator> outputs{dynamicRNNOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("DynamicRNNV3FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "DynamicRNNV3") {
            findTranspose = true;
        }
    }
    EXPECT_EQ(findTranspose, true);
}
