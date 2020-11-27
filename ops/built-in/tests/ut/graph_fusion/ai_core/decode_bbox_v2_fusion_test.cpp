#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_detect_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class decode_bbox_v2_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "decode_bbox_v2_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "decode_bbox_v2_fusion_test TearDown" << std::endl;
    }
};

// TEST_F(decode_bbox_v2_fusion_test, decode_bbox_v2_fusion_test_1) {
//     ge::Graph graph("decode_bbox_v2_fusion_test_1");
// 
//     auto boxesData = op::Data("boxesData");
//     std::vector<int64_t> dims_x{1024, 4};
//     ge::Shape shape_x(dims_x);
//     ge::TensorDesc tensorDescX(shape_x, FORMAT_ND,  DT_FLOAT16);
//     boxesData.update_input_desc_x(tensorDescX);
//     boxesData.update_output_desc_y(tensorDescX);
// 
//     auto anchorsData = op::Data("anchorsData");
//     std::vector<int64_t> dims_score{1024, 4};
//     ge::Shape shape_score(dims_score);
//     ge::TensorDesc tensorDescScore(shape_score, FORMAT_ND,  DT_FLOAT16);
//     anchorsData.update_input_desc_x(tensorDescScore);
//     anchorsData.update_output_desc_y(tensorDescScore);
// 
//     auto decodeOp = op::DecodeBboxV2("DecodeBboxV2_1");
//     decodeOp.set_input_boxes(boxesData);
//     decodeOp.set_input_anchors(anchorsData);
//     decodeOp.set_attr_decode_clip(0.0);
//     decodeOp.set_attr_scales({1.0, 1.0, 1.0, 1.0});
//     decodeOp.set_attr_reversed_box(false);
// 
//     std::vector<Operator> inputs{boxesData, anchorsData};
//     std::vector<Operator> outputs{decodeOp};
// 
//     graph.SetInputs(inputs).SetOutputs(outputs);
//     ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
//     fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
//     fe::FusionPassTestUtils::RunGraphFusionPass("DecodeBboxV2InsertTransposePass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
// 
//     bool findTranspose = false;
//     for (auto node: compute_graph_ptr->GetAllNodes()) {
//         if (node->GetType() == "TransposeD") {
//             findTranspose = true;
//         }
//     }
//     EXPECT_EQ(findTranspose, true);
// }
