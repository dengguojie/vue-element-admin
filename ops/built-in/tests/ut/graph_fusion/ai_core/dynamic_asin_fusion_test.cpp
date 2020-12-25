#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;
using namespace std;

class dynamic_asin_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "dynamic_asin SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "dynamic_asin TearDown" << std::endl;
    }
};

TEST_F(dynamic_asin_fusion_test, dynamic_asin_fusion_test_1) {

    const char* cls = "dynamic_asin_fusion_test_1 ";

    cout << cls << "start !!!" << endl;

    // input x shape: [-1, -1]
    // input x range: [(1, 2), (2, 3)]
    // output y shape: [-1, -1]
    // output y range: [(1, 2), (2, 3)]

    ge::Graph graph(cls);

    // init x tensor
    ge::Shape input_x_shape(std::vector<int64_t>{-1, -1});
    ge::TensorDesc input_x_tensor_desc(input_x_shape, ge::FORMAT_NHWC, ge::DT_FLOAT16);

    std::vector<std::pair<int64_t, int64_t>> input_x_range;
    input_x_range.push_back(std::pair<int64_t, int64_t>{1, 2});
    input_x_range.push_back(std::pair<int64_t, int64_t>{2, 3});
    input_x_tensor_desc.SetShapeRange(input_x_range);

    auto input_x_data = op::Data("input_x");
    input_x_data.update_input_desc_x(input_x_tensor_desc);
    input_x_data.update_output_desc_y(input_x_tensor_desc);

    // build op input
    auto output_op = op::Asin("Asin_1");
    output_op.set_input_x(input_x_data);

    // add op into graph
    std::vector<Operator> inputs{input_x_data};
    std::vector<Operator> outputs{output_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    cout << cls << "op build success!!!" << endl;

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    // // fe::FusionPassTestUtils::RunGraphFusionPass("AMulMaximumFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    // // GE_DUMP(compute_graph_ptr, "mul_maximum_fusion_test_1_after");

    bool is_find_op = false;
    bool is_shape_match = false;
    bool is_range_match = false;

    vector<int64_t> expect_shape{-1, -1};
    std::vector<std::pair<int64_t, int64_t>> expect_range;
    expect_range.push_back(std::pair<int64_t, int64_t>{1, 2});
    expect_range.push_back(std::pair<int64_t, int64_t>{2, 3});

    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Asin") {
            is_find_op = true;
            cout << cls << "is_find_op:" << is_find_op << endl;

            auto output_desc = node->GetOpDesc()->GetOutputDesc(0);
            std::vector<int64_t> output_dims = output_desc.GetShape().GetDims();
            cout << cls << "output_dims size:" << output_dims.size()  << endl;
            cout << cls << "output_dims:[";
            for (int i = 0; i < output_dims.size(); ++i) {
                if (i < (output_dims.size() - 1)) {
                    cout << output_dims[i] << ",";
                } else {
                    cout << output_dims[i];
                }
            }
            cout << "]" << endl;
            if (output_dims == expect_shape) {
                is_shape_match = true;
            }

            std::vector<std::pair<int64_t, int64_t>> output_range;
            output_desc.GetShapeRange(output_range);
            cout << cls << "output_range size:" << output_range.size()  << std::endl;
            cout << cls << "output_range:[";
            for (int i = 0; i < output_range.size(); ++i) {
                cout << "(" << output_range[i].first << "," << output_range[i].second << ")";
            }
            cout << "]" << endl;
            if (output_range == expect_range) {
                is_range_match = false;
            }
        }
    }

    // EXPECT_EQ(is_find_op, true);
    // EXPECT_EQ(is_shape_match, true);
    // EXPECT_EQ(is_range_match, true);

    cout << cls << "end !!!" << endl;
}
