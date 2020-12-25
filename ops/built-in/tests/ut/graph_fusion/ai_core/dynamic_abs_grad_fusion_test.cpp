#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "reduce_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "elewise_calculation_ops.h"

using namespace ge;
using namespace op;
using namespace std;

class DynamicAbsGradFusionTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "DynamicAbsGradFusionTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "DynamicAbsGradFusionTest TearDown" << std::endl;
    }
};

TEST_F(DynamicAbsGradFusionTest, DynamicAbsGradFusionTest_1) {

    const char* cls = "DynamicAbsGradFusionTest_1 ";

    cout << cls << "start !!!" << endl;

    // input x shape: [-1, -1]
    // input x range: [(1, 2), (2, 3)]
    // output y shape: [-1, -1]
    // output y range: [(1, 2), (2, 3)]

    ge::Graph graph(cls);

    // input y tensor
    auto input_y_data = op::Data("input_y");
    ge::Shape input_y_shape(std::vector<int64_t>{-1, -1});

    ge::TensorDesc input_y_tensor_desc(input_y_shape, ge::FORMAT_NHWC, ge::DT_FLOAT16);
    std::vector<std::pair<int64_t, int64_t>> input_y_range;
    input_y_range.push_back(std::pair<int64_t, int64_t>{1, 2});
    input_y_range.push_back(std::pair<int64_t, int64_t>{2, 3});
    input_y_tensor_desc.SetShapeRange(input_y_range);

    input_y_data.update_input_desc_x(input_y_tensor_desc);
    input_y_data.update_output_desc_y(input_y_tensor_desc);

    // input dy tensor
    auto input_dy_data = op::Data("input_dy");
    ge::Shape input_dy_shape(std::vector<int64_t>{-1, -1});
    ge::TensorDesc input_dy_tensor_desc(input_dy_shape, ge::FORMAT_NHWC, ge::DT_FLOAT16);
    std::vector<std::pair<int64_t, int64_t>> input_dy_range;
    input_dy_range.push_back(std::pair<int64_t, int64_t>{1, 2});
    input_dy_range.push_back(std::pair<int64_t, int64_t>{2, 3});
    input_dy_tensor_desc.SetShapeRange(input_dy_range);
    input_dy_data.update_input_desc_x(input_dy_tensor_desc);
    input_dy_data.update_output_desc_y(input_dy_tensor_desc);

    // build op input
    auto output_op = op::AbsGrad("AbsGrad_1");
    output_op.set_input_y(input_y_data);
    output_op.set_input_dy(input_dy_data);

    // add op into graph
    std::vector<Operator> inputs{input_y_data, input_dy_data};
    std::vector<Operator> outputs{output_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    cout << cls << "op build success!!!" << endl;

    ge::ComputeGraphPtr computeGraphPtr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(computeGraphPtr);

    bool isFindOp = false;
    bool isShapeMatch = false;
    bool isRangeMatch = false;

    vector<int64_t> expect_shape{-1, -1};
    std::vector<std::pair<int64_t, int64_t>> expect_range;
    expect_range.push_back(std::pair<int64_t, int64_t>{1, 2});
    expect_range.push_back(std::pair<int64_t, int64_t>{2, 3});

    for (auto node: computeGraphPtr->GetAllNodes()) {
        if (node->GetType() == "AbsGrad") {
            isFindOp = true;
            cout << cls << "isFindOp:" << isFindOp << endl;

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
                isShapeMatch = true;
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
                isRangeMatch = true;
            }
        }
    }

    EXPECT_EQ(isFindOp, true);
    EXPECT_EQ(isShapeMatch, true);
    EXPECT_EQ(isRangeMatch, true);

    cout << cls << "end !!!" << endl;
}
