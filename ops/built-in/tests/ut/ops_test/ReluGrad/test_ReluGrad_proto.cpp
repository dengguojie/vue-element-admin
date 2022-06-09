#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "nonlinear_fuc_ops.h"
#include "common/utils/ut_op_common.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"

class ReluGrad : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "ReluGrad Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "ReluGrad Proto Test TearDown" << std::endl;
    }
};

TEST_F(ReluGrad, relugrad_tsest_1) {
    ge::op::ReluGrad ReluGrad_op;
    ge::TensorDesc tensor_gradients_desc;
    ge::Shape gradients_shape({16, 16});
    tensor_gradients_desc.SetDataType(ge::DT_FLOAT16);
    tensor_gradients_desc.SetShape(gradients_shape);
    tensor_gradients_desc.SetOriginShape(gradients_shape);
    tensor_gradients_desc.SetFormat(ge::FORMAT_ND);
    ge::TensorDesc tensor_features_desc;
    ge::Shape features_shape({1, 16});
    tensor_features_desc.SetDataType(ge::DT_FLOAT16);
    tensor_features_desc.SetShape(features_shape);
    tensor_features_desc.SetOriginShape(features_shape);
    tensor_features_desc.SetFormat(ge::FORMAT_ND);
    // update input
    ReluGrad_op.UpdateInputDesc("gradients", tensor_gradients_desc);
    ReluGrad_op.UpdateInputDesc("features", tensor_features_desc);
    // infer
    auto ret = ReluGrad_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    // compare
    auto output_desc = ReluGrad_op.GetOutputDesc("backprops");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {16, 16};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
    CommonInferShapeOperator(ReluGrad_op, {}, {expected_output_shape});
    auto output_desc1 = ReluGrad_op.GetOutputDesc(0);
    EXPECT_EQ(output_desc1.GetShape().GetDims(), expected_output_shape);
}