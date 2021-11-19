#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "transformation_ops.h"
#include "array_ops.h"
#include "common/utils/ut_op_util.h"

using namespace ge;
using namespace op;

class depthwiseWeight6DTo4D_proto_test:public testing::Test{
    protected:
        static void SetUpTestCase(){
            std::cout<<"depthwiseWeight6DTo4D_proto_test SetUp"<<std::endl;
        }
        static void TearDownTestCase(){
            std::cout<<"depthwiseWeight6DTo4D_proto_test TearDown"<<std::endl;
        }
};
TEST_F(depthwiseWeight6DTo4D_proto_test, depthwiseWeight6DTo4D_proto_test_1) {
    ge::op::DepthwiseWeight6DTo4D op;
        vector<vector<int64_t>> input_shapes = {
        {1, 10, 10, 4, 5, 5}
    };
    std::vector<std::pair<int64_t,int64_t>> input_range0 = {{1, 1}, {10, 10}, {10, 10}, {4, 4}};
    TENSOR_INPUT_WITH_SHAPE(op, x, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_NHWC, input_range0);
    op.set_attr_channel_size(16);
    std::vector<int64_t> expected_shape = { 10, 10, 16, 4 };
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc("y").GetShape().GetDims(), expected_shape);
    ret = op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}    