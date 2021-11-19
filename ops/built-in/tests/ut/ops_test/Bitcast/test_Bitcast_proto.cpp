#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "transformation_ops.h"
#include "array_ops.h"
#include "common/utils/ut_op_util.h"

using namespace ge;
using namespace op;

class bitcast_test:public testing::Test{
    protected:
        static void SetUpTestCase(){
            std::cout<<"bitcast_test SetUp"<<std::endl;
        }
        static void TearDownTestCase(){
            std::cout<<"bitcast_test TearDown"<<std::endl;
        }
};
TEST_F(bitcast_test, bitcast_test_1) {
    ge::op::Bitcast op;
        vector<vector<int64_t>> input_shapes = {
        {1, 10, 10, 4}
    };
    std::vector<std::pair<int64_t,int64_t>> input_range0 = {{1, 1}, {10, 10}, {10, 10}, {4, 4}};
    TENSOR_INPUT_WITH_SHAPE(op, x, input_shapes[0], ge::DT_INT32, ge::FORMAT_NHWC, input_range0);
    op.set_attr_type(ge::DT_FLOAT16);
    std::vector<int64_t> expected_shape = {1, 10, 10, 4, 2};
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc("y").GetShape().GetDims(), expected_shape);
    ret = op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
TEST_F(bitcast_test, bitcast_test_2) {
    ge::op::Bitcast op;
        vector<vector<int64_t>> input_shapes = {
        {1, 10, 10, 4}
    };
    std::vector<std::pair<int64_t,int64_t>> input_range0 = {{1, 1}, {10, 10}, {10, 10}, {4, 4}};
    TENSOR_INPUT_WITH_SHAPE(op, x, input_shapes[0], ge::DT_INT32, ge::FORMAT_NHWC, input_range0);
    op.set_attr_type(ge::DT_DUAL);
    std::vector<int64_t> expected_shape = {1, 10, 10, 4};
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
TEST_F(bitcast_test, bitcast_test_3) {
    ge::op::Bitcast op;
        vector<vector<int64_t>> input_shapes = {
        {1, 10, 10, 4}
    };
    std::vector<std::pair<int64_t,int64_t>> input_range0 = {{1, 1}, {10, 10}, {10, 10}, {4, 4}};
    TENSOR_INPUT_WITH_SHAPE(op, x, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_NHWC, input_range0);
    op.set_attr_type(ge::DT_FLOAT16);
    std::vector<int64_t> expected_shape = {1, 10, 10, 4};
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
TEST_F(bitcast_test, bitcast_test_4) {
    ge::op::Bitcast op;
        vector<vector<int64_t>> input_shapes = {
        {1, 10, 10, 4}
    };
    std::vector<std::pair<int64_t,int64_t>> input_range0 = {{1, 1}, {10, 10}, {10, 10}, {4, 4}};
    TENSOR_INPUT_WITH_SHAPE(op, x, input_shapes[0], ge::DT_DUAL, ge::FORMAT_NHWC, input_range0);
    op.set_attr_type(ge::DT_INT32);
    std::vector<int64_t> expected_shape = {1, 10, 10, 4};
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
TEST_F(bitcast_test, bitcast_test_5) {
    ge::op::Bitcast op;
        vector<vector<int64_t>> input_shapes = {
        {}
    };
    std::vector<std::pair<int64_t,int64_t>> input_range0 = {{1, 1}, {10, 10}, {10, 10}, {4, 4}};
    TENSOR_INPUT_WITH_SHAPE(op, x, input_shapes[0], ge::DT_DUAL, ge::FORMAT_NHWC, input_range0);
    op.set_attr_type(ge::DT_INT32);
    std::vector<int64_t> expected_shape = {};
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
TEST_F(bitcast_test, bitcast_test_6) {
    ge::op::Bitcast op;
        vector<vector<int64_t>> input_shapes = {
        {1, 10, 10, 4}
    };
    std::vector<std::pair<int64_t,int64_t>> input_range0 = {{1, 1}, {10, 10}, {10, 10}, {4, 4}};
    TENSOR_INPUT_WITH_SHAPE(op, x, input_shapes[0], ge::DT_INT32, ge::FORMAT_NHWC, input_range0);
    std::vector<int64_t> expected_shape = {1, 10, 10, 4};
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}




