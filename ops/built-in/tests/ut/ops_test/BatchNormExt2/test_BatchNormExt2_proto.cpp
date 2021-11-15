#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "array_ops.h"
#include "op_proto_test_util.h"
#include "nn_batch_norm_ops.h"
#include "common/utils/ut_op_util.h"

using namespace ge;
using namespace op;

class batchNormExt2_test : public testing::Test {
  protected:
  static void SetUpTestCase() {
    std::cout << "batchNormExt2_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "batchNormExt2_test TearDown" << std::endl;
  }
};
TEST_F(batchNormExt2_test, batchNormExt2_test_1) {
    ge::op::BatchNormExt2 op;
    vector<vector<int64_t>> input_shapes = {
        {10, 10, 10,10},
        {2, 2, 2, 2},
        {3, 3, 3, 3},
        {4, 4, 4, 4},
        {4, 4, 4, 4}
    };
    std::vector<std::pair<int64_t,int64_t>> input_range0 = {{10, 10}, {10, 10}, {10, 10}, {10, 10}};
    std::vector<std::pair<int64_t,int64_t>> input_range1 = {{2, 2}, {2, 2}, {2, 2}, {2, 2}};
    std::vector<std::pair<int64_t,int64_t>> input_range2 = {{3, 3}, {3, 3}, {3, 3}, {3, 3}};
    std::vector<std::pair<int64_t,int64_t>> input_range3 = {{4, 4}, {4, 4}, {4, 4}, {4, 4}};
    std::vector<std::pair<int64_t,int64_t>> input_range4 = {{4, 4}, {4, 4}, {4, 4}, {4, 4}};
    vector<ge::DataType> dtypes = {ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT};
    TENSOR_INPUT_WITH_SHAPE(op, input_x, input_shapes[0], dtypes[0], ge::FORMAT_NHWC, input_range0);
    TENSOR_INPUT_WITH_SHAPE(op, input_scale, input_shapes[1], dtypes[1], ge::FORMAT_NHWC, input_range1);
    TENSOR_INPUT_WITH_SHAPE(op, input_offset, input_shapes[2], dtypes[2], ge::FORMAT_NHWC, input_range2);
    TENSOR_INPUT_WITH_SHAPE(op, input_mean, input_shapes[3], dtypes[3], ge::FORMAT_NHWC, input_range3);
    TENSOR_INPUT_WITH_SHAPE(op, input_variance, input_shapes[4], dtypes[4], ge::FORMAT_NHWC, input_range4);
    op.set_attr_data_format("NHWC");
    op.set_attr_epsilon(0.01);
    op.set_attr_is_training(true);
    std::vector<int64_t> expected_shape = {10, 10, 10,10};
    std::vector<int64_t> expected_shape1 = {2, 2, 2, 2};
    std::vector<int64_t> expected_shape2 = {2, 2, 2, 2};
    std::vector<int64_t> expected_shape3 = {2, 2, 2, 2};
    std::vector<int64_t> expected_shape4 = {2, 2, 2, 2};
    auto ret = op.InferShapeAndType();
    // check result
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc("output_y").GetShape().GetDims(), expected_shape);
    EXPECT_EQ(op.GetOutputDesc("output_mean").GetShape().GetDims(), expected_shape1);
    EXPECT_EQ(op.GetOutputDesc("output_variance").GetShape().GetDims(), expected_shape2);
    EXPECT_EQ(op.GetOutputDesc("output_reserve_space_1").GetShape().GetDims(), expected_shape2);
    EXPECT_EQ(op.GetOutputDesc("output_reserve_space_2").GetShape().GetDims(), expected_shape2);

    ret = op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
TEST_F(batchNormExt2_test, batchNormExt2_test_2) {
    ge::op::BatchNormExt2 op;
    vector<vector<int64_t>> input_shapes = {
        {10, 10, 10,10},
        {2, 2, 2, 2},
        {3, 3, 3, 3},
        {4, 4, 4, 4},
        {4, 4, 4, 4}
    };
    std::vector<std::pair<int64_t,int64_t>> input_range0 = {{10, 10}, {10, 10}, {10, 10}, {10, 10}};
    std::vector<std::pair<int64_t,int64_t>> input_range1 = {{2, 2}, {2, 2}, {2, 2}, {2, 2}};
    std::vector<std::pair<int64_t,int64_t>> input_range2 = {{3, 3}, {3, 3}, {3, 3}, {3, 3}};
    std::vector<std::pair<int64_t,int64_t>> input_range3 = {{4, 4}, {4, 4}, {4, 4}, {4, 4}};
    std::vector<std::pair<int64_t,int64_t>> input_range4 = {{4, 4}, {4, 4}, {4, 4}, {4, 4}};
    vector<ge::DataType> dtypes = {ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT};
    TENSOR_INPUT_WITH_SHAPE(op, input_x, input_shapes[0], dtypes[0], ge::FORMAT_NHWC, input_range0);
    TENSOR_INPUT_WITH_SHAPE(op, input_scale, input_shapes[1], ge::DT_FLOAT16, ge::FORMAT_NHWC, input_range1);
    TENSOR_INPUT_WITH_SHAPE(op, input_offset, input_shapes[2], dtypes[2], ge::FORMAT_NHWC, input_range2);
    TENSOR_INPUT_WITH_SHAPE(op, input_mean, input_shapes[3], dtypes[3], ge::FORMAT_NHWC, input_range3);
    TENSOR_INPUT_WITH_SHAPE(op, input_variance, input_shapes[4], dtypes[4], ge::FORMAT_NHWC, input_range4);
    op.set_attr_data_format("ND");
    op.set_attr_epsilon(0.01);
    op.set_attr_is_training(true);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
    ret = op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}