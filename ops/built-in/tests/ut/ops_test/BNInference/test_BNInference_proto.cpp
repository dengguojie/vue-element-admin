#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "array_ops.h"
#include "op_proto_test_util.h"
#include "graph/utils/tensor_utils.h"
#include "common/utils/ut_op_util.h"
#include "nn_batch_norm_ops.h"


using namespace ge;
using namespace op;

class BNInference_test : public testing::Test {
  protected:
  static void SetUpTestCase() {
    std::cout << "BNInference_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BNInference_test TearDown" << std::endl;
  }
};
TEST_F(BNInference_test, BNInference_test_1) {
    ge::op::BNInference op;
        vector<vector<int64_t>> input_shapes = {
        {10, 11, 11,10},
        {2, 2, 2, 2},
        {3, 3, 3, 3},
        {4, 4, 4, 4},
        {5, 5, 5, 5},
        {6, 6, 6, 6}
    };
    std::vector<std::pair<int64_t,int64_t>> input_range0 = {{10, 10}, {11, 11}, {11, 11}, {10, 10}};
    std::vector<std::pair<int64_t,int64_t>> input_range1 = {{2, 2}, {2, 2}, {2, 2}, {2, 2}};
    std::vector<std::pair<int64_t,int64_t>> input_range2 = {{3, 3}, {3, 3}, {3, 3}, {3, 3}};
    std::vector<std::pair<int64_t,int64_t>> input_range3 = {{4, 4}, {4, 4}, {4, 4}, {4, 4}};
    std::vector<std::pair<int64_t,int64_t>> input_range4 = {{5, 5}, {5, 5}, {5, 5}, {5, 5}};
    std::vector<std::pair<int64_t,int64_t>> input_range5 = {{6, 6}, {6, 6}, {6, 6}, {6, 6}};
    vector<ge::DataType> dtypes = {ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16};
    TENSOR_INPUT_WITH_SHAPE(op, x, input_shapes[0], dtypes[0], ge::FORMAT_NHWC, input_range0);
    TENSOR_INPUT_WITH_SHAPE(op, mean, input_shapes[1], dtypes[1], ge::FORMAT_NHWC, input_range1);
    TENSOR_INPUT_WITH_SHAPE(op, variance, input_shapes[2], dtypes[2], ge::FORMAT_NHWC, input_range2);
    TENSOR_INPUT_WITH_SHAPE(op, momentum, input_shapes[3], dtypes[3], ge::FORMAT_NHWC, input_range3);
    TENSOR_INPUT_WITH_SHAPE(op, scale, input_shapes[4], dtypes[4], ge::FORMAT_NHWC, input_range4);
    TENSOR_INPUT_WITH_SHAPE(op, offset, input_shapes[5], dtypes[5], ge::FORMAT_NHWC, input_range5);
    op.set_attr_epsilon(0.001);
    op.set_attr_mode(true);
    op.set_attr_use_global_stats(2);
    std::vector<int64_t> expected_shape = {10, 11, 11,10};
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc("y").GetShape().GetDims(), expected_shape);
    ret = op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
