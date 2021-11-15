#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "array_ops.h"
#include "op_proto_test_util.h"
#include "graph/utils/tensor_utils.h"
#include "nn_batch_norm_ops.h"
#include "common/utils/ut_op_util.h"

using namespace ge;
using namespace op;

class l2NormalizeGrad_test : public testing::Test {
  protected:
  static void SetUpTestCase() {
    std::cout << "l2NormalizeGrad_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "l2NormalizeGrad_test TearDown" << std::endl;
  }
};
TEST_F(l2NormalizeGrad_test, l2NormalizeGrad_test_1) {
    ge::op::L2NormalizeGrad op;
        vector<vector<int64_t>> input_shapes = {
        {10, 11, 11,10},
        {2, 2, 2, 2},
        {3, 3, 3, 3}
    };
    std::vector<std::pair<int64_t,int64_t>> input_range0 = {{10, 10}, {11, 11}, {11, 11}, {10, 10}};
    std::vector<std::pair<int64_t,int64_t>> input_range1 = {{2, 2}, {2, 2}, {2, 2}, {2, 2}};
    std::vector<std::pair<int64_t,int64_t>> input_range2 = {{3, 3}, {3, 3}, {3, 3}, {3, 3}};
    vector<ge::DataType> dtypes = {ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16};
    TENSOR_INPUT_WITH_SHAPE(op, x, input_shapes[0], dtypes[0], ge::FORMAT_NHWC, input_range0);
    TENSOR_INPUT_WITH_SHAPE(op, y, input_shapes[1], dtypes[1], ge::FORMAT_NHWC, input_range1);
    TENSOR_INPUT_WITH_SHAPE(op, dy, input_shapes[2], dtypes[2], ge::FORMAT_NHWC, input_range2);
    op.set_attr_dim({5});
    op.set_attr_eps(0.001);
    std::vector<int64_t> expected_shape = {10, 11, 11,10};
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc("dx").GetShape().GetDims(), expected_shape);
}
