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

class l2Normalize_test : public testing::Test {
  protected:
  static void SetUpTestCase() {
    std::cout << "l2Normalize_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "l2Normalize_test TearDown" << std::endl;
  }
};
TEST_F(l2Normalize_test, l2Normalize_test_1) {
    ge::op::L2Normalize op;
    vector<int64_t> input_shapes = {10};
    std::vector<std::pair<int64_t,int64_t>> input_range0 = {{10, 10}};
    ge::DataType dtype = ge::DT_FLOAT16;
    TENSOR_INPUT_WITH_SHAPE(op, x, input_shapes, dtype, ge::FORMAT_NHWC, input_range0);
    op.set_attr_axis({1});
    op.set_attr_eps(1e-5);
    std::vector<int64_t> expected_shape = {10};
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc("y").GetShape().GetDims(), expected_shape);
}