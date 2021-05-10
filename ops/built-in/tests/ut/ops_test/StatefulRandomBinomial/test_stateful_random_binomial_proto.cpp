#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "stateful_random_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class StatefulRandomBinomialTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "StatefulRandomBinomialTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StatefulRandomBinomialTest TearDown" << std::endl;
  }
};

TEST_F(StatefulRandomBinomialTest, StatefulRandomBinomialTest_infer_shape_counts_failed) {
  ge::op::StatefulRandomBinomial op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2,2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2,2},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc1);
  op.UpdateInputDesc("algorithm", tensor_desc1);
  op.UpdateInputDesc("shape", tensor_desc1);
  op.UpdateInputDesc("counts", tensor_desc1);
  op.UpdateInputDesc("probs", tensor_desc1);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StatefulRandomBinomialTest, StatefulRandomBinomialTest_infer_shape_probs_failed) {
  ge::op::StatefulRandomBinomial op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2,2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2,2},
                                             ge::FORMAT_ND, shape_range);

  auto tensor_desc2 = create_desc_shape_range({},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc1);
  op.UpdateInputDesc("algorithm", tensor_desc1);
  op.UpdateInputDesc("shape", tensor_desc1);
  op.UpdateInputDesc("counts", tensor_desc2);
  op.UpdateInputDesc("probs", tensor_desc1);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StatefulRandomBinomialTest, StatefulRandomBinomialTest_infer_shape_shape_failed) {
  ge::op::StatefulRandomBinomial op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2,2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2,2},
                                             ge::FORMAT_ND, shape_range);

  auto tensor_desc2 = create_desc_shape_range({},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc1);
  op.UpdateInputDesc("algorithm", tensor_desc1);
  op.UpdateInputDesc("shape", tensor_desc1);
  op.UpdateInputDesc("counts", tensor_desc2);
  op.UpdateInputDesc("probs", tensor_desc2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StatefulRandomBinomialTest, StatefulRandomBinomialTest_infer_shape_dtype_failed) {
  ge::op::StatefulRandomBinomial op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2,2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2,2},
                                             ge::FORMAT_ND, shape_range);

  auto tensor_desc2 = create_desc_shape_range({},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc1);
  op.UpdateInputDesc("algorithm", tensor_desc1);

  ge::TensorDesc const_desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value[2] = {7, 6};
  auto const_op = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc, (uint8_t *)const_value, 2 * sizeof(int32_t)));
  op.set_input_shape(const_op);
  op.UpdateInputDesc("shape", const_desc);

  op.UpdateInputDesc("counts", tensor_desc2);
  op.UpdateInputDesc("probs", tensor_desc2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}