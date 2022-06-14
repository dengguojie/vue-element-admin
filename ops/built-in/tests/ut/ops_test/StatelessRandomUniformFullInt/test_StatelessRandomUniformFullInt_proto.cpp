#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "stateless_random_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class stateless_random_uniform_full_int_infer_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "stateless_random_uniform_full_int_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "stateless_random_uniform_full_int_infer_test TearDown" << std::endl;
  }
};

TEST_F(stateless_random_uniform_full_int_infer_test, stateless_random_uniform_full_int_infer_test_1) {
  // new op
  ge::op::StatelessRandomUniformFullInt op;

  ge::TensorDesc tensor_desc_seed(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT64);
  tensor_desc_seed.SetOriginShape(ge::Shape({2}));
  op.UpdateInputDesc("seed", tensor_desc_seed);

  ge::TensorDesc tensor_desc_shape(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_shape.SetOriginShape(ge::Shape({1}));
  op.UpdateInputDesc("shape", tensor_desc_shape);

  op.SetAttr("dtype", DT_INT32);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.GetOutputDescByName("y");
  std::vector<int64_t> expected_y_shape = {-1};
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_INT32);
  EXPECT_EQ(y_desc.GetShape().GetDims(), expected_y_shape);
}

TEST_F(stateless_random_uniform_full_int_infer_test, stateless_random_uniform_full_int_infer_test_2) {
  // new op
  ge::op::StatelessRandomUniformFullInt op;
  // set input info
  ge::TensorDesc tensor_desc_seed(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT64);
  tensor_desc_seed.SetOriginShape(ge::Shape({1}));
  op.UpdateInputDesc("seed", tensor_desc_seed);

  ge::TensorDesc tensor_desc_shape(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_shape.SetOriginShape(ge::Shape({1}));
  op.UpdateInputDesc("shape", tensor_desc_shape);

  op.SetAttr("dtype", DT_INT32);
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateless_random_uniform_full_int_infer_test, stateless_random_uniform_full_int_infer_test_3) {
  // new op
  ge::op::StatelessRandomUniformFullInt op;
  // set input info
  ge::TensorDesc tensor_desc_seed(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT64);
  tensor_desc_seed.SetOriginShape(ge::Shape({1, 5}));
  op.UpdateInputDesc("seed", tensor_desc_seed);

  ge::TensorDesc tensor_desc_shape(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_shape.SetOriginShape(ge::Shape({1}));
  op.UpdateInputDesc("shape", tensor_desc_shape);

  op.SetAttr("dtype", DT_INT32);
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
TEST_F(stateless_random_uniform_full_int_infer_test, stateless_random_uniform_full_int_infer_test_4) {
  // new op
  ge::op::StatelessRandomUniformFullInt op;
  // set input info
  ge::TensorDesc tensor_desc_seed(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT64);
  tensor_desc_seed.SetOriginShape(ge::Shape({2}));
  op.UpdateInputDesc("seed", tensor_desc_seed);

  ge::TensorDesc tensor_desc_shape(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_shape.SetOriginShape(ge::Shape({1}));
  op.UpdateInputDesc("shape", tensor_desc_shape);

  op.SetAttr("dtype", DT_INT64);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_INT64);
}
