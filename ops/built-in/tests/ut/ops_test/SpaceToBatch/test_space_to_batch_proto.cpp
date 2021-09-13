#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "transformation_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class space_to_batch_infer_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "space_to_batch_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "space_to_batch_infer_test TearDown" << std::endl;
  }
};

TEST_F(space_to_batch_infer_test, space_to_batch_infer_test_1) {
  //new op
  ge::op::SpaceToBatch op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_x.SetOriginShape(ge::Shape({1, 5}));
  op.UpdateInputDesc("x", tensor_desc_x);
  ge::TensorDesc tensor_desc_paddings(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_paddings.SetOriginShape(ge::Shape({1, 5}));
  op.UpdateInputDesc("paddings", tensor_desc_paddings);
  op.SetAttr("block_size", 9);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(space_to_batch_infer_test, space_to_batch_infer_test_2) {
  //new op
  ge::op::SpaceToBatch op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_x.SetOriginShape(ge::Shape({1, 5}));
  op.UpdateInputDesc("x", tensor_desc_x);
  ge::TensorDesc tensor_desc_paddings(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_paddings.SetOriginShape(ge::Shape({1, 5}));
  op.UpdateInputDesc("paddings", tensor_desc_paddings);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(space_to_batch_infer_test, space_to_batch_infer_test_3) {
  //new op
  ge::op::SpaceToBatch op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_x.SetOriginShape(ge::Shape({1, 5}));
  op.UpdateInputDesc("x", tensor_desc_x);
  ge::TensorDesc tensor_desc_paddings(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_paddings.SetOriginShape(ge::Shape({1, 5}));
  op.UpdateInputDesc("paddings", tensor_desc_paddings);
  op.SetAttr("block_size", 9);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(space_to_batch_infer_test, space_to_batch_infer_test_4) {
  //new op
  ge::op::SpaceToBatch op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({-1, 5, 5, 7}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("x", tensor_desc_x);
  ge::TensorDesc tensor_desc_paddings(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("paddings", tensor_desc_paddings);
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(space_to_batch_infer_test, space_to_batch_infer_test_5) {
  //new op
  ge::op::SpaceToBatch op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({-1, 5, 7, 9}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("x", tensor_desc_x);
  ge::TensorDesc tensor_desc_paddings(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("paddings", tensor_desc_paddings);
  op.SetAttr("block_size", 1);
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}