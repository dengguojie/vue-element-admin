#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "selection_ops.h"

// ----------------TopK-------------------
class InplaceTopKDistanceProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "InplaceTopKDistance Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "InplaceTopKDistance Proto Test TearDown" << std::endl;
  }
};

TEST_F(InplaceTopKDistanceProtoTest, InplaceTopK_infer_shape_suc) {
  ge::op::InplaceTopKDistance op; 
  op.UpdateInputDesc("topk_pq_distance", create_desc({10}, ge::DT_FLOAT16));
  op.UpdateInputDesc("topk_pq_index", create_desc({10}, ge::DT_INT32));
  op.UpdateInputDesc("topk_pq_ivf", create_desc({10}, ge::DT_INT32));

  op.UpdateInputDesc("pq_distance", create_desc({6}, ge::DT_FLOAT16));
  op.UpdateInputDesc("pq_index", create_desc({6}, ge::DT_INT32));
  op.UpdateInputDesc("pq_ivf", create_desc({}, ge::DT_INT32));
  op.SetAttr("order", "asc");

  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(InplaceTopKDistanceProtoTest, InplaceTopK_infer_shape_error1) {
  ge::op::InplaceTopKDistance op; 
  op.UpdateInputDesc("topk_pq_distance", create_desc({10}, ge::DT_FLOAT16));
  op.UpdateInputDesc("topk_pq_index", create_desc({10}, ge::DT_INT32));
  op.UpdateInputDesc("topk_pq_ivf", create_desc({9}, ge::DT_INT32));

  op.UpdateInputDesc("pq_distance", create_desc({6}, ge::DT_FLOAT16));
  op.UpdateInputDesc("pq_index", create_desc({6}, ge::DT_INT32));
  op.UpdateInputDesc("pq_ivf", create_desc({}, ge::DT_INT32));
  op.SetAttr("order", "asc");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(InplaceTopKDistanceProtoTest, InplaceTopK_infer_shape_error2) {
  ge::op::InplaceTopKDistance op; 
  op.UpdateInputDesc("topk_pq_distance", create_desc({10,10}, ge::DT_FLOAT16));
  op.UpdateInputDesc("topk_pq_index", create_desc({10,10}, ge::DT_INT32));
  op.UpdateInputDesc("topk_pq_ivf", create_desc({10,10}, ge::DT_INT32));

  op.UpdateInputDesc("pq_distance", create_desc({6}, ge::DT_FLOAT16));
  op.UpdateInputDesc("pq_index", create_desc({6}, ge::DT_INT32));
  op.UpdateInputDesc("pq_ivf", create_desc({}, ge::DT_INT32));
  op.SetAttr("order", "asc");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(InplaceTopKDistanceProtoTest, InplaceTopK_infer_shape_error3) {
  ge::op::InplaceTopKDistance op; 
  op.UpdateInputDesc("topk_pq_distance", create_desc({10}, ge::DT_FLOAT16));
  op.UpdateInputDesc("topk_pq_index", create_desc({10}, ge::DT_INT32));
  op.UpdateInputDesc("topk_pq_ivf", create_desc({10}, ge::DT_INT32));

  op.UpdateInputDesc("pq_distance", create_desc({6}, ge::DT_FLOAT16));
  op.UpdateInputDesc("pq_index", create_desc({5}, ge::DT_INT32));
  op.UpdateInputDesc("pq_ivf", create_desc({}, ge::DT_INT32));
  op.SetAttr("order", "asc");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(InplaceTopKDistanceProtoTest, InplaceTopK_infer_shape_error4) {
  ge::op::InplaceTopKDistance op; 
  op.UpdateInputDesc("topk_pq_distance", create_desc({10}, ge::DT_FLOAT16));
  op.UpdateInputDesc("topk_pq_index", create_desc({10}, ge::DT_INT32));
  op.UpdateInputDesc("topk_pq_ivf", create_desc({10}, ge::DT_INT32));

  op.UpdateInputDesc("pq_distance", create_desc({6,6}, ge::DT_FLOAT16));
  op.UpdateInputDesc("pq_index", create_desc({6,6}, ge::DT_INT32));
  op.UpdateInputDesc("pq_ivf", create_desc({}, ge::DT_INT32));
  op.SetAttr("order", "asc");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}