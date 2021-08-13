#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "selection_ops.h"

class TopKPQDistanceMerge : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TopKPQDistanceMerge Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TopKPQDistanceMerge Proto Test TearDown" << std::endl;
  }
};

TEST_F(TopKPQDistanceMerge, TopKPQDistanceMerge_infershape_test_1){
  ge::op::TopKPQDistanceMerge op;
  op.UpdateInputDesc("sorted_distance", create_desc({52}, ge::DT_FLOAT16));
  op.UpdateInputDesc("pq_ivf", create_desc({52}, ge::DT_INT32));
  op.UpdateInputDesc("pq_index", create_desc({52}, ge::DT_INT32));
  int k = 26;
  op.SetAttr("k", k);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  std::vector<int64_t> expectedOutputShape = {26};
  auto outputDesc1 = op.GetOutputDesc("topk_distance");
  auto outputDesc2 = op.GetOutputDesc("topk_ivf");
  auto outputDesc3 = op.GetOutputDesc("topk_index");
  EXPECT_EQ(outputDesc1.GetShape().GetDims(), expectedOutputShape);
  EXPECT_EQ(outputDesc2.GetShape().GetDims(), expectedOutputShape);
  EXPECT_EQ(outputDesc3.GetShape().GetDims(), expectedOutputShape);
}

TEST_F(TopKPQDistanceMerge, TopKPQDistanceMerge_infershape_test_diff_shape_failed){
  ge::op::TopKPQDistanceMerge op;
  op.UpdateInputDesc("sorted_distance", create_desc({100}, ge::DT_FLOAT16));
  op.UpdateInputDesc("pq_ivf", create_desc({52}, ge::DT_INT32));
  op.UpdateInputDesc("pq_index", create_desc({200}, ge::DT_INT32));
  int k = 26;
  op.SetAttr("k", k);

  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(TopKPQDistanceMerge, TopKPQDistanceMerge_infershape_test_invalid_k_failed){
  ge::op::TopKPQDistanceMerge op;
  op.UpdateInputDesc("sorted_distance", create_desc({52}, ge::DT_FLOAT16));
  op.UpdateInputDesc("pq_ivf", create_desc({52}, ge::DT_INT32));
  op.UpdateInputDesc("pq_index", create_desc({52}, ge::DT_INT32));
  int k = 1025;
  op.SetAttr("k", k);

  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
