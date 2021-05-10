#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "image_ops.h"
#include "nn_detect_ops.h"  

class NonMaxSuppressionWithOverlapsTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "NonMaxSuppressionWithOverlaps SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "NonMaxSuppressionWithOverlaps TearDown" << std::endl;
  }
};

TEST_F(NonMaxSuppressionWithOverlapsTest, NonMaxSuppressionWithOverlaps_infer_shape) {
  ge::op::NonMaxSuppressionWithOverlaps op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2,2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2,2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc2 = create_desc_shape_range({2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, {{}});
  auto tensor_desc3 = create_desc_shape_range({},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, {{}});

  op.UpdateInputDesc("overlaps", tensor_desc1);
  op.UpdateInputDesc("scores", tensor_desc2);
  op.UpdateInputDesc("max_output_size", tensor_desc3);
  op.UpdateInputDesc("overlap_threshold", tensor_desc3);
  op.UpdateInputDesc("score_threshold", tensor_desc3);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(NonMaxSuppressionWithOverlapsTest, NonMaxSuppressionWithOverlaps_infer_shape_overlaps_failed) {
  ge::op::NonMaxSuppressionWithOverlaps op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2,2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2,2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc2 = create_desc_shape_range({2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, {{}});
  auto tensor_desc3 = create_desc_shape_range({},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, {{}});

  op.UpdateInputDesc("overlaps", tensor_desc2);
  op.UpdateInputDesc("scores", tensor_desc2);
  op.UpdateInputDesc("max_output_size", tensor_desc3);
  op.UpdateInputDesc("overlap_threshold", tensor_desc3);
  op.UpdateInputDesc("score_threshold", tensor_desc3);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(NonMaxSuppressionWithOverlapsTest, NonMaxSuppressionWithOverlaps_infer_shape_scores_failed) {
  ge::op::NonMaxSuppressionWithOverlaps op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2,2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2,2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc2 = create_desc_shape_range({2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, {{}});
  auto tensor_desc3 = create_desc_shape_range({},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, {{}});

  op.UpdateInputDesc("overlaps", tensor_desc1);
  op.UpdateInputDesc("scores", tensor_desc1);
  op.UpdateInputDesc("max_output_size", tensor_desc3);
  op.UpdateInputDesc("overlap_threshold", tensor_desc3);
  op.UpdateInputDesc("score_threshold", tensor_desc3);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(NonMaxSuppressionWithOverlapsTest, NonMaxSuppressionWithOverlaps_infer_shape_max_output_size_failed) {
  ge::op::NonMaxSuppressionWithOverlaps op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2,2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2,2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc2 = create_desc_shape_range({2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, {{}});
  auto tensor_desc3 = create_desc_shape_range({},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, {{}});

  op.UpdateInputDesc("overlaps", tensor_desc1);
  op.UpdateInputDesc("scores", tensor_desc2);
  op.UpdateInputDesc("max_output_size", tensor_desc2);
  op.UpdateInputDesc("overlap_threshold", tensor_desc3);
  op.UpdateInputDesc("score_threshold", tensor_desc3);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(NonMaxSuppressionWithOverlapsTest, NonMaxSuppressionWithOverlaps_infer_shape_overlap_threshold_failed) {
  ge::op::NonMaxSuppressionWithOverlaps op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2,2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2,2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc2 = create_desc_shape_range({2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, {{}});
  auto tensor_desc3 = create_desc_shape_range({},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, {{}});

  op.UpdateInputDesc("overlaps", tensor_desc1);
  op.UpdateInputDesc("scores", tensor_desc2);
  op.UpdateInputDesc("max_output_size", tensor_desc3);
  op.UpdateInputDesc("overlap_threshold", tensor_desc2);
  op.UpdateInputDesc("score_threshold", tensor_desc3);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(NonMaxSuppressionWithOverlapsTest, NonMaxSuppressionWithOverlaps_infer_shape_score_threshold_failed) {
  ge::op::NonMaxSuppressionWithOverlaps op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2,2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2,2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc2 = create_desc_shape_range({2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, {{}});
  auto tensor_desc3 = create_desc_shape_range({},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, {{}});

  op.UpdateInputDesc("overlaps", tensor_desc1);
  op.UpdateInputDesc("scores", tensor_desc2);
  op.UpdateInputDesc("max_output_size", tensor_desc3);
  op.UpdateInputDesc("overlap_threshold", tensor_desc3);
  op.UpdateInputDesc("score_threshold", tensor_desc2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
