#include <gtest/gtest.h>

#include <iostream>

#include "image_ops.h"
#include "op_proto_test_util.h"

class IMGWarpOffsets : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "IMGWarpOffsets SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "IMGWarpOffsets TearDown" << std::endl;
  }
};

TEST_F(IMGWarpOffsets, IMGWarpOffsets_infershape_success) {
  ge::op::IMGWarpOffsets op;
  op.UpdateInputDesc("images", create_desc({1, 2, 2, 3}, ge::DT_FLOAT16));
  op.UpdateInputDesc("offsets", create_desc({1, 4, 2, 2}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(IMGWarpOffsets, IMGWarpOffsets_infershape_input0_failed) {
  ge::op::IMGWarpOffsets op;
  op.UpdateInputDesc("images", create_desc({1, 2, 2}, ge::DT_FLOAT16));
  op.UpdateInputDesc("offsets", create_desc({1, 4, 2, 2}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(IMGWarpOffsets, IMGWarpOffsets_infershape_input0_dim_failed) {
  ge::op::IMGWarpOffsets op;
  op.UpdateInputDesc("images", create_desc({1, 2, 2, 2}, ge::DT_FLOAT16));
  op.UpdateInputDesc("offsets", create_desc({1, 4, 2, 2}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(IMGWarpOffsets, IMGWarpOffsets_infershape_input1_failed) {
  ge::op::IMGWarpOffsets op;
  op.UpdateInputDesc("images", create_desc({1, 2, 2, 3}, ge::DT_FLOAT16));
  op.UpdateInputDesc("offsets", create_desc({1, 4, 2}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(IMGWarpOffsets, IMGWarpOffsets_infershape_input1_dim_failed) {
  ge::op::IMGWarpOffsets op;
  op.UpdateInputDesc("images", create_desc({1, 2, 2, 3}, ge::DT_FLOAT16));
  op.UpdateInputDesc("offsets", create_desc({1, 2, 4, 2}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(IMGWarpOffsets, IMGWarpOffsets_infershape_input0_input1_dim_failed) {
  ge::op::IMGWarpOffsets op;
  op.UpdateInputDesc("images", create_desc({2, 2, 2, 3}, ge::DT_FLOAT16));
  op.UpdateInputDesc("offsets", create_desc({1, 4, 2, 2}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}