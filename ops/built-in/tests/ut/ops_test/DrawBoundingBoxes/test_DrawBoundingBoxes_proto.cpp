#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "image_ops.h"

class DrawBoundingBoxesTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "drawboundingboxes SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "drawboundingboxes TearDown" << std::endl;
  }
};

TEST_F(DrawBoundingBoxesTest, infershape_00) {
  ge::op::DrawBoundingBoxes op;
  op.UpdateInputDesc("images", create_desc({3,5,5,1}, ge::DT_INT32));
  op.UpdateInputDesc("boxes", create_desc({1,1,4}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(DrawBoundingBoxesTest, infershape_01) {
  ge::op::DrawBoundingBoxes op;
  op.UpdateInputDesc("images", create_desc({3,5,5,1}, ge::DT_INT32));
  op.UpdateInputDesc("boxes", create_desc({1,1,2}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}