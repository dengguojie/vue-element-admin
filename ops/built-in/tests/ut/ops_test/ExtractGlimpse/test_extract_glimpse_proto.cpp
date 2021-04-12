#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "image_ops.h"

class ExtractGlimpse : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ExtractGlimpse SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ExtractGlimpse TearDown" << std::endl;
  }
};

TEST_F(ExtractGlimpse, extract_glimpse_infershape_err_test_1){
  ge::op::ExtractGlimpse op;
  op.UpdateInputDesc("x", create_desc({16, 16, 16, 16}, ge::DT_FLOAT));
  op.UpdateInputDesc("size", create_desc({}, ge::DT_FLOAT));
  op.UpdateInputDesc("offsets", create_desc({2, 2}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ExtractGlimpse, extract_glimpse_infershape_err_test_2){
  ge::op::ExtractGlimpse op;
  op.UpdateInputDesc("x", create_desc({16, 16, 16, 16}, ge::DT_FLOAT));
  op.UpdateInputDesc("size", create_desc({3}, ge::DT_FLOAT));
  op.UpdateInputDesc("offsets", create_desc({2, 2}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
