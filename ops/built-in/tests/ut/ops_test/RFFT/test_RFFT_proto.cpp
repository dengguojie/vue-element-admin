#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "spectral_ops.h"
using namespace ge;
using namespace op;

class RFFTTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RFFT test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RFFT test TearDown" << std::endl;
  }
};

TEST_F(RFFTTest, infer_shape_00) {
  ge::op::RFFT op;
  op.UpdateInputDesc("input", create_desc({}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RFFTTest, infer_shape_01) {
  ge::op::RFFT op;
  op.UpdateInputDesc("input", create_desc({1}, ge::DT_FLOAT));
  op.UpdateInputDesc("fft_length", create_desc({1, 1}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RFFTTest, infer_shape_02) {
  ge::op::RFFT op;
  op.UpdateInputDesc("input", create_desc({1}, ge::DT_FLOAT));
  op.UpdateInputDesc("fft_length", create_desc({2}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RFFTTest, infer_shape_03) {
  ge::op::RFFT op;
  op.UpdateInputDesc("input", create_desc({1}, ge::DT_FLOAT));
  op.UpdateInputDesc("fft_length", create_desc({-1}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
