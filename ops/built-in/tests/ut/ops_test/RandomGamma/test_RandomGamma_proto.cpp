#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "random_ops.h"
#include "split_combination_ops.h"

using namespace ge;
using namespace op;

class RandomGammaTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RandomGamma test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RandomGamma test TearDown" << std::endl;
  }
};

TEST_F(RandomGammaTest, infer_shape_00) {
  ge::op::RandomGamma op;
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(RandomGammaTest, infer_shape_01) {
  ge::op::RandomGamma op;
  op.UpdateInputDesc("shape", create_desc({ge::UNKNOWN_DIM}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}