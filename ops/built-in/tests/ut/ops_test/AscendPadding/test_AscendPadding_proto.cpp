#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "pad_ops.h"
#include "inference_context.h"

class ascendPadding : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ascendPadding Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ascendPadding Proto Test TearDown" << std::endl;
  }
};

TEST_F(ascendPadding, ascendPadding_infershape_input0_rank_failed){
  ge::op::AscendPadding op;
  op.UpdateInputDesc("x", create_desc({}, ge::DT_INT32));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ascendPadding, ascendPadding_infershape_input0_dim_failed){
  ge::op::AscendPadding op;
  op.UpdateInputDesc("x", create_desc({2, 3}, ge::DT_INT32));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

/*
TEST_F(ascendPadding, ascendPadding_infershape_attr_pad_dim_size_failed){
  ge::op::AscendPadding op;
  op.UpdateInputDesc("x", create_desc({2, 1}, ge::DT_INT32));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
*/

TEST_F(ascendPadding, ascendPadding_infershape_attr_pad_dim_size_value_failed){
  ge::op::AscendPadding op;
  op.UpdateInputDesc("x", create_desc({2, 1}, ge::DT_INT32));
  op.SetAttr("pad_dim_size", 0);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}