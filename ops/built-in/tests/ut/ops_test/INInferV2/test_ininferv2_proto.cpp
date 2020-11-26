#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "reduce_ops.h"

class in_infer_v2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "in_infer_v2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "in_infer_v2 TearDown" << std::endl;
  }
};


TEST_F(in_infer_v2, in_infer_v2_infershape_diff_test_1) {
  ge::op::INInferV2 op;

  op.UpdateInputDesc("x", create_desc_with_ori({4, 1, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0,{4, 1, 64, 64, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("gamma", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("beta", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("mean", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("variance", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));

  op.SetAttr("epsilon", (float)0.00001);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

}

TEST_F(in_infer_v2, in_infer_v2_infershape_diff_test_2) {
  ge::op::INInferV2 op;

  op.UpdateInputDesc("x", create_desc_with_ori({4, 1, 100, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0,{4, 1, 100, 64, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("gamma", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("beta", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("mean", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("variance", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));

  op.SetAttr("epsilon", (float)0.00001);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

}

TEST_F(in_infer_v2, in_infer_v2_infershape_diff_test_3) {
  ge::op::INInferV2 op;

  op.UpdateInputDesc("x", create_desc_with_ori({4, 1, 100, 64, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 100, 64, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("gamma", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("beta", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("mean", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("variance", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));

  op.SetAttr("epsilon", (float)0.00001);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

}

TEST_F(in_infer_v2, in_infer_v2_infershape_diff_test_4) {
  ge::op::INInferV2 op;

  op.UpdateInputDesc("x", create_desc_with_ori({5, 1, 100, 64, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 1, 100, 64, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("gamma", create_desc_with_ori({5, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("beta", create_desc_with_ori({5, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("mean", create_desc_with_ori({5, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("variance", create_desc_with_ori({5, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));

  op.SetAttr("epsilon", (float)0.00001);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

}

TEST_F(in_infer_v2, in_infer_v2_infershape_diff_test_5) {
  ge::op::INInferV2 op;

  op.UpdateInputDesc("x", create_desc_with_ori({5, 3, 1, 64, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 3, 1, 64, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("gamma", create_desc_with_ori({5, 3, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 3, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("beta", create_desc_with_ori({5, 3, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 3, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("mean", create_desc_with_ori({5, 3, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 3, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("variance", create_desc_with_ori({5, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 3, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));

  op.SetAttr("epsilon", (float)0.00001);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

}