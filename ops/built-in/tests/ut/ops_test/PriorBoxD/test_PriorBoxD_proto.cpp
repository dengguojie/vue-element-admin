#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"

class PriorBoxD : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "PriorBoxD Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "PriorBoxD Proto Test TearDown" << std::endl;
  }
};

TEST_F(PriorBoxD, prior_box_infershape_test_1) {
  ge::op::PriorBoxD op;

  auto tensor_desc = create_desc_with_ori({2, 3, 5}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2, 3, 5}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  auto img_desc = create_desc_with_ori({2, 3, 300, 300, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2, 3, 300, 300, 16}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("img", img_desc);
  auto data_h_desc = create_desc_with_ori({5,1,1,1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {5,1,1,1}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("data_h", data_h_desc);
  auto data_w_desc = create_desc_with_ori({5,1,1,1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {5,1,1,1}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("data_w", data_w_desc);
  auto box_height_desc = create_desc_with_ori({6,1,1,1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {6,1,1,1}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("box_height", box_height_desc);
  auto box_width_desc = create_desc_with_ori({6,1,1,1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {6,1,1,1}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("box_width", box_width_desc);

  op.SetAttr("min_size", std::vector<float>{162.0});
  op.SetAttr("max_size", std::vector<float>{213.0});
  op.SetAttr("img_h", 300);
  op.SetAttr("img_w", 300);
  op.SetAttr("step_h", 64.0f);
  op.SetAttr("step_w", 64.0f);
  op.SetAttr("flip", true);
  op.SetAttr("clip", false);
  op.SetAttr("offset", 0.5f);
  op.SetAttr("variance", std::vector<float>{0.1});

  // auto status1 = op.VerifyAllAttr(true);
  // EXPECT_EQ(status1, ge::GRAPH_SUCCESS);
  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}