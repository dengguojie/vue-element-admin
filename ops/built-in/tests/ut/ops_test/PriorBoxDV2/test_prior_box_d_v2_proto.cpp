#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"

class PriorBoxDV2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "PriorBoxDV2 Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "PriorBoxDV2 Proto Test TearDown" << std::endl;
  }
};

TEST_F(PriorBoxDV2, prior_box_infershape_test_1) {
  ge::op::PriorBoxDV2 op;

  auto tensor_desc = create_desc_with_ori({2, 16, 5, 5}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2, 16, 5, 5}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  auto img_desc = create_desc_with_ori({2, 16, 300, 300}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2, 16, 300, 300}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("img", img_desc);
  auto box_desc = create_desc_with_ori({1, 2, 160,1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 2, 160,1}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("boxes", box_desc);

  op.SetAttr("min_size", std::vector<float>{30.0});
  op.SetAttr("max_size", std::vector<float>{60.0});
  op.SetAttr("img_h", 300);
  op.SetAttr("img_w", 300);
  op.SetAttr("step_h", 8.0f);
  op.SetAttr("step_w", 8.0f);
  op.SetAttr("flip", true);
  op.SetAttr("clip", false);
  op.SetAttr("offset", 0.5f);
  op.SetAttr("variance", std::vector<float>{0.1, 0.1, 0.2, 0.2});

  // auto status1 = op.VerifyAllAttr(true);
  // EXPECT_EQ(status1, ge::GRAPH_SUCCESS);
  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(PriorBoxDV2, prior_box_infershape_test_2) {
  ge::op::PriorBoxDV2 op;

  auto tensor_desc = create_desc_with_ori({2, 16, 5, 5}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2, 16, 5, 5}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  auto img_desc = create_desc_with_ori({2, 16, 300, 300}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2, 16, 300, 300}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("img", img_desc);
  auto box_desc = create_desc_with_ori({1, 2, 160,1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 2, 160,1}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("boxes", box_desc);

  op.SetAttr("min_size", std::vector<float>{30.0});
  op.SetAttr("max_size", std::vector<float>{60.0});
  op.SetAttr("img_h", 300);
  op.SetAttr("img_w", 300);
  op.SetAttr("step_h", 8.0f);
  op.SetAttr("step_w", 8.0f);
  op.SetAttr("flip", true);
  op.SetAttr("clip", false);
  op.SetAttr("offset", 0.5f);
  op.SetAttr("variance", std::vector<float>{0.1});

  // auto status1 = op.VerifyAllAttr(true);
  // EXPECT_EQ(status1, ge::GRAPH_SUCCESS);
  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(PriorBoxDV2, prior_box_infershape_test_3) {
  ge::op::PriorBoxDV2 op;

  auto tensor_desc = create_desc_with_ori({2, 16, 5}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2, 16, 5}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  auto img_desc = create_desc_with_ori({2, 16, 300, 300}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2, 16, 300, 300}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("img", img_desc);
  auto box_desc = create_desc_with_ori({1, 2, 160,1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 2, 160,1}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("boxes", box_desc);

  op.SetAttr("min_size", std::vector<float>{30.0});
  op.SetAttr("max_size", std::vector<float>{60.0});
  op.SetAttr("img_h", 300);
  op.SetAttr("img_w", 300);
  op.SetAttr("step_h", 8.0f);
  op.SetAttr("step_w", 8.0f);
  op.SetAttr("flip", true);
  op.SetAttr("clip", false);
  op.SetAttr("offset", 0.5f);
  op.SetAttr("variance", std::vector<float>{0.1});

  // auto status1 = op.VerifyAllAttr(true);
  // EXPECT_EQ(status1, ge::GRAPH_SUCCESS);
  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// TEST_F(PriorBoxDV2, prior_box_infershape_test_4) {
//   ge::op::PriorBoxDV2 op;

//   auto tensor_desc = create_desc_with_ori({2, 16, 5, 5}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2, 16, 5, 5}, ge::FORMAT_NCHW);
//   op.UpdateInputDesc("x", tensor_desc);
//   auto img_desc = create_desc_with_ori({2, 16, 300, 300}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2, 16, 300, 300}, ge::FORMAT_NCHW);
//   op.UpdateInputDesc("img", img_desc);
//   auto box_desc = create_desc_with_ori({1, 2, 160,1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 2, 160,1}, ge::FORMAT_NCHW);
//   op.UpdateInputDesc("boxes", box_desc);

//   op.SetAttr("min_size", std::vector<float>{30.0});
//   op.SetAttr("max_size", std::vector<float>{60.0});
//   op.SetAttr("img_h", 300);
//   op.SetAttr("img_w", 300);
//   op.SetAttr("step_h", 8.0f);
//   op.SetAttr("step_w", 8.0f);
//   op.SetAttr("flip", true);
//   op.SetAttr("clip", false);
//   op.SetAttr("offset", 0.5f);
//   op.SetAttr("variance", std::vector<float>{0.0});

//   // auto status1 = op.VerifyAllAttr(true);
//   // EXPECT_EQ(status1, ge::GRAPH_SUCCESS);
//   auto status = op.InferShapeAndType();
//   EXPECT_EQ(status, ge::GRAPH_FAILED);
// }