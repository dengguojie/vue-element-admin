#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"


class KLDivTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "KLDiv Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "KLDiv Proto Test TearDown" << std::endl;
  }
};

TEST_F(KLDivTest, kl_div_infershape_test) {

  ge::op::KLDiv op;

  op.UpdateInputDesc("x", create_desc({16, 2, 16, 16}, ge::DT_FLOAT));
  op.UpdateInputDesc("target", create_desc({16, 2, 16, 16}, ge::DT_FLOAT));
  op.SetAttr("reduction", "sum");

  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto out_var_desc = op.GetOutputDesc("y");
  EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_FLOAT);
  
  std::vector<int64_t> expected_var_output_shape = {};
  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);
}

TEST_F(KLDivTest, kl_div_infershape_test_02) {

  ge::op::KLDiv op;

  op.UpdateInputDesc("x", create_desc({16, 2, 16, 16}, ge::DT_FLOAT));
  op.UpdateInputDesc("target", create_desc({16, 2, 16, 16}, ge::DT_FLOAT));
  op.SetAttr("reduction", "none");

  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto out_var_desc = op.GetOutputDesc("y");
  EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_FLOAT);
  
  std::vector<int64_t> expected_var_output_shape = {16, 2, 16, 16};
  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);
}

TEST_F(KLDivTest, kl_div_verify_success_test_01) {

  ge::op::KLDiv op;

  op.UpdateInputDesc("x", create_desc({16, 2, 16, 16}, ge::DT_FLOAT));
  op.UpdateInputDesc("target", create_desc({16, 2, 16, 16}, ge::DT_FLOAT));
  op.SetAttr("reduction", "batchmean");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(KLDivTest, kl_div_verify_success_test_02) {

  ge::op::KLDiv op;

  op.UpdateInputDesc("x", create_desc({16, 2, 16, 16}, ge::DT_FLOAT));
  op.UpdateInputDesc("target", create_desc({16, 2, 16, 16}, ge::DT_FLOAT));
  op.SetAttr("reduction", "batchmean");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(KLDivTest, kl_div_infershape_test_dynamic_01){
    
    ge::op::KLDiv op;

    auto x_desc = create_desc_shape_range({-1,}, ge::DT_FLOAT16, ge::FORMAT_ND, {1,}, ge::FORMAT_ND, {{1, 1},});
    op.UpdateInputDesc("x", x_desc);

    auto target_desc = create_desc_shape_range({-1,}, ge::DT_FLOAT16, ge::FORMAT_ND, {1,}, ge::FORMAT_ND, {{1, 1},});
    op.UpdateInputDesc("target", target_desc);

    op.SetAttr("reduction", "sum");

    // inference shape and shape range
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret,ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT16);

    std::vector<int64_t> expected_output_shape = {};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape);
}
