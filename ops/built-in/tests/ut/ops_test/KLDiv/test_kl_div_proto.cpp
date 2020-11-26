#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"


// ----------------KLDivProtoTest Begin-------------------
class KLDivProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "KLDiv Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "KLDiv Proto Test TearDown" << std::endl;
  }
};


TEST_F(KLDivProtoTest, kl_div_infershape_test){
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

TEST_F(KLDivProtoTest, kl_div_verify_success_01_test){
  ge::op::KLDiv op;
  op.UpdateInputDesc("x", create_desc({16, 2, 16, 16}, ge::DT_FLOAT));
  op.UpdateInputDesc("target", create_desc({16, 2, 16, 16}, ge::DT_FLOAT));
  op.SetAttr("reduction", "batchmean");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(KLDivProtoTest, kl_div_verify_success_02_test){
  ge::op::KLDiv op;
  op.UpdateInputDesc("x", create_desc({16, 2, 16, 16}, ge::DT_FLOAT));
  op.UpdateInputDesc("target", create_desc({16, 2, 16, 16}, ge::DT_FLOAT));
  op.SetAttr("reduction", "batchmean");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

// TODO fix me run failed
//TEST_F(KLDivProtoTest, sparse_apply_adadelta_verify_failed_01_test){
//  ge::op::KLDiv op;
//  op.UpdateInputDesc("x", create_desc({16, 2, 16, 16}, ge::DT_FLOAT));
//  op.UpdateInputDesc("target", create_desc({16, 2, 16, 16}, ge::DT_FLOAT));
//  op.SetAttr("reduc", "sum");
//
//  auto status = op.VerifyAllAttr(true);
//  EXPECT_EQ(status, ge::GRAPH_FAILED);
//}
// ----------------KLDivProtoTest End-------------------




