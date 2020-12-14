#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_pooling_ops.h"

// ----------------AvgPool1D-------------------
class AvgPool1DProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AvgPool1D Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AvgPool1D Proto Test TearDown" << std::endl;
  }
};

// TODO fix me run failed
//TEST_F(AvgPool1DProtoTest,apply_avg_pool1d_infershape_verify_test) {
//  ge::op::AvgPool1D op;
//  op.UpdateInputDesc("x", create_desc({16,1,1,16000,16}, ge::DT_FLOAT16));
//  op.SetAttr("ksize", 4);
//  op.SetAttr("strides",2);
//  op.SetAttr("pads",{0,0});
//  op.SetAttr("ceil_mode",false);
//  op.SetAttr("count_include_pad",false);
//
//  auto status = op.VerifyAllAttr(true);
//  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
//
//  auto ret = op.InferShapeAndType();
//  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
//
//  auto y_desc = op.GetOutputDesc("y");
//  EXPECT_EQ(y_desc.GetDataType(), ge::DT_FLOAT16);
//  std::vector<int64_t> expected_y_output_shape = {16,1,1,8000,16};
//  EXPECT_EQ(y_desc.GetShape().GetDims(), expected_y_output_shape);
//}

//TEST_F(AvgPool1DProtoTest, apply_avg_pool1d_verify_test) {
//  ge::op::AvgPool1D op;
//  op.UpdateInputDesc("x", create_desc({16,1,1,16000,16}, ge::DT_FLOAT16));
//  op.SetAttr("ksize", 4);
//  op.SetAttr("strides",2);
//  op.SetAttr("pads",{0,0});
//  op.SetAttr("ceil_mode",false);
//  op.SetAttr("count_include_pad",false);
//
//  auto status = op.VerifyAllAttr(true);
//  EXPECT_EQ(status, ge::GRAPH_FAILED);
//}
