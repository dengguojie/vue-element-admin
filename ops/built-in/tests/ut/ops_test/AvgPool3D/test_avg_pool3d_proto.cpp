#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_pooling_ops.h"

// ----------------AvgPool3D-------------------
class AvgPool3DProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AvgPool3D Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AvgPool3D Proto Test TearDown" << std::endl;
  }
};
// TODO fix me run failed
//TEST_F(AvgPool3DProtoTest,apply_avg_pool3d_infershape_verify_test) {
//  ge::op::AvgPool3D op;
//  op.UpdateInputDesc("x", create_desc({1, 4, 7, 7, 1024}, ge::DT_FLOAT16));
//  op.SetAttr("ksize", {1,2,7,7,1});
//  op.SetAttr("strides",{1,1,1,1,1});
//  op.SetAttr("pads",{0,0,0,0,0,0});
//  op.SetAttr("ceil_mode",false);
//  op.SetAttr("count_include_pad",true);
//  op.SetAttr("divisor_override",0);
//  op.SetAttr("data_format","NDHWC");
//
//  auto status = op.VerifyAllAttr(true);
//  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
//
//  auto ret = op.InferShapeAndType();
//  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
//
//  auto y_desc = op.GetOutputDesc("y");
//  EXPECT_EQ(y_desc.GetDataType(), ge::DT_FLOAT16);
//  std::vector<int64_t> expected_y_output_shape = {1, 3, 1, 1, 1024};
//  EXPECT_EQ(y_desc.GetShape().GetDims(), expected_y_output_shape);
//}

TEST_F(AvgPool3DProtoTest, apply_avg_pool3d_verify_test) {
  ge::op::AvgPool3D op;
  op.UpdateInputDesc("x", create_desc({1, 4, 7, 7, 1024}, ge::DT_FLOAT16));
  op.SetAttr("ksize", {1,2,7,7,1,1});
  op.SetAttr("strides",{1,1,1,1,1,1});
  op.SetAttr("pads",{0,0,0,0,0,0});
  op.SetAttr("ceil_mode",false);
  op.SetAttr("count_include_pad",true);
  op.SetAttr("divisor_override",0);
  op.SetAttr("data_format","NDHWC");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}
