#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_pooling_ops.h"

// ----------------AvgPool3DD-------------------
class AvgPool3DDProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AvgPool3DD Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AvgPool3DD Proto Test TearDown" << std::endl;
  }
};

TEST_F(AvgPool3DDProtoTest, apply_avg_pool3d_d_verify_test) {
  ge::op::AvgPool3DD op;
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
