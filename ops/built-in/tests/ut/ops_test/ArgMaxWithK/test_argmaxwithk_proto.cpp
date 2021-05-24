#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class ArgMaxWithKTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ArgMaxWithK Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ArgMaxWithK Test TearDown" << std::endl;
  }
};

TEST_F(ArgMaxWithKTest, ArgMaxWithKTest_infershape_test_1) {
  ge::op::ArgMaxWithK op;
  // set x input shape
  ge::TensorDesc xTensorDesc;
  ge::Shape xShape({2, 2, 3, 3});
  xTensorDesc.SetDataType(ge::DT_FLOAT16);
  xTensorDesc.SetShape(xShape);

  op.UpdateInputDesc("x", xTensorDesc);
  op.SetAttr("axis", 1);
  op.SetAttr("out_max_val", false);
  op.SetAttr("topk", 0);  
  
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}