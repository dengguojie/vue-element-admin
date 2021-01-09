#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"

class tensorArrayConcat : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "tensorArrayConcat Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "tensorArrayConcat Proto Test TearDown" << std::endl;
  }
};

TEST_F(tensorArrayConcat, tensorArrayConcat_infershape_diff_test){
  ge::op::TensorArrayConcat op;
  op.SetAttr("dtype", ge::DT_INT64);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}