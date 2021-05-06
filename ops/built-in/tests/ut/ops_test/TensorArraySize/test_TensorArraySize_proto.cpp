#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"
#include "inference_context.h"

class tensorArraySize : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "tensorArraySize Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "tensorArraySize Proto Test TearDown" << std::endl;
  }
};

TEST_F(tensorArraySize, tensorArraySize_infershape_input0_rank_fail){
  ge::op::TensorArraySize op;
  op.UpdateInputDesc("handle", create_desc({2}, ge::DT_RESOURCE));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}