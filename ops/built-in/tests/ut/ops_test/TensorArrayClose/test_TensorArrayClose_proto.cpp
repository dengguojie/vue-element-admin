#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"
#include "inference_context.h"

class tensorArrayClose : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "tensorArrayClose Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "tensorArrayClose Proto Test TearDown" << std::endl;
  }
};

TEST_F(tensorArrayClose, tensorArrayClose_infershape_input0_rank_failed){
  ge::op::TensorArrayClose op;
  op.UpdateInputDesc("handle", create_desc({2, 3}, ge::DT_RESOURCE));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}