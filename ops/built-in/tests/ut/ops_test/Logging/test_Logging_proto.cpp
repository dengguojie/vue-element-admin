#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "logging_ops.h"

class logging : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "logging SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "logging TearDown" << std::endl;
  }
};

TEST_F(logging, print_v2_infershape){
  ge::op::PrintV2 op;
   op.UpdateInputDesc("x", create_desc({}, ge::DT_STRING));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(logging, print_v2_infershape_error){
  ge::op::PrintV2 op;
   op.UpdateInputDesc("x", create_desc({4, 3, 4}, ge::DT_STRING));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
