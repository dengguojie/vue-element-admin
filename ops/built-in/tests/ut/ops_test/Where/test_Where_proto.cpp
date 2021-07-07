#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "inference_context.h"

class where : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "where Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "where Proto Test TearDown" << std::endl;
  }
};

TEST_F(where, where_infershape_success){
  ge::op::Where op;
  op.UpdateInputDesc("x", create_desc({2, 3}, ge::DT_INT32));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}