#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"

class editDistance : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "editDistance Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "editDistance Proto Test TearDown" << std::endl;
  }
};

TEST_F(editDistance, editDistance_infershape_diff_test){
  ge::op::EditDistance op;
  auto ret = op.InferShapeAndType();
}