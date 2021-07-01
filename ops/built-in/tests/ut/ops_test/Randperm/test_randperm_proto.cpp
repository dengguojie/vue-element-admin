#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "random_ops.h"

class randperm_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "randperm SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "randperm TearDown" << std::endl;
    }
};

TEST_F(randperm_test, randperm_infer_shape_test) {
  //new op
  ge::op::Randperm op;
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

