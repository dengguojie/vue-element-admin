#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"

class UpdateTensorDescProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "UpdateTensorDesc Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "UpdateTensorDesc Proto Test TearDown" << std::endl;
  }
};

TEST_F(UpdateTensorDescProtoTest, UpdateTensorDescVerifyTest_1) {
  ge::op::UpdateTensorDesc op;
  std::vector<int64_t> shape_list = {16, 12, 5, 32};
  op.SetAttr("shape", shape_list);
  auto status = op.VerifyAllAttr(true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

