#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "random_ops.h"

class linspace : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "linspace Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "linspace Proto Test TearDown" << std::endl;
  }
};

TEST_F(linspace, linspace_infershape_numunknown_test){
  ge::op::LinSpace op;
  op.UpdateInputDesc("start", create_desc({}, ge::DT_FLOAT));
  op.UpdateInputDesc("stop", create_desc({}, ge::DT_FLOAT));
  op.UpdateInputDesc("num", create_desc({}, ge::DT_INT32));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}