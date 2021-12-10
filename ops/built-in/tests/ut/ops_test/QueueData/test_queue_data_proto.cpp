#include <gtest/gtest.h>

#include <iostream>

#include "array_ops.h"
#include "op_proto_test_util.h"

class QueueDataTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "QueueData SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "QueueData TearDown" << std::endl;
  }
};

TEST_F(QueueDataTest, test000) {
  ge::op::QueueData op;
  op.set_attr_output_types({ge::DT_INT8});
  op.set_attr_output_shapes({{224, 224, 3}});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(QueueDataTest, test001) {
  ge::op::QueueData op;
  op.set_attr_output_types({ge::DT_INT8});
  op.set_attr_output_shapes({{224, 224, 3}, {1}});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(QueueDataTest, test002) {
  ge::op::QueueData op;
  op.set_attr_output_types({ge::DT_STRING});
  op.set_attr_output_shapes({{1}});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
