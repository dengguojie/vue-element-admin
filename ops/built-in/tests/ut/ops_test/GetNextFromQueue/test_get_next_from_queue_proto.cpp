#include <gtest/gtest.h>

#include <iostream>

#include "data_flow_ops.h"`
#include "op_proto_test_util.h"

class GetNextFromQueueTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GetNextFromQueue SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GetNextFromQueue TearDown" << std::endl;
  }
};

TEST_F(GetNextFromQueueTest, test000) {
  ge::op::GetNextFromQueue op;
  op.set_attr_output_types({ge::DT_INT8});
  op.set_attr_output_shapes({{224, 224, 3}});
  op.create_dynamic_output_y(1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(GetNextFromQueueTest, test001) {
  ge::op::GetNextFromQueue op;
  op.set_attr_output_types({ge::DT_INT8});
  op.set_attr_output_shapes({{224, 224, 3}, {1}});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(GetNextFromQueueTest, test002) {
  ge::op::GetNextFromQueue op;
  op.set_attr_output_types({ge::DT_INT8});
  op.set_attr_output_shapes({{224, 224, 3}});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
