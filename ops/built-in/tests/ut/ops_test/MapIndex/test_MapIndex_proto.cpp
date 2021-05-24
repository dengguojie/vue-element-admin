#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "control_flow_ops.h"

class MapIndexProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MapIndex Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MapIndex Proto Test TearDown" << std::endl;
  }
};

TEST_F(MapIndexProtoTest, map_index_infershape_test_1) {
  ge::op::MapIndex op;

  op.UpdateInputDesc("x", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("data_seq", create_desc({80}, ge::DT_INT32));

  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(MapIndexProtoTest, map_index_infershape_test_2) {
  ge::op::MapIndex op;

  op.UpdateInputDesc("x", create_desc({160}, ge::DT_INT32));
  op.UpdateInputDesc("data_seq", create_desc({1600}, ge::DT_INT32));

  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

