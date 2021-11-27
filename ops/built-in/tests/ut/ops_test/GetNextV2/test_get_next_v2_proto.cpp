#include <gtest/gtest.h>

#include <iostream>

#include "data_flow_ops.h"
#include "op_proto_test_util.h"

class GetNextV2Test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GetNextV2Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GetNextV2Test TearDown" << std::endl;
  }
};

TEST_F(GetNextV2Test, test000) {
  ge::op::GetNextV2 op;
  op.set_attr_output_types({ge::DT_INT8});
  op.set_attr_output_shapes({{224, 224, 3}});
  op.create_dynamic_output_y(1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(GetNextV2Test, test001) {
  ge::op::GetNextV2 op;
  op.set_attr_output_types({ge::DT_INT8});
  op.set_attr_output_shapes({{224, 224, 3}, {1}});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(GetNextV2Test, test002) {
  ge::op::GetNextV2 op;
  op.set_attr_output_types({ge::DT_INT8});
  op.set_attr_output_shapes({{224, 224, 3}});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(GetNextV2Test, test003) {
  ge::op::GetNextV2 op;
  op.set_attr_output_types({ge::DT_INT8});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(GetNextV2Test, test004) {
  ge::op::GetNextV2 op;
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}