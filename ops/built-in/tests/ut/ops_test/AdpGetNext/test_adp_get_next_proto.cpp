#include <gtest/gtest.h>

#include <iostream>

#include "data_flow_ops.h"
#include "op_proto_test_util.h"

class AdpGetNextTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AdpGetNextTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AdpGetNextTest TearDown" << std::endl;
  }
};

TEST_F(AdpGetNextTest, test000) {
  ge::op::AdpGetNext op;
  op.set_attr_output_types({ge::DT_INT8});
  op.set_attr_output_shapes({{224, 224, 3}});
  op.create_dynamic_output_y(1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(AdpGetNextTest, test001) {
  ge::op::AdpGetNext op;
  op.set_attr_output_types({ge::DT_INT8});
  op.set_attr_output_shapes({{224, 224, 3}, {1}});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(AdpGetNextTest, test002) {
  ge::op::AdpGetNext op;
  op.set_attr_output_types({ge::DT_INT8});
  op.set_attr_output_shapes({{224, 224, 3}});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
