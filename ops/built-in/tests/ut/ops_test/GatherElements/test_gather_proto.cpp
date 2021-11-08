#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "selection_ops.h"

class GatherElementsTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "gather_elements SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "gather_elements TearDown" << std::endl;
  }
};

TEST_F(GatherElementsTest, gather_elements_test_1) {
    ge::op::GatherElements op;

    ge::TensorDesc tensor_desc1;
    ge::Shape shape1({2, 2});
    tensor_desc1.SetDataType(ge::DT_FLOAT16);
    tensor_desc1.SetShape(shape1);
    tensor_desc1.SetOriginShape(shape1);
    op.UpdateInputDesc("x", tensor_desc1);

    ge::TensorDesc tensor_desc2;
    ge::Shape shape2({2, 2});
    tensor_desc2.SetDataType(ge::DT_INT64);
    tensor_desc2.SetShape(shape2);
    tensor_desc2.SetOriginShape(shape2);
    op.UpdateInputDesc("index", tensor_desc2);

    int attr_value = 1;
    op.SetAttr("dim", attr_value);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {2, 2};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}