#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "selection_ops.h"

class GatherDTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "gather_d SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "gather_d TearDown" << std::endl;
  }
};

TEST_F(GatherDTest, gather_d_test_1) {
    ge::op::GatherD op;

    ge::TensorDesc tensor_desc0;
    ge::Shape shape0({2, 2});
    tensor_desc0.SetDataType(ge::DT_FLOAT16);
    tensor_desc0.SetShape(shape0);
    tensor_desc0.SetOriginShape(shape0);
    op.UpdateInputDesc("x", tensor_desc0);

    ge::TensorDesc tensor_desc1;
    ge::Shape shape1({1});
    tensor_desc1.SetDataType(ge::DT_INT32);
    tensor_desc1.SetShape(shape1);
    tensor_desc1.SetOriginShape(shape1);
    op.UpdateInputDesc("dim", tensor_desc1);

    ge::TensorDesc tensor_desc2;
    ge::Shape shape2({2, 2});
    tensor_desc2.SetDataType(ge::DT_INT64);
    tensor_desc2.SetShape(shape2);
    tensor_desc2.SetOriginShape(shape2);
    op.UpdateInputDesc("index", tensor_desc2);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {2, 2};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}