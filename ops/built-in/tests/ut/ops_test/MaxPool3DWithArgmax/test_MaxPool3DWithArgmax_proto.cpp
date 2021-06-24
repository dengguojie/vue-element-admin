#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "op_proto_test_util.h"
#include "nn_pooling_ops.h"

class MaxPool3DWithArgmaxTest : public testing::Test {
  protected:
    static void SetUpTestCase() {
      std::cout << "max_pool3d_with_argmax test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "max_pool3d_with_argmax test TearDown" << std::endl;
    }
};

TEST_F(MaxPool3DWithArgmaxTest, max_pool3d_with_argmax_test_case_0) {
  ge::op::MaxPool3DWithArgmax op;

  ge::TensorDesc tensorDesc1;
  ge::Shape shape1({1, 10, 1, 6, 6, 16});
  tensorDesc1.SetDataType(ge::DT_FLOAT16);
  tensorDesc1.SetShape(shape1);

  op.UpdateInputDesc("x", tensorDesc1);
  
  std::vector<int64_t> ksize {1, 1, 1, 2, 2};
  std::vector<int64_t> strides {1, 1, 1, 2, 2};
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {1, 10, 1, 3, 3, 16};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

  auto argmax_desc = op.GetOutputDesc("argmax");
  EXPECT_EQ(argmax_desc.GetDataType(), ge::DT_UINT16);
  std::vector<int64_t> expected_argmax_shape = {1, 10, 4, 1, 16};
  EXPECT_EQ(argmax_desc.GetShape().GetDims(), expected_argmax_shape);
}

