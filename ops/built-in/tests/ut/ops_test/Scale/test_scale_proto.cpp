#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"

class ScaleTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ScaleTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ScaleTest TearDown" << std::endl;
  }
};

TEST_F(ScaleTest, scale_test_infershape_test_1) {
  ge::op::Scale op;
  // set x input shape
  ge::TensorDesc xTensorDesc;
  ge::Shape xShape({1,64,7,7});
  xTensorDesc.SetDataType(ge::DT_FLOAT16);
  xTensorDesc.SetShape(xShape);

  // set scale input shape
  ge::TensorDesc scaleTensorDesc;
  ge::Shape scaleShape({1});
  scaleTensorDesc.SetDataType(ge::DT_FLOAT16);
  scaleTensorDesc.SetShape(scaleShape);

  op.UpdateInputDesc("x", xTensorDesc);
  op.UpdateInputDesc("scale", scaleTensorDesc);

  op.SetAttr("scale_from_blob", false);
  op.SetAttr("axis", 0);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
