#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"


class Dropoutdomaskv3dTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Dropoutdomaskv3dTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Dropoutdomaskv3dTest TearDown" << std::endl;
  }
};

TEST_F(Dropoutdomaskv3dTest, Dropoutdomaskv3d_test_infershape_diff_test_1) {
  ge::op::DropOutDoMaskV3D op;
  ge::TensorDesc tensorDesc;
  ge::Shape shape({128,128});
  tensorDesc.SetDataType(ge::DT_FLOAT16);
  tensorDesc.SetShape(shape);
  op.UpdateInputDesc("x", tensorDesc);

  ge::TensorDesc tensorDesc1;
  ge::Shape shape1({128,128});
  tensorDesc1.SetDataType(ge::DT_UINT8);
  tensorDesc1.SetShape(shape1);
  op.UpdateInputDesc("mask", tensorDesc1);

  op.SetAttr("keep_prob", 0.1f);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}