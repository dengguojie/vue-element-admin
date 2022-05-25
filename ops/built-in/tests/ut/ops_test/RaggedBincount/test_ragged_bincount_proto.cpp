#include <gtest/gtest.h>

#include <iostream>

#include "math_ops.h"
#include "op_proto_test_util.h"

class RaggedBincountTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RaggedBincount test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RaggedBincount test TearDown" << std::endl;
  }
};

TEST_F(RaggedBincountTest, test000) {
  ge::op::RaggedBincount op;

  ge::TensorDesc splits_tensor;
  ge::Shape  splits_shape({10});
  splits_tensor.SetDataType(ge::DT_INT64);
  splits_tensor.SetShape(splits_shape);
  splits_tensor.SetOriginShape(splits_shape);
  op.UpdateInputDesc("splits", splits_tensor);

  ge::TensorDesc values_tensor;
  ge::Shape  values_shape({10,40});
  values_tensor.SetDataType(ge::DT_INT64);
  values_tensor.SetShape(values_shape);
  values_tensor.SetOriginShape(values_shape);
  op.UpdateInputDesc("values", values_tensor);

  ge::TensorDesc size_tensor;

  size_tensor.SetDataType(ge::DT_INT64);
  op.UpdateInputDesc("size", size_tensor);
  ge::TensorDesc weights_tensor;
  ge::Shape  weights_shape({10});
  weights_tensor.SetDataType(ge::DT_INT64);
  weights_tensor.SetShape(weights_shape);
  weights_tensor.SetOriginShape(weights_shape);
  op.UpdateInputDesc("weights", weights_tensor);

  op.SetAttr("binary_output", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("output");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT64);
  EXPECT_EQ(output_desc.GetShape().GetDims()[0], 9);
  EXPECT_EQ(output_desc.GetShape().GetDims()[1], -1);
}

TEST_F(RaggedBincountTest, test001) {
  ge::op::RaggedBincount op;

  ge::TensorDesc splits_tensor;
  ge::Shape  splits_shape({10, 1});
  splits_tensor.SetDataType(ge::DT_INT64);
  splits_tensor.SetShape(splits_shape);
  splits_tensor.SetOriginShape(splits_shape);
  op.UpdateInputDesc("splits", splits_tensor);

  ge::TensorDesc values_tensor;
  ge::Shape  values_shape({10,40});
  values_tensor.SetDataType(ge::DT_INT64);
  values_tensor.SetShape(values_shape);
  values_tensor.SetOriginShape(values_shape);
  op.UpdateInputDesc("values", values_tensor);

  ge::TensorDesc size_tensor;

  size_tensor.SetDataType(ge::DT_INT64);
  op.UpdateInputDesc("size", size_tensor);
  ge::TensorDesc weights_tensor;
  ge::Shape  weights_shape({10});
  weights_tensor.SetDataType(ge::DT_INT64);
  weights_tensor.SetShape(weights_shape);
  weights_tensor.SetOriginShape(weights_shape);
  op.UpdateInputDesc("weights", weights_tensor);

  op.SetAttr("binary_output", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RaggedBincountTest, test002) {
  ge::op::RaggedBincount op;

  ge::TensorDesc splits_tensor;
  ge::Shape  splits_shape({10});
  splits_tensor.SetDataType(ge::DT_INT64);
  splits_tensor.SetShape(splits_shape);
  splits_tensor.SetOriginShape(splits_shape);
  op.UpdateInputDesc("splits", splits_tensor);

  ge::TensorDesc values_tensor;
  ge::Shape  values_shape({10,40,20});
  values_tensor.SetDataType(ge::DT_INT64);
  values_tensor.SetShape(values_shape);
  values_tensor.SetOriginShape(values_shape);
  op.UpdateInputDesc("values", values_tensor);

  ge::TensorDesc size_tensor;

  size_tensor.SetDataType(ge::DT_INT64);
  op.UpdateInputDesc("size", size_tensor);
  ge::TensorDesc weights_tensor;
  ge::Shape  weights_shape({10});
  weights_tensor.SetDataType(ge::DT_INT64);
  weights_tensor.SetShape(weights_shape);
  weights_tensor.SetOriginShape(weights_shape);
  op.UpdateInputDesc("weights", weights_tensor);

  op.SetAttr("binary_output", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RaggedBincountTest, test003) {
  ge::op::RaggedBincount op;

  ge::TensorDesc splits_tensor;
  ge::Shape  splits_shape({10});
  splits_tensor.SetDataType(ge::DT_INT64);
  splits_tensor.SetShape(splits_shape);
  splits_tensor.SetOriginShape(splits_shape);
  op.UpdateInputDesc("splits", splits_tensor);

  ge::TensorDesc values_tensor;
  ge::Shape  values_shape({10,40,20});
  values_tensor.SetDataType(ge::DT_INT64);
  values_tensor.SetShape(values_shape);
  values_tensor.SetOriginShape(values_shape);
  op.UpdateInputDesc("values", values_tensor);

  ge::TensorDesc size_tensor;

  size_tensor.SetDataType(ge::DT_INT64);
  op.UpdateInputDesc("size", size_tensor);
  ge::TensorDesc weights_tensor;
  ge::Shape  weights_shape({10});
  weights_tensor.SetDataType(ge::DT_INT64);
  weights_tensor.SetShape(weights_shape);
  weights_tensor.SetOriginShape(weights_shape);
  op.UpdateInputDesc("weights", weights_tensor);

  op.SetAttr("binary_output", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RaggedBincountTest, test004) {
  ge::op::RaggedBincount op;

  ge::TensorDesc splits_tensor;
  ge::Shape  splits_shape({10});
  splits_tensor.SetDataType(ge::DT_INT64);
  splits_tensor.SetShape(splits_shape);
  splits_tensor.SetOriginShape(splits_shape);
  op.UpdateInputDesc("splits", splits_tensor);

  ge::TensorDesc values_tensor;
  ge::Shape  values_shape({10,40});
  values_tensor.SetDataType(ge::DT_INT64);
  values_tensor.SetShape(values_shape);
  values_tensor.SetOriginShape(values_shape);
  op.UpdateInputDesc("values", values_tensor);

  ge::TensorDesc size_tensor;

  size_tensor.SetDataType(ge::DT_INT64);
  ge::Shape  size_shape({10,40});
  size_tensor.SetOriginShape(size_shape);
  size_tensor.SetShape(size_shape);
  op.UpdateInputDesc("size", size_tensor);
  ge::TensorDesc weights_tensor;
  ge::Shape  weights_shape({10});
  weights_tensor.SetDataType(ge::DT_INT64);
  weights_tensor.SetShape(weights_shape);
  weights_tensor.SetOriginShape(weights_shape);
  op.UpdateInputDesc("weights", weights_tensor);

  op.SetAttr("binary_output", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RaggedBincountTest, test005) {
  ge::op::RaggedBincount op;

  ge::TensorDesc splits_tensor;
  ge::Shape  splits_shape({10});
  splits_tensor.SetDataType(ge::DT_INT64);
  splits_tensor.SetShape(splits_shape);
  splits_tensor.SetOriginShape(splits_shape);
  op.UpdateInputDesc("splits", splits_tensor);

  ge::TensorDesc values_tensor;
  ge::Shape  values_shape({10,40});
  values_tensor.SetDataType(ge::DT_INT64);
  values_tensor.SetShape(values_shape);
  values_tensor.SetOriginShape(values_shape);
  op.UpdateInputDesc("values", values_tensor);

  ge::TensorDesc size_tensor;

  size_tensor.SetDataType(ge::DT_INT64);
  op.UpdateInputDesc("size", size_tensor);
  ge::TensorDesc weights_tensor;
  ge::Shape  weights_shape({10});
  weights_tensor.SetDataType(ge::DT_INT32);
  weights_tensor.SetShape(weights_shape);
  weights_tensor.SetOriginShape(weights_shape);
  op.UpdateInputDesc("weights", weights_tensor);

  op.SetAttr("binary_output", false);

  ge::TensorDesc output_tensor;
  output_tensor.SetDataType(ge::DT_INT64);
  op.UpdateOutputDesc("output", output_tensor);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

}