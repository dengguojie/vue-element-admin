#include <gtest/gtest.h>
#include <vector>
#include "rnn.h"

class DynamicGruTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "dynamic_gru test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "dynamic_gru test TearDown" << std::endl;
  }
};

TEST_F(DynamicGruTest, dynamic_gru_test_case_1) {
  int t = 3;
  int batch = 16;
  int inputSize = 32;
  int outputSize = 48;
  ge::op::DynamicGRU dynamic_gru_op;

  ge::TensorDesc xDesc;
  ge::Shape xShape({t, batch, inputSize});
  xDesc.SetDataType(ge::DT_FLOAT16);
  xDesc.SetShape(xShape);
  xDesc.SetOriginShape(xShape);
  dynamic_gru_op.UpdateInputDesc("x", xDesc);

  ge::TensorDesc wDesc;
  ge::Shape wShape({inputSize + outputSize, 2 * outputSize});
  wDesc.SetDataType(ge::DT_FLOAT16);
  wDesc.SetShape(wShape);
  wDesc.SetOriginShape(wShape);
  dynamic_gru_op.UpdateInputDesc("w", wDesc);

  ge::TensorDesc cwDesc;
  ge::Shape cwShape({inputSize + outputSize, outputSize});
  cwDesc.SetDataType(ge::DT_FLOAT16);
  cwDesc.SetShape(cwShape);
  cwDesc.SetOriginShape(cwShape);
  dynamic_gru_op.UpdateInputDesc("cw", cwDesc);

  ge::TensorDesc bDesc;
  ge::Shape bShape({
      outputSize,
  });
  bDesc.SetDataType(ge::DT_FLOAT);
  bDesc.SetShape(bShape);
  dynamic_gru_op.UpdateInputDesc("b", bDesc);

  auto ret = dynamic_gru_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = dynamic_gru_op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {t, batch, outputSize};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(dynamic_gru_op.GetInputDesc("w").GetFormat(), ge::FORMAT_HWCN);
  EXPECT_EQ(dynamic_gru_op.GetInputDesc("cw").GetFormat(), ge::FORMAT_HWCN);
}

TEST_F(DynamicGruTest, dynamic_gru_test_case_2) {
  int t = 3;
  int batch = 16;
  int inputSize = 32;
  int outputSize = 48;
  ge::op::DynamicGRU dynamic_gru_op;

  ge::TensorDesc xDesc;
  ge::Shape xShape({1, t, batch, inputSize});
  xDesc.SetDataType(ge::DT_FLOAT16);
  xDesc.SetShape(xShape);
  dynamic_gru_op.UpdateInputDesc("x", xDesc);

  ge::TensorDesc wDesc;
  ge::Shape wShape({inputSize + outputSize, 2 * outputSize});
  wDesc.SetDataType(ge::DT_FLOAT16);
  wDesc.SetShape(wShape);
  dynamic_gru_op.UpdateInputDesc("w", wDesc);

  ge::TensorDesc cwDesc;
  ge::Shape cwShape({inputSize + outputSize, outputSize});
  cwDesc.SetDataType(ge::DT_FLOAT16);
  cwDesc.SetShape(cwShape);
  dynamic_gru_op.UpdateInputDesc("cw", cwDesc);

  ge::TensorDesc bDesc;
  ge::Shape bShape({
      outputSize,
  });
  bDesc.SetDataType(ge::DT_FLOAT);
  bDesc.SetShape(bShape);
  bDesc.SetOriginShape(bShape);
  dynamic_gru_op.UpdateInputDesc("b", bDesc);

  auto ret = dynamic_gru_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(DynamicGruTest, dynamic_gru_test_case_3) {
  int t = 3;
  int batch = 16;
  int inputSize = 32;
  int outputSize = 48;
  ge::op::CommonGRU dynamic_gru_op;

  ge::TensorDesc xDesc;
  ge::Shape xShape({t, batch, inputSize});
  xDesc.SetDataType(ge::DT_FLOAT);
  xDesc.SetShape(xShape);
  xDesc.SetOriginShape(xShape);
  dynamic_gru_op.UpdateInputDesc("x", xDesc);

  ge::TensorDesc wDesc;
  ge::Shape wShape({inputSize + outputSize, 2 * outputSize});
  wDesc.SetDataType(ge::DT_FLOAT);
  wDesc.SetShape(wShape);
  wDesc.SetOriginShape(wShape);
  dynamic_gru_op.UpdateInputDesc("w", wDesc);

  ge::TensorDesc cwDesc;
  ge::Shape cwShape({inputSize + outputSize, outputSize});
  cwDesc.SetDataType(ge::DT_FLOAT);
  cwDesc.SetShape(cwShape);
  cwDesc.SetOriginShape(cwShape);
  dynamic_gru_op.UpdateInputDesc("r", cwDesc);

  ge::TensorDesc bDesc;
  ge::Shape bShape({
      outputSize,
  });
  bDesc.SetDataType(ge::DT_FLOAT);
  bDesc.SetShape(bShape);
  bDesc.SetOriginShape(bShape);
  dynamic_gru_op.UpdateInputDesc("y", bDesc);
  dynamic_gru_op.UpdateInputDesc("y_h", bDesc);
  auto ret = dynamic_gru_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}