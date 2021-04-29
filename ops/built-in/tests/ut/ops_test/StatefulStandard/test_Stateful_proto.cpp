#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "stateful_random_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class stateful_infer_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "arg_min_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "arg_min_infer_test TearDown" << std::endl;
  }
};

TEST_F(stateful_infer_test, stateful_standard_normalv2_infer_test_1) {
  //new op
  ge::op::StatefulStandardNormalV2 op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("x", tensor_desc_x);
  ge::TensorDesc tensor_desc_algorithm(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("algorithm", tensor_desc_algorithm);
  ge::TensorDesc tensor_desc_shape(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("shape", tensor_desc_shape);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateful_infer_test, stateful_standard_normalv2_infer_test_2) {
  //new op
  ge::op::StatefulStandardNormalV2 op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("x", tensor_desc_x);
  ge::TensorDesc tensor_desc_algorithm(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("algorithm", tensor_desc_algorithm);
  ge::TensorDesc tensor_desc_shape(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("shape", tensor_desc_shape);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateful_infer_test, stateful_standard_normalv2_infer_test_3) {
  //new op
  ge::op::StatefulStandardNormalV2 op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("x", tensor_desc_x);
  ge::TensorDesc tensor_desc_algorithm(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("algorithm", tensor_desc_algorithm);

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(5 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[5] = {2, 3, 4, 5, 6};
  constTensor.SetData((uint8_t*)constData, 5 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_shape(const0);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(stateful_infer_test, stateful_truncated_normal_infer_test_1) {
  //new op
  ge::op::StatefulTruncatedNormal op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("x", tensor_desc_x);
  ge::TensorDesc tensor_desc_algorithm(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("algorithm", tensor_desc_algorithm);
  ge::TensorDesc tensor_desc_shape(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("shape", tensor_desc_shape);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateful_infer_test, stateful_truncated_normal_infer_test_2) {
  //new op
  ge::op::StatefulTruncatedNormal op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("x", tensor_desc_x);
  ge::TensorDesc tensor_desc_algorithm(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("algorithm", tensor_desc_algorithm);
  ge::TensorDesc tensor_desc_shape(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("shape", tensor_desc_shape);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateful_infer_test, stateful_truncated_normal_infer_test_3) {
  //new op
  ge::op::StatefulTruncatedNormal op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("x", tensor_desc_x);
  ge::TensorDesc tensor_desc_algorithm(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("algorithm", tensor_desc_algorithm);

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(5 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[5] = {2, 3, 4, 5, 6};
  constTensor.SetData((uint8_t*)constData, 5 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_shape(const0);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(stateful_infer_test, stateful_uniform_infer_test_1) {
  //new op
  ge::op::StatefulUniform op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("x", tensor_desc_x);
  ge::TensorDesc tensor_desc_algorithm(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("algorithm", tensor_desc_algorithm);
  ge::TensorDesc tensor_desc_shape(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("shape", tensor_desc_shape);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateful_infer_test, stateful_uniform_infer_test_2) {
  //new op
  ge::op::StatefulUniform op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("x", tensor_desc_x);
  ge::TensorDesc tensor_desc_algorithm(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("algorithm", tensor_desc_algorithm);
  ge::TensorDesc tensor_desc_shape(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("shape", tensor_desc_shape);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateful_infer_test, stateful_uniform_infer_test_3) {
  //new op
  ge::op::StatefulUniform op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("x", tensor_desc_x);
  ge::TensorDesc tensor_desc_algorithm(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("algorithm", tensor_desc_algorithm);

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(5 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[5] = {2, 3, 4, 5, 6};
  constTensor.SetData((uint8_t*)constData, 5 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_shape(const0);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(stateful_infer_test, stateful_uniform_full_int_infer_test_1) {
  //new op
  ge::op::StatefulUniformFullInt op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("x", tensor_desc_x);
  ge::TensorDesc tensor_desc_algorithm(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("algorithm", tensor_desc_algorithm);
  ge::TensorDesc tensor_desc_shape(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("shape", tensor_desc_shape);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateful_infer_test, stateful_uniform_full_int_infer_test_2) {
  //new op
  ge::op::StatefulUniformFullInt op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("x", tensor_desc_x);
  ge::TensorDesc tensor_desc_algorithm(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("algorithm", tensor_desc_algorithm);
  ge::TensorDesc tensor_desc_shape(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("shape", tensor_desc_shape);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateful_infer_test, stateful_uniform_full_int_infer_test_3) {
  //new op
  ge::op::StatefulUniformFullInt op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("x", tensor_desc_x);
  ge::TensorDesc tensor_desc_algorithm(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("algorithm", tensor_desc_algorithm);

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(5 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[5] = {2, 3, 4, 5, 6};
  constTensor.SetData((uint8_t*)constData, 5 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_shape(const0);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(stateful_infer_test, stateful_uniform_int_infer_test_1) {
  //new op
  ge::op::StatefulUniformInt op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("x", tensor_desc_x);
  ge::TensorDesc tensor_desc_algorithm(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("algorithm", tensor_desc_algorithm);
  ge::TensorDesc tensor_desc_shape(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("shape", tensor_desc_shape);
  ge::TensorDesc tensor_desc_minval(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("minval", tensor_desc_minval);
  ge::TensorDesc tensor_desc_maxval(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("maxval", tensor_desc_maxval);

  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateful_infer_test, stateful_uniform_int_infer_test_2) {
  //new op
  ge::op::StatefulUniformInt op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("x", tensor_desc_x);
  ge::TensorDesc tensor_desc_algorithm(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("algorithm", tensor_desc_algorithm);
  ge::TensorDesc tensor_desc_shape(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("shape", tensor_desc_shape);
  ge::TensorDesc tensor_desc_minval(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("minval", tensor_desc_minval);
  ge::TensorDesc tensor_desc_maxval(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("maxval", tensor_desc_maxval);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateful_infer_test, stateful_uniform_int_infer_test_3) {
  //new op
  ge::op::StatefulUniformInt op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("x", tensor_desc_x);
  ge::TensorDesc tensor_desc_algorithm(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("algorithm", tensor_desc_algorithm);
  ge::TensorDesc tensor_desc_shape(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("shape", tensor_desc_shape);
  ge::TensorDesc tensor_desc_minval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("minval", tensor_desc_minval);
  ge::TensorDesc tensor_desc_maxval(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("maxval", tensor_desc_maxval);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateful_infer_test, stateful_uniform_int_infer_test_4) {
  //new op
  ge::op::StatefulUniformInt op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("x", tensor_desc_x);
  ge::TensorDesc tensor_desc_algorithm(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("algorithm", tensor_desc_algorithm);
  ge::TensorDesc tensor_desc_shape(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("shape", tensor_desc_shape);
  ge::TensorDesc tensor_desc_minval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("minval", tensor_desc_minval);
  ge::TensorDesc tensor_desc_maxval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("maxval", tensor_desc_maxval);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateful_infer_test, stateful_uniform_int_infer_test_5) {
  //new op
  ge::op::StatefulUniformInt op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("x", tensor_desc_x);
  ge::TensorDesc tensor_desc_algorithm(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("algorithm", tensor_desc_algorithm);
    ge::TensorDesc tensor_desc_minval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("minval", tensor_desc_minval);
  ge::TensorDesc tensor_desc_maxval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("maxval", tensor_desc_maxval);

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(5 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[5] = {2, 3, 4, 5, 6};
  constTensor.SetData((uint8_t*)constData, 5 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_shape(const0);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}