#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "stateless_random_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class stateless_infer_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "stateless_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "stateless_infer_test TearDown" << std::endl;
  }
};

TEST_F(stateless_infer_test, stateless_multinomial_infer_test_1) {
  //new op
  ge::op::StatelessMultinomial op;
  // set input info
  ge::TensorDesc tensor_desc_logits(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_FLOAT);
  op.UpdateInputDesc("logits", tensor_desc_logits);
  ge::TensorDesc tensor_desc_num_samples(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("num_samples", tensor_desc_num_samples);
  ge::TensorDesc tensor_desc_seed(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("seed", tensor_desc_seed);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateless_infer_test, stateless_multinomial_infer_test_2) {
  //new op
  ge::op::StatelessMultinomial op;
  // set input info
  ge::TensorDesc tensor_desc_logits(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_FLOAT);
  op.UpdateInputDesc("logits", tensor_desc_logits);
  ge::TensorDesc tensor_desc_seed(ge::Shape({2, 1}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("seed", tensor_desc_seed);

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {3};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_num_samples(const0);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  ge::TensorDesc tensor_desc_seed_2(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("seed", tensor_desc_seed_2);
  ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateless_infer_test, stateless_multinomial_infer_test_3) {
  //new op
  ge::op::StatelessMultinomial op;
  // set input info
  ge::TensorDesc tensor_desc_logits(ge::Shape({1, 5, 3}), ge::FORMAT_ND, ge::DT_FLOAT);
  op.UpdateInputDesc("logits", tensor_desc_logits);
  ge::TensorDesc tensor_desc_seed(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("seed", tensor_desc_seed);

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {3};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_num_samples(const0);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateless_infer_test, stateless_multinomial_infer_test_4) {
  //new op
  ge::op::StatelessMultinomial op;
  // set input info
  ge::TensorDesc tensor_desc_logits(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_FLOAT);
  op.UpdateInputDesc("logits", tensor_desc_logits);
  ge::TensorDesc tensor_desc_num_samples(ge::Shape({3}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("num_samples", tensor_desc_num_samples);
  ge::TensorDesc tensor_desc_seed(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("seed", tensor_desc_seed);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);


  ge::TensorDesc tensor_desc_num_samples_2(ge::Shape(), ge::FORMAT_ND, ge::DT_INT64);
  op.UpdateInputDesc("num_samples", tensor_desc_num_samples_2);
  ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateless_infer_test, stateless_multinomial_infer_test_5) {
  //new op
  ge::op::StatelessMultinomial op;
  // set input info
  ge::TensorDesc tensor_desc_logits(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_FLOAT);
  op.UpdateInputDesc("logits", tensor_desc_logits);
  ge::TensorDesc tensor_desc_num_samples(ge::Shape(), ge::FORMAT_ND, ge::DT_INT64);
  op.UpdateInputDesc("num_samples", tensor_desc_num_samples);
  ge::TensorDesc tensor_desc_seed(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("seed", tensor_desc_seed);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateless_infer_test, stateless_random_uniform_int_infer_test_1) {
  //new op
  ge::op::StatelessRandomUniformInt op;
  // set input info
  //ge::TensorDesc tensor_desc_shape(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  //op.UpdateInputDesc("shape", tensor_desc_shape);
  ge::TensorDesc tensor_desc_seed(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT64);
  tensor_desc_seed.SetOriginShape(ge::Shape({2}));
  op.UpdateInputDesc("seed", tensor_desc_seed);
  ge::TensorDesc tensor_desc_minval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_minval.SetOriginShape(ge::Shape());
  op.UpdateInputDesc("minval", tensor_desc_minval);
  ge::TensorDesc tensor_desc_maxval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_maxval.SetOriginShape(ge::Shape());
  op.UpdateInputDesc("maxval", tensor_desc_maxval);

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetOriginShape(ge::Shape({1}));
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {3};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_shape(const0);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(stateless_infer_test, stateless_random_uniform_int_infer_test_2) {
  //new op
  ge::op::StatelessRandomUniformInt op;
  // set input info
  ge::TensorDesc tensor_desc_shape(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("shape", tensor_desc_shape);
  ge::TensorDesc tensor_desc_seed(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT64);
  op.UpdateInputDesc("seed", tensor_desc_seed);
  ge::TensorDesc tensor_desc_minval(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("minval", tensor_desc_minval);
  ge::TensorDesc tensor_desc_maxval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("maxval", tensor_desc_maxval);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  ge::TensorDesc tensor_desc_minval_2(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("minval", tensor_desc_minval_2);
  ge::TensorDesc tensor_desc_maxval_2(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("maxval", tensor_desc_maxval_2);
  ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateless_infer_test, stateless_random_uniform_int_infer_test_3) {
  //new op
  ge::op::StatelessRandomUniformInt op;
  // set input info
  ge::TensorDesc tensor_desc_shape(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("shape", tensor_desc_shape);
  ge::TensorDesc tensor_desc_seed(ge::Shape({2, 1}), ge::FORMAT_ND, ge::DT_INT64);
  op.UpdateInputDesc("seed", tensor_desc_seed);
  ge::TensorDesc tensor_desc_minval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("minval", tensor_desc_minval);
  ge::TensorDesc tensor_desc_maxval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("maxval", tensor_desc_maxval);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  ge::TensorDesc tensor_desc_seed_2(ge::Shape({3}), ge::FORMAT_ND, ge::DT_INT64);
  op.UpdateInputDesc("seed", tensor_desc_seed_2);
  ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateless_infer_test, stateless_random_uniform_int_infer_test_4) {
  //new op
  ge::op::StatelessRandomUniformInt op;
  // set input info
  ge::TensorDesc tensor_desc_shape(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("shape", tensor_desc_shape);
  ge::TensorDesc tensor_desc_seed(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT64);
  op.UpdateInputDesc("seed", tensor_desc_seed);
  ge::TensorDesc tensor_desc_minval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("minval", tensor_desc_minval);
  ge::TensorDesc tensor_desc_maxval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("maxval", tensor_desc_maxval);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}