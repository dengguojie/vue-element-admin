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

class stateless_random_uniform_int_v2_infer_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "stateless_random_uniform_int_v2_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "stateless_random_uniform_int_v2_infer_test TearDown" << std::endl;
  }
};

TEST_F(stateless_random_uniform_int_v2_infer_test, stateless_random_uniform_int_v2_infer_test_1) {
  // new op
  ge::op::StatelessRandomUniformIntV2 op;

  ge::TensorDesc tensor_desc_key(ge::Shape({1}), ge::FORMAT_ND, ge::DT_UINT64);
  tensor_desc_key.SetOriginShape(ge::Shape({1}));
  op.UpdateInputDesc("key", tensor_desc_key);
  
  ge::TensorDesc tensor_desc_minval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_minval.SetOriginShape(ge::Shape());
  op.UpdateInputDesc("minval", tensor_desc_minval);
  ge::TensorDesc tensor_desc_maxval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_maxval.SetOriginShape(ge::Shape());
  op.UpdateInputDesc("maxval", tensor_desc_maxval);

  ge::TensorDesc tensor_desc_counter(ge::Shape({1}), ge::FORMAT_ND, ge::DT_UINT64);
  tensor_desc_counter.SetOriginShape(ge::Shape({1}));
  op.UpdateInputDesc("counter", tensor_desc_counter);

  ge::Tensor constTensor_alg;
  ge::TensorDesc constDesc_alg(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc_alg.SetSize(1 * sizeof(int32_t));
  constTensor_alg.SetTensorDesc(constDesc_alg);
  int32_t constData_alg[1] = {2};
  constTensor_alg.SetData((uint8_t*)constData_alg, 1 * sizeof(int32_t));
  auto const0_alg = ge::op::Constant().set_attr_value(constTensor_alg);
  op.set_input_alg(const0_alg);

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

  auto y_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_y_shape = {3};
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_INT32);
  EXPECT_EQ(y_desc.GetShape().GetDims(), expected_y_shape);
}

TEST_F(stateless_random_uniform_int_v2_infer_test, stateless_random_uniform_int_v2_infer_test_2) {
  // new op
  ge::op::StatelessRandomUniformIntV2 op;

  ge::TensorDesc tensor_desc_key(ge::Shape({1}), ge::FORMAT_ND, ge::DT_UINT64);
  tensor_desc_key.SetOriginShape(ge::Shape({1}));
  op.UpdateInputDesc("key", tensor_desc_key);
  
  ge::TensorDesc tensor_desc_minval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_minval.SetOriginShape(ge::Shape());
  op.UpdateInputDesc("minval", tensor_desc_minval);
  ge::TensorDesc tensor_desc_maxval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_maxval.SetOriginShape(ge::Shape());
  op.UpdateInputDesc("maxval", tensor_desc_maxval);

  ge::TensorDesc tensor_desc_counter(ge::Shape({1}), ge::FORMAT_ND, ge::DT_UINT64);
  tensor_desc_counter.SetOriginShape(ge::Shape({1}));
  op.UpdateInputDesc("counter", tensor_desc_counter);

  ge::Tensor constTensor_alg;
  ge::TensorDesc constDesc_alg(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc_alg.SetSize(1 * sizeof(int32_t));
  constTensor_alg.SetTensorDesc(constDesc_alg);
  int32_t constData_alg[1] = {1};
  constTensor_alg.SetData((uint8_t*)constData_alg, 1 * sizeof(int32_t));
  auto const0_alg = ge::op::Constant().set_attr_value(constTensor_alg);
  op.set_input_alg(const0_alg);

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
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateless_random_uniform_int_v2_infer_test, stateless_random_uniform_int_v2_infer_test_3) {
  // new op
  ge::op::StatelessRandomUniformIntV2 op;

  ge::TensorDesc tensor_desc_key(ge::Shape({1}), ge::FORMAT_ND, ge::DT_UINT64);
  tensor_desc_key.SetOriginShape(ge::Shape({1}));
  op.UpdateInputDesc("key", tensor_desc_key);
  
  ge::TensorDesc tensor_desc_minval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_minval.SetOriginShape(ge::Shape());
  op.UpdateInputDesc("minval", tensor_desc_minval);
  ge::TensorDesc tensor_desc_maxval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_maxval.SetOriginShape(ge::Shape());
  op.UpdateInputDesc("maxval", tensor_desc_maxval);

  ge::TensorDesc tensor_desc_counter(ge::Shape({2}), ge::FORMAT_ND, ge::DT_UINT64);
  tensor_desc_counter.SetOriginShape(ge::Shape({2}));
  op.UpdateInputDesc("counter", tensor_desc_counter);

  ge::Tensor constTensor_alg;
  ge::TensorDesc constDesc_alg(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc_alg.SetSize(1 * sizeof(int32_t));
  constTensor_alg.SetTensorDesc(constDesc_alg);
  int32_t constData_alg[1] = {2};
  constTensor_alg.SetData((uint8_t*)constData_alg, 1 * sizeof(int32_t));
  auto const0_alg = ge::op::Constant().set_attr_value(constTensor_alg);
  op.set_input_alg(const0_alg);

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
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateless_random_uniform_int_v2_infer_test, stateless_random_uniform_int_v2_infer_test_4) {
  // new op
  ge::op::StatelessRandomUniformIntV2 op;

  ge::TensorDesc tensor_desc_key(ge::Shape({1}), ge::FORMAT_ND, ge::DT_UINT64);
  tensor_desc_key.SetOriginShape(ge::Shape({1}));
  op.UpdateInputDesc("key", tensor_desc_key);
  
  ge::TensorDesc tensor_desc_minval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_minval.SetOriginShape(ge::Shape());
  op.UpdateInputDesc("minval", tensor_desc_minval);
  ge::TensorDesc tensor_desc_maxval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_maxval.SetOriginShape(ge::Shape());
  op.UpdateInputDesc("maxval", tensor_desc_maxval);

  ge::TensorDesc tensor_desc_counter(ge::Shape({1, 2}), ge::FORMAT_ND, ge::DT_UINT64);
  tensor_desc_counter.SetOriginShape(ge::Shape({1, 2}));
  op.UpdateInputDesc("counter", tensor_desc_counter);

  ge::Tensor constTensor_alg;
  ge::TensorDesc constDesc_alg(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc_alg.SetSize(1 * sizeof(int32_t));
  constTensor_alg.SetTensorDesc(constDesc_alg);
  int32_t constData_alg[1] = {1};
  constTensor_alg.SetData((uint8_t*)constData_alg, 1 * sizeof(int32_t));
  auto const0_alg = ge::op::Constant().set_attr_value(constTensor_alg);
  op.set_input_alg(const0_alg);

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
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateless_random_uniform_int_v2_infer_test, stateless_random_uniform_int_v2_infer_test_5) {
  // new op
  ge::op::StatelessRandomUniformIntV2 op;
  // set input info
  ge::TensorDesc tensor_desc_key(ge::Shape({1}), ge::FORMAT_ND, ge::DT_UINT64);
  tensor_desc_key.SetOriginShape(ge::Shape({1}));
  op.UpdateInputDesc("key", tensor_desc_key);
  
  ge::TensorDesc tensor_desc_minval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_minval.SetOriginShape(ge::Shape());
  op.UpdateInputDesc("minval", tensor_desc_minval);
  ge::TensorDesc tensor_desc_maxval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_maxval.SetOriginShape(ge::Shape());
  op.UpdateInputDesc("maxval", tensor_desc_maxval);
  ge::TensorDesc tensor_desc_counter(ge::Shape({2}), ge::FORMAT_ND, ge::DT_UINT64);
  tensor_desc_counter.SetOriginShape(ge::Shape({2}));
  op.UpdateInputDesc("counter", tensor_desc_counter);

  ge::Tensor constTensor_alg;
  ge::TensorDesc constDesc_alg(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc_alg.SetSize(1 * sizeof(int32_t));
  constTensor_alg.SetTensorDesc(constDesc_alg);
  int32_t constData_alg[1] = {1};
  constTensor_alg.SetData((uint8_t*)constData_alg, 1 * sizeof(int32_t));
  auto const0_alg = ge::op::Constant().set_attr_value(constTensor_alg);
  op.set_input_alg(const0_alg);

  ge::TensorDesc tensor_desc_shape(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("shape", tensor_desc_shape);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateless_random_uniform_int_v2_infer_test, stateless_random_uniform_int_v2_infer_test_6) {
  // new op
  ge::op::StatelessRandomUniformIntV2 op;

  ge::TensorDesc tensor_desc_key(ge::Shape({2}), ge::FORMAT_ND, ge::DT_UINT64);
  tensor_desc_key.SetOriginShape(ge::Shape({2}));
  op.UpdateInputDesc("key", tensor_desc_key);
  
  ge::TensorDesc tensor_desc_minval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_minval.SetOriginShape(ge::Shape());
  op.UpdateInputDesc("minval", tensor_desc_minval);
  ge::TensorDesc tensor_desc_maxval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_maxval.SetOriginShape(ge::Shape());
  op.UpdateInputDesc("maxval", tensor_desc_maxval);

  ge::TensorDesc tensor_desc_counter(ge::Shape({2}), ge::FORMAT_ND, ge::DT_UINT64);
  tensor_desc_counter.SetOriginShape(ge::Shape({2}));
  op.UpdateInputDesc("counter", tensor_desc_counter);

  ge::Tensor constTensor_alg;
  ge::TensorDesc constDesc_alg(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc_alg.SetSize(1 * sizeof(int32_t));
  constTensor_alg.SetTensorDesc(constDesc_alg);
  int32_t constData_alg[1] = {1};
  constTensor_alg.SetData((uint8_t*)constData_alg, 1 * sizeof(int32_t));
  auto const0_alg = ge::op::Constant().set_attr_value(constTensor_alg);
  op.set_input_alg(const0_alg);

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetOriginShape(ge::Shape({1}));
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {3};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_shape(const0);

  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateless_random_uniform_int_v2_infer_test, stateless_random_uniform_int_v2_infer_test_7) {
  // new op
  ge::op::StatelessRandomUniformIntV2 op;

  ge::TensorDesc tensor_desc_key(ge::Shape({1, 1}), ge::FORMAT_ND, ge::DT_UINT64);
  tensor_desc_key.SetOriginShape(ge::Shape({1, 1}));
  op.UpdateInputDesc("key", tensor_desc_key);
  
  ge::TensorDesc tensor_desc_minval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_minval.SetOriginShape(ge::Shape());
  op.UpdateInputDesc("minval", tensor_desc_minval);
  ge::TensorDesc tensor_desc_maxval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_maxval.SetOriginShape(ge::Shape());
  op.UpdateInputDesc("maxval", tensor_desc_maxval);

  ge::TensorDesc tensor_desc_counter(ge::Shape({2}), ge::FORMAT_ND, ge::DT_UINT64);
  tensor_desc_counter.SetOriginShape(ge::Shape({2}));
  op.UpdateInputDesc("counter", tensor_desc_counter);

  ge::Tensor constTensor_alg;
  ge::TensorDesc constDesc_alg(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc_alg.SetSize(1 * sizeof(int32_t));
  constTensor_alg.SetTensorDesc(constDesc_alg);
  int32_t constData_alg[1] = {1};
  constTensor_alg.SetData((uint8_t*)constData_alg, 1 * sizeof(int32_t));
  auto const0_alg = ge::op::Constant().set_attr_value(constTensor_alg);
  op.set_input_alg(const0_alg);

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetOriginShape(ge::Shape({1}));
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {3};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_shape(const0);

  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateless_random_uniform_int_v2_infer_test, stateless_random_uniform_int_v2_infer_test_8) {
  // new op
  ge::op::StatelessRandomUniformIntV2 op;

  ge::TensorDesc tensor_desc_key(ge::Shape({1}), ge::FORMAT_ND, ge::DT_UINT64);
  tensor_desc_key.SetOriginShape(ge::Shape({1}));
  op.UpdateInputDesc("key", tensor_desc_key);
  
  ge::TensorDesc tensor_desc_minval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_minval.SetOriginShape(ge::Shape());
  op.UpdateInputDesc("minval", tensor_desc_minval);
  ge::TensorDesc tensor_desc_maxval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_maxval.SetOriginShape(ge::Shape());
  op.UpdateInputDesc("maxval", tensor_desc_maxval);

  ge::TensorDesc tensor_desc_counter(ge::Shape({2}), ge::FORMAT_ND, ge::DT_UINT64);
  tensor_desc_counter.SetOriginShape(ge::Shape({2}));
  op.UpdateInputDesc("counter", tensor_desc_counter);

  ge::TensorDesc tensor_desc_alg(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_alg.SetOriginShape(ge::Shape({1}));
  op.UpdateInputDesc("alg", tensor_desc_alg);

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
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateless_random_uniform_int_v2_infer_test, stateless_random_uniform_int_v2_infer_test_9) {
  // new op
  ge::op::StatelessRandomUniformIntV2 op;

  ge::TensorDesc tensor_desc_key(ge::Shape({1}), ge::FORMAT_ND, ge::DT_UINT64);
  tensor_desc_key.SetOriginShape(ge::Shape({1}));
  op.UpdateInputDesc("key", tensor_desc_key);
  
  ge::TensorDesc tensor_desc_minval(ge::Shape({1,1}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_minval.SetOriginShape(ge::Shape({1,1}));
  op.UpdateInputDesc("minval", tensor_desc_minval);
  ge::TensorDesc tensor_desc_maxval(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_maxval.SetOriginShape(ge::Shape());
  op.UpdateInputDesc("maxval", tensor_desc_maxval);

  ge::TensorDesc tensor_desc_counter(ge::Shape({1}), ge::FORMAT_ND, ge::DT_UINT64);
  tensor_desc_counter.SetOriginShape(ge::Shape({1}));
  op.UpdateInputDesc("counter", tensor_desc_counter);

  ge::Tensor constTensor_alg;
  ge::TensorDesc constDesc_alg(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc_alg.SetSize(1 * sizeof(int32_t));
  constTensor_alg.SetTensorDesc(constDesc_alg);
  int32_t constData_alg[1] = {2};
  constTensor_alg.SetData((uint8_t*)constData_alg, 1 * sizeof(int32_t));
  auto const0_alg = ge::op::Constant().set_attr_value(constTensor_alg);
  op.set_input_alg(const0_alg);

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetOriginShape(ge::Shape({1}));
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {3};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_shape(const0);

  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  ge::TensorDesc tensor_desc_minval_2(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_minval_2.SetOriginShape(ge::Shape());
  op.UpdateInputDesc("minval", tensor_desc_minval_2);
  ge::TensorDesc tensor_desc_maxval_2(ge::Shape({1,1}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_maxval_2.SetOriginShape(ge::Shape({1,1}));
  op.UpdateInputDesc("maxval", tensor_desc_maxval_2);
  ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}