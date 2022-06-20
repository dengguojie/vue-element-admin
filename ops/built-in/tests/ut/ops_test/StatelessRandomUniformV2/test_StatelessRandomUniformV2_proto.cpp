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

class stateless_random_uniform_v2_infer_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "stateless_random_uniform_v2_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "stateless_random_uniform_v2_infer_test TearDown" << std::endl;
  }
};

TEST_F(stateless_random_uniform_v2_infer_test, stateless_random_uniform_v2_infer_test_4) {
  // new op
  ge::op::StatelessRandomUniformV2 op;

  ge::TensorDesc tensor_desc_key(ge::Shape({1}), ge::FORMAT_ND, ge::DT_UINT64);
  tensor_desc_key.SetOriginShape(ge::Shape({1}));
  op.UpdateInputDesc("key", tensor_desc_key);

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

  op.SetAttr("dtype", DT_FLOAT);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateless_random_uniform_v2_infer_test, stateless_random_uniform_v2_infer_test_6) {
  // new op
  ge::op::StatelessRandomUniformV2 op;

  ge::TensorDesc tensor_desc_key(ge::Shape({2}), ge::FORMAT_ND, ge::DT_UINT64);
  tensor_desc_key.SetOriginShape(ge::Shape({2}));
  op.UpdateInputDesc("key", tensor_desc_key);

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

  op.SetAttr("dtype", DT_FLOAT);

  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateless_random_uniform_v2_infer_test, stateless_random_uniform_v2_infer_test_7) {
  // new op
  ge::op::StatelessRandomUniformV2 op;

  ge::TensorDesc tensor_desc_key(ge::Shape({1, 1}), ge::FORMAT_ND, ge::DT_UINT64);
  tensor_desc_key.SetOriginShape(ge::Shape({1, 1}));
  op.UpdateInputDesc("key", tensor_desc_key);

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

  op.SetAttr("dtype", DT_FLOAT);

  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateless_random_uniform_v2_infer_test, stateless_random_uniform_v2_infer_test_8) {
  // new op
  ge::op::StatelessRandomUniformV2 op;

  ge::TensorDesc tensor_desc_key(ge::Shape({1}), ge::FORMAT_ND, ge::DT_UINT64);
  tensor_desc_key.SetOriginShape(ge::Shape({1}));
  op.UpdateInputDesc("key", tensor_desc_key);

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

  op.SetAttr("dtype", DT_FLOAT);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(stateless_random_uniform_v2_infer_test, stateless_random_uniform_v2_infer_test_9) {
  ge::op::StatelessRandomUniformV2 op;
  op.UpdateInputDesc("shape", create_desc({1}, ge::DT_INT32));
  op.UpdateInputDesc("key", create_desc({1}, ge::DT_UINT64));
  op.UpdateInputDesc("counter", create_desc({2}, ge::DT_UINT64));
  op.UpdateInputDesc("alg", create_desc({}, ge::DT_INT32));

  op.SetAttr("dtype", DT_FLOAT);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
