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

class StatelessRandomGammaV2Test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "StatelessRandomGammaV2Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StatelessRandomGammaV2Test TearDown" << std::endl;
  }
};

TEST_F(StatelessRandomGammaV2Test, stateless_random_gamma_v2_infer_succ) {
  ge::op::StatelessRandomGammaV2 op;
  // shape
  ge::Tensor shape_constTensor;
  ge::TensorDesc shape_constDesc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  shape_constDesc.SetOriginShape(ge::Shape({2}));
  shape_constDesc.SetSize(2 * sizeof(int32_t));
  shape_constTensor.SetTensorDesc(shape_constDesc);
  int32_t constData[2] = {2, 3};
  shape_constTensor.SetData((uint8_t*)constData, 2 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(shape_constTensor);
  op.set_input_shape(const0);
  // seed
  ge::TensorDesc tensor_desc_seed(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_seed.SetOriginShape(ge::Shape({2}));
  op.UpdateInputDesc("seed", tensor_desc_seed);
  // alpha
  ge::TensorDesc tensor_desc_alpha(ge::Shape({3}), ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc_alpha.SetOriginShape(ge::Shape({3}));
  op.UpdateInputDesc("alpha", tensor_desc_alpha);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  std::vector<int64_t> expected_out_shape = {2, 3};
  auto output_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_out_shape);
}

TEST_F(StatelessRandomGammaV2Test, stateless_random_gamma_v2_infer_shape_const_fail) {
  ge::op::StatelessRandomGammaV2 op;
  // shape
  ge::Tensor shape_constTensor;
  ge::TensorDesc shape_constDesc(ge::Shape({2, 2}), ge::FORMAT_ND, ge::DT_INT32);
  shape_constDesc.SetOriginShape(ge::Shape({2, 2}));
  shape_constDesc.SetSize(4 * sizeof(int32_t));
  shape_constTensor.SetTensorDesc(shape_constDesc);
  int32_t constData[4] = {2, 3, 2, 3};
  shape_constTensor.SetData((uint8_t*)constData, 4 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(shape_constTensor);
  op.set_input_shape(const0);
  // seed
  ge::TensorDesc tensor_desc_seed(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_seed.SetOriginShape(ge::Shape({2}));
  op.UpdateInputDesc("seed", tensor_desc_seed);
  // alpha
  ge::TensorDesc tensor_desc_alpha(ge::Shape({3}), ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc_alpha.SetOriginShape(ge::Shape({3}));
  op.UpdateInputDesc("alpha", tensor_desc_alpha);
  auto ret = op.InferShapeAndType();
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StatelessRandomGammaV2Test, stateless_random_gamma_v2_infer_seed_rank1_fail) {
  ge::op::StatelessRandomGammaV2 op;
  // shape
  ge::Tensor shape_constTensor;
  ge::TensorDesc shape_constDesc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  shape_constDesc.SetOriginShape(ge::Shape({2}));
  shape_constDesc.SetSize(2 * sizeof(int32_t));
  shape_constTensor.SetTensorDesc(shape_constDesc);
  int32_t constData[2] = {2, 3};
  shape_constTensor.SetData((uint8_t*)constData, 2 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(shape_constTensor);
  op.set_input_shape(const0);
  // seed
  ge::TensorDesc tensor_desc_seed(ge::Shape({2, 2}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_seed.SetOriginShape(ge::Shape({2, 2}));
  op.UpdateInputDesc("seed", tensor_desc_seed);
  // alpha
  ge::TensorDesc tensor_desc_alpha(ge::Shape({3}), ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc_alpha.SetOriginShape(ge::Shape({3}));
  op.UpdateInputDesc("alpha", tensor_desc_alpha);
  auto ret = op.InferShapeAndType();
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(StatelessRandomGammaV2Test, stateless_random_gamma_v2_infer_seed_dim0_fail) {
  ge::op::StatelessRandomGammaV2 op;
  // shape
  ge::Tensor shape_constTensor;
  ge::TensorDesc shape_constDesc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  shape_constDesc.SetOriginShape(ge::Shape({2}));
  shape_constDesc.SetSize(2 * sizeof(int32_t));
  shape_constTensor.SetTensorDesc(shape_constDesc);
  int32_t constData[2] = {2, 3};
  shape_constTensor.SetData((uint8_t*)constData, 2 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(shape_constTensor);
  op.set_input_shape(const0);
  // seed
  ge::TensorDesc tensor_desc_seed(ge::Shape({3}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_seed.SetOriginShape(ge::Shape({3}));
  op.UpdateInputDesc("seed", tensor_desc_seed);
  // alpha
  ge::TensorDesc tensor_desc_alpha(ge::Shape({3}), ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc_alpha.SetOriginShape(ge::Shape({3}));
  op.UpdateInputDesc("alpha", tensor_desc_alpha);
  auto ret = op.InferShapeAndType();
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(StatelessRandomGammaV2Test, stateless_random_gamma_v2_infer_alpha_rank1_fail) {
  ge::op::StatelessRandomGammaV2 op;
  // shape
  ge::Tensor shape_constTensor;
  ge::TensorDesc shape_constDesc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  shape_constDesc.SetOriginShape(ge::Shape({2}));
  shape_constDesc.SetSize(2 * sizeof(int32_t));
  shape_constTensor.SetTensorDesc(shape_constDesc);
  int32_t constData[2] = {2, 3};
  shape_constTensor.SetData((uint8_t*)constData, 2 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(shape_constTensor);
  op.set_input_shape(const0);
  // seed
  ge::TensorDesc tensor_desc_seed(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_seed.SetOriginShape(ge::Shape({2}));
  op.UpdateInputDesc("seed", tensor_desc_seed);
  // alpha
  ge::TensorDesc tensor_desc_alpha(ge::Shape({3, 1}), ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc_alpha.SetOriginShape(ge::Shape({3, 1}));
  op.UpdateInputDesc("alpha", tensor_desc_alpha);
  auto ret = op.InferShapeAndType();
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StatelessRandomGammaV2Test, stateless_random_gamma_v2_infer_alpha_dim_fail) {
  ge::op::StatelessRandomGammaV2 op;
  // shape
  ge::Tensor shape_constTensor;
  ge::TensorDesc shape_constDesc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  shape_constDesc.SetOriginShape(ge::Shape({2}));
  shape_constDesc.SetSize(2 * sizeof(int32_t));
  shape_constTensor.SetTensorDesc(shape_constDesc);
  int32_t constData[2] = {2, 3};
  shape_constTensor.SetData((uint8_t*)constData, 2 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(shape_constTensor);
  op.set_input_shape(const0);
  // seed
  ge::TensorDesc tensor_desc_seed(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_seed.SetOriginShape(ge::Shape({2}));
  op.UpdateInputDesc("seed", tensor_desc_seed);
  // alpha
  ge::TensorDesc tensor_desc_alpha(ge::Shape({2}), ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc_alpha.SetOriginShape(ge::Shape({2}));
  op.UpdateInputDesc("alpha", tensor_desc_alpha);
  auto ret = op.InferShapeAndType();
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}