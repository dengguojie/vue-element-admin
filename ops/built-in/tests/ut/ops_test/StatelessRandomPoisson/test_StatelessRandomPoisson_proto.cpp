#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "stateless_random_ops.h"
#include "array_ops.h"
#include "util/util.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class StatelessRandomPoissonTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "StatelessRandomPoissonTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StatelessRandomPoissonTest TearDown" << std::endl;
  }
};

TEST_F(StatelessRandomPoissonTest, stateless_random_poisson_seed_dim_2_fail) {
  //new op
  ge::op::StatelessRandomPoisson op;

  ge::TensorDesc tensor_desc_shape(ge::Shape({4}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_shape.SetOriginShape(ge::Shape({4}));
  op.UpdateInputDesc("shape", tensor_desc_shape);

  ge::TensorDesc tensor_desc_lam(ge::Shape({5}), ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc_lam.SetOriginShape(ge::Shape({5}));
  op.UpdateInputDesc("lam", tensor_desc_lam);

  ge::TensorDesc tensor_desc_seed(ge::Shape({2, 2}), ge::FORMAT_ND, ge::DT_INT64);
  op.UpdateInputDesc("seed", tensor_desc_seed);
  op.SetAttr("dtype",DT_FLOAT);

  auto ret = op.InferShapeAndType();
  auto status = op.VerifyAllAttr(true);
  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}


TEST_F(StatelessRandomPoissonTest, stateless_random_poisson_seed_dim0_not_2_fail) {
  //new op
  ge::op::StatelessRandomPoisson op;

  ge::TensorDesc tensor_desc_shape(ge::Shape({4}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_shape.SetOriginShape(ge::Shape({4}));
  op.UpdateInputDesc("shape", tensor_desc_shape);

  ge::TensorDesc tensor_desc_lam(ge::Shape({5}), ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc_lam.SetOriginShape(ge::Shape({5}));
  op.UpdateInputDesc("lam", tensor_desc_lam);

  ge::TensorDesc tensor_desc_seed(ge::Shape({3}), ge::FORMAT_ND, ge::DT_INT64);
  op.UpdateInputDesc("seed", tensor_desc_seed);
  op.SetAttr("dtype",DT_FLOAT);

  auto ret = op.InferShapeAndType();
  auto status = op.VerifyAllAttr(true);
  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(StatelessRandomPoissonTest, stateless_random_poisson_infer_lam_rank1_fail) {
  //new op
  ge::op::StatelessRandomPoisson op;

  ge::TensorDesc tensor_desc_shape(ge::Shape({4}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_shape.SetOriginShape(ge::Shape({4}));
  op.UpdateInputDesc("shape", tensor_desc_shape);

  ge::TensorDesc tensor_desc_lam(ge::Shape({4, 1}), ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc_lam.SetOriginShape(ge::Shape({4, 1}));
  op.UpdateInputDesc("lam", tensor_desc_lam);

  ge::TensorDesc tensor_desc_seed(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT64);
  op.UpdateInputDesc("seed", tensor_desc_seed);
  op.SetAttr("dtype",DT_FLOAT);

  auto ret = op.InferShapeAndType();
  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StatelessRandomPoissonTest, stateless_random_poisson_shape_fail) {
  //new op
  ge::op::StatelessRandomPoisson op;

  ge::TensorDesc tensor_desc_lam(ge::Shape({5}), ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc_lam.SetOriginShape(ge::Shape({5}));
  op.UpdateInputDesc("lam", tensor_desc_lam);

  ge::TensorDesc tensor_desc_seed(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT64);
  op.UpdateInputDesc("seed", tensor_desc_seed);
  op.SetAttr("dtype",DT_FLOAT);

  auto ret = op.InferShapeAndType();
  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StatelessRandomPoissonTest, stateless_random_poisson_infer_1_succ) {
  //new op
  ge::op::StatelessRandomPoisson op;

  ge::TensorDesc tensor_desc_shape(ge::Shape({4}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_shape.SetOriginShape(ge::Shape({4}));
  op.UpdateInputDesc("shape", tensor_desc_shape);

  ge::TensorDesc tensor_desc_lam(ge::Shape({5}), ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc_lam.SetOriginShape(ge::Shape({5}));
  op.UpdateInputDesc("lam", tensor_desc_lam);

  ge::TensorDesc tensor_desc_seed(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT64);
  op.UpdateInputDesc("seed", tensor_desc_seed);
  op.SetAttr("dtype",DT_FLOAT);

  auto ret = op.InferShapeAndType();
  std::vector<int64_t> expected_out_shape = {-1, -1, -1, -1};
  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_out_shape);
}
