#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "sdca_ops.h"

class sdca_optimizer_v2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "sdca_optimizer_v2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "sdca_optimizer_v2 TearDown" << std::endl;
  }
};

TEST_F(sdca_optimizer_v2, sdca_optimizer_v2_infershape_test_1) {
  ge::op::SdcaOptimizerV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{10, 10}, {10, 10}};
  auto tensor_desc = create_desc_shape_range({10, 10},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {10, 10},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("sparse_weights", tensor_desc);
  op.UpdateInputDesc("dense_weights", tensor_desc);

  op.SetAttr("num_sparse_features", 2);
  op.SetAttr("num_dense_features", 2);

  op.create_dynamic_output_out_delta_sparse_weights(2);
  op.create_dynamic_output_out_delta_dense_weights(2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(sdca_optimizer_v2, sdca_optimizer_v2_infershape_test_2) {
  ge::op::SdcaOptimizerV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{10, 10}, {10, 10}};
  auto tensor_desc = create_desc_shape_range({10, 10},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {10, 10},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("sparse_weights", tensor_desc);
  op.UpdateInputDesc("dense_weights", tensor_desc);

  op.SetAttr("num_sparse_features", 2);
  op.SetAttr("num_dense_features", 2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(sdca_optimizer_v2, sdca_optimizer_v2_infershape_test_3) {
  ge::op::SdcaOptimizerV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{10, 10}, {10, 10}};
  auto tensor_desc = create_desc_shape_range({10, 10},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {10, 10},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("sparse_weights", tensor_desc);
  op.UpdateInputDesc("dense_weights", tensor_desc);

  op.SetAttr("num_sparse_features", 2);
  op.SetAttr("num_dense_features", 2);

  op.create_dynamic_output_out_delta_sparse_weights(2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}