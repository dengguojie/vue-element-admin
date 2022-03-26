#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"

class BalanceRoisTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "balance_rois test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "balance_rois test TearDown" << std::endl;
  }
};

TEST_F(BalanceRoisTest, balance_rois_test_case_1) {
  ge::op::BalanceRois balance_rois_op;
  balance_rois_op.UpdateInputDesc("rois", create_desc({1000, 5}, ge::DT_FLOAT));

  auto ret = balance_rois_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto rerois_desc = balance_rois_op.GetOutputDesc("balance_rois");
  std::vector<int64_t> expected_rerois_shape = {1000, 5};
  EXPECT_EQ(rerois_desc.GetShape().GetDims(), expected_rerois_shape);

  auto index_desc = balance_rois_op.GetOutputDesc("index");
  std::vector<int64_t> expected_index_shape = {1000};
  EXPECT_EQ(index_desc.GetShape().GetDims(), expected_index_shape);
}
