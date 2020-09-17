#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "selection_ops.h"

// ----------------CumLseD-------------------

class CumulativeLogsumexpProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "CumulativeLogsumexp Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "CumulativeLogsumexp Proto Test TearDown" << std::endl;
  }
};


TEST_F(CumulativeLogsumexpProtoTest, cum_lse_add_infershape_diff_test) {
  ge::op::CumulativeLogsumexp op;
  op.UpdateInputDesc("x", create_desc({2, 3, 4}, ge::DT_FLOAT16));
  ge::op::Constant axis;  
  axis.SetAttr("value", 1);
  op.SetAttr("exclusive", false);
  op.SetAttr("reverse", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

// ----------------CumLseD-------------------

class cumulative_logsumexp_d : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "cum_lse_d SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "cum_lse_d TearDown" << std::endl;
  }
};

TEST_F(cumulative_logsumexp_d, cum_lse_d_infershape_diff_test) {
  ge::op::CumulativeLogsumexpD op;
  op.UpdateInputDesc("x", create_desc({2, 3, 4}, ge::DT_FLOAT16));
  int64_t axis=1;
  op.SetAttr("axis", axis);
  op.SetAttr("exclusive", false);
  op.SetAttr("reverse", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);

  std::vector<int64_t> expected_output_shape = {2, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
