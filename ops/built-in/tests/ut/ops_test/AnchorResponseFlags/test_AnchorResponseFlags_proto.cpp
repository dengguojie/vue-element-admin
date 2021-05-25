#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"

class AnchorResponseFlags : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AnchorResponseFlags Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AnchorResponseFlags Proto Test TearDown" << std::endl;
  }
};

TEST_F(AnchorResponseFlags, anchor_response_flags_infershape_test_1){
  ge::op::AnchorResponseFlags op;
  op.SetAttr("featmap_size", {10,10});
  op.SetAttr("num_base_anchors", 3);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("flags");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_UINT8);
  std::vector<int64_t> expected_output_shape = {300};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
