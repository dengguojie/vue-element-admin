#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "reduce_ops.h"

class BNTrainingUpdateV2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BNTrainingUpdateV2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BNTrainingUpdateV2 TearDown" << std::endl;
  }
};

TEST_F(BNTrainingUpdateV2, bn_training_update_v2_test_1) {
  ge::op::BNTrainingUpdateV2 op;

  std::vector<std::pair<int64_t,int64_t>> shape_x_range = {{2,2}, {1,1000}, {1,1000}, {1,1000}, {16,16}};
  std::vector<std::pair<int64_t,int64_t>> shape_scale_range = {{1,1}, {1,1000}, {1,1000}, {1,1000}, {16,16}};

  auto tensor_desc_x = create_desc_shape_range({-1, -1, -1, -1, 16},
                                                ge::DT_FLOAT, ge::FORMAT_NC1HWC0,
                                                {-1, -1, -1, -1, 16},
                                                ge::FORMAT_NC1HWC0, shape_x_range);
  auto tensor_desc_sum = create_desc_shape_range({-1, -1, -1, -1, 16},
                                                ge::DT_FLOAT, ge::FORMAT_NC1HWC0,
                                                {-1, -1, -1, -1, 16},
                                                ge::FORMAT_NC1HWC0, shape_scale_range);
  auto tensor_desc_square_sum = create_desc_shape_range({-1, -1, -1, -1, 16},
                                                ge::DT_FLOAT, ge::FORMAT_NC1HWC0,
                                                {-1, -1, -1, -1, 16},
                                                ge::FORMAT_NC1HWC0, shape_scale_range);
  auto tensor_desc_scale = create_desc_shape_range({-1, -1, -1, -1, 16},
                                                ge::DT_FLOAT, ge::FORMAT_NC1HWC0,
                                                {-1, -1, -1, -1, 16},
                                                ge::FORMAT_NC1HWC0, shape_scale_range);
  auto tensor_desc_offset = create_desc_shape_range({-1, -1, -1, -1, 16},
                                                ge::DT_FLOAT, ge::FORMAT_NC1HWC0,
                                                {-1, -1, -1, -1, 16},
                                                ge::FORMAT_NC1HWC0, shape_scale_range);


  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("sum", tensor_desc_sum);
  op.UpdateInputDesc("square_sum", tensor_desc_square_sum);
  op.UpdateInputDesc("scale", tensor_desc_scale);
  op.UpdateInputDesc("offset", tensor_desc_offset);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
