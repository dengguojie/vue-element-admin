#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "inference_context.h"

class where : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "where Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "where Proto Test TearDown" << std::endl;
  }
};

TEST_F(where, where_infershape_success){
  ge::op::Where op;
  op.UpdateInputDesc("x", create_desc({2, 3}, ge::DT_INT32));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(where, where_infershape_unknow_dim){
  ge::op::Where op;
  auto shape_x = std::vector<int64_t>({-1});
  std::vector<std::pair<int64_t, int64_t>> range_x = {{1, 6}};
  auto format_x = ge::FORMAT_NC1HWC0;
  auto tensor_desc_x = create_desc_shape_range(shape_x, ge::DT_INT64, format_x, shape_x, format_x, range_x);
  op.UpdateInputDesc("x", tensor_desc_x);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}