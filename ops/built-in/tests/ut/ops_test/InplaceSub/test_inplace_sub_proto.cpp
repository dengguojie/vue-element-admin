#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "selection_ops.h"
#include "common/utils/ut_op_util.h"

// ----------------InplaceSub-------------------
class inplace_sub : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "inplace_sub SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "inplace_sub TearDown" << std::endl;
  }
};

using namespace ut_util;

TEST_F(inplace_sub, inplace_sub_infershape_diff_test) {
  ge::op::InplaceSub op;
  op.UpdateInputDesc("x", create_desc({2, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("v", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  vector<uint32_t> value = {0, 1, 2, 3};
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(op, indices, vector<int64_t>({4}), ge::DT_INT32, FORMAT_ND, value);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
