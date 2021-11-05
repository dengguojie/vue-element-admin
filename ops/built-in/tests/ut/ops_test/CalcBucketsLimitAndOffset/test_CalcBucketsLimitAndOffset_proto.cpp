#include <gtest/gtest.h>

#include <iostream>

#include "vector_search.h"
#include "op_proto_test_util.h"

class CalcBucketsLimitAndOffset : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "CalcBucketsLimitAndOffset SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "CalcBucketsLimitAndOffset TearDown" << std::endl;
  }
};

TEST_F(CalcBucketsLimitAndOffset, CalcBucketsLimitAndOffset_infershape_success) {
  ge::op::CalcBucketsLimitAndOffset op;
  op.UpdateInputDesc("bucket_list", create_desc({100}, ge::DT_INT32));
  op.UpdateInputDesc("ivf_counts", create_desc({200}, ge::DT_INT32));
  op.UpdateInputDesc("ivf_offset", create_desc({200}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}