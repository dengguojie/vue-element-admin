#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"

class cacheallindextolocal : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "cacheallindextolocal SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "cacheallindextolocal TearDown" << std::endl;
  }
};

TEST_F(cacheallindextolocal, cacheallindextolocal_infershape_diff_test){
  ge::op::CacheAllIndexToLocal op;
  op.SetAttr("dtype", ge::DT_INT64);
  op.UpdateInputDesc("cache", create_desc({}, ge::DT_RESOURCE));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("local_idx");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_output_shape = {-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
