#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "selection_ops.h"

class segment_max : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "segment_max SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "segment_max TearDown" << std::endl;
  }
};

TEST_F(segment_max, segment_max_infershape_test_1) {
  ge::op::SegmentMax op;
  op.UpdateInputDesc("x", create_desc_shape_range({9, 10, 2, 6, 7},
                                                  ge::DT_INT32,
                                                  ge::FORMAT_ND,
                                                  {9, 10, 2, 6, 7},
                                                  ge::FORMAT_ND,
                                                  {{9,9},{10,10},{2,2},{6,6},{7,7}}
                                                  ));
  op.UpdateInputDesc("segment_ids", create_desc_shape_range({9, 10, 2},
                                                            ge::DT_INT32,
                                                            ge::FORMAT_ND,
                                                            {9, 10, 2},
                                                            ge::FORMAT_ND,
                                                            {{9,9},{10,10},{2,2}}
                                                            ));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(segment_max, segment_max_infershape_test_2) {
  ge::op::SegmentMax op;
  op.UpdateInputDesc("x", create_desc_shape_range({},
                                                  ge::DT_INT32,
                                                  ge::FORMAT_ND,
                                                  {},
                                                  ge::FORMAT_ND,
                                                  {}
                                                  ));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}