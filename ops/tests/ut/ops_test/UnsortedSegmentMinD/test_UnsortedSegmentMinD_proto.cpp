#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "selection_ops.h"

class UnsortedSegmentMinD : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "UnsortedSegmentMinD SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "UnsortedSegmentMinD TearDown" << std::endl;
    }
};

TEST_F(UnsortedSegmentMinD, UnsortedSegmentMinD_case) {
    ge::op::UnsortedSegmentMinD op;

    op.UpdateInputDesc("x", create_desc({16,}, ge::DT_FLOAT));
    op.UpdateInputDesc("segment_ids", create_desc({16,}, ge::DT_INT32));
    op.SetAttr("num_segments", 5);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {5, };
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}