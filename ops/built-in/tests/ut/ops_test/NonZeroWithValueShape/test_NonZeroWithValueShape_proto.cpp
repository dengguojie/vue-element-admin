#include <gtest/gtest.h>
#include <iostream>
#include "array_ops.h"
#include "op_proto_test_util.h"

class NonZeroWithValueShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "NonZeroWithValueShape test SetUp" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "NonZeroWithValueShape Teardown" << std::endl;
    }
};

TEST_F(NonZeroWithValueShapeTest, NonZeroWithValueShape_infer_shape) {
    ge::op::NonZeroWithValueShape op;
    op.UpdateInputDesc("value", create_desc({4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("index", create_desc({8}, ge::DT_INT32));
    op.UpdateInputDesc("count", create_desc({1}, ge::DT_INT32));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    std::vector<int64_t> y_dims = { -1 };
    auto y0 = op.GetOutputDesc("out_value");
    EXPECT_EQ(y0.GetDataType(), ge::DT_FLOAT16);
    EXPECT_EQ(y0.GetShape().GetDims(), y_dims);
    std::vector<std::pair<int64_t,int64_t>> output_shape_range0;
    EXPECT_EQ(y0.GetShapeRange(output_shape_range0), ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t,int64_t>> expected_shape_range0 = {{1, 4}};
    EXPECT_EQ(output_shape_range0, expected_shape_range0);

    auto y1 = op.GetOutputDesc("out_index");
    EXPECT_EQ(y1.GetDataType(), ge::DT_INT32);
    EXPECT_EQ(y1.GetShape().GetDims(), y_dims);
    std::vector<std::pair<int64_t,int64_t>> output_shape_range1;
    EXPECT_EQ(y1.GetShapeRange(output_shape_range1), ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t,int64_t>> expected_shape_range1 = {{1, 8}};
    EXPECT_EQ(output_shape_range1, expected_shape_range1);
}