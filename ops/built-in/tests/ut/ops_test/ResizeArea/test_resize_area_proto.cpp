#include <gtest/gtest.h>
#include <vector>
#include "image_ops.h"
#include "op_proto_test_util.h"

class ResizeAreaTest : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "resize_area test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "resize_area test TearDown" << std::endl;
    }
};

TEST_F(ResizeAreaTest, resize_area_test_image_size_err1) {

    ge::op::ResizeArea resize_area_op;
    resize_area_op.UpdateInputDesc("images", create_desc({1}, ge::DT_FLOAT));
    resize_area_op.UpdateInputDesc("size", create_desc({1}, ge::DT_INT32));

    auto ret = resize_area_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ResizeAreaTest, resize_area_test_image_size_err2) {

    ge::op::ResizeArea resize_area_op;
    resize_area_op.UpdateInputDesc("images", create_desc({16, 16, 16, 16}, ge::DT_FLOAT));
    resize_area_op.UpdateInputDesc("size", create_desc({}, ge::DT_INT32));

    auto ret = resize_area_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ResizeAreaTest, resize_area_test_image_size_err3) {

    ge::op::ResizeArea resize_area_op;
    resize_area_op.UpdateInputDesc("images", create_desc({16, 16, 16, 16}, ge::DT_FLOAT));
    resize_area_op.UpdateInputDesc("size", create_desc({1}, ge::DT_INT32));

    auto ret = resize_area_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}