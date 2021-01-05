#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_pooling_ops.h"

class Mask2argmaxTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "Mask2argmax SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "Mask2argmax TearDown" << std::endl;
    }
};

// TEST_F(Mask2argmaxTest, mask2argmax_test_infershape_diff_test_1) {
//     ge::op::Mask2Argmax op;
//     op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
//     op.UpdateInputDesc("mask", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
//     op.UpdateOutputDesc("argmax", create_desc_with_ori({4, 32, 32, 16}, ge::DT_FLOAT, ge::FORMAT_NHWC,{4, 32, 32, 16},ge::FORMAT_NHWC));
//     op.SetAttr("ksize", {1,3,3,1});
//     op.SetAttr("strides", {1,2,2,1});
//     op.SetAttr("padding", "SAME");
//     op.SetAttr("originshape", {4,64,64,16});
//     auto ret = op.InferShapeAndType();
//     EXPECT_EQ(ret, ge::GRAPH_FAILED);
// }
