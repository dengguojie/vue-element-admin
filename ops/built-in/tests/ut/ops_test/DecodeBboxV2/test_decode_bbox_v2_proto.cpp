#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"

class DecodeBboxV2Test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "decode_bbox_v2 test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "decode_bbox_v2 test TearDown" << std::endl;
    }
};

TEST_F(DecodeBboxV2Test, decode_bbox_v2_test_case_1) {
    ge::op::DecodeBboxV2 decode_bbox_v2_op;
    decode_bbox_v2_op.UpdateInputDesc("boxes", create_desc_with_ori({4, 2}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 2}, ge::FORMAT_ND));
    decode_bbox_v2_op.UpdateInputDesc("anchors", create_desc_with_ori({4, 2}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 2}, ge::FORMAT_ND));
    decode_bbox_v2_op.SetAttr("scales", std::vector<float>{1.0, 1.0, 1.0, 1.0});
    decode_bbox_v2_op.SetAttr("decode_clip", (float)0.0);
    decode_bbox_v2_op.SetAttr("reversed_box", true);
    auto ret = decode_bbox_v2_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = decode_bbox_v2_op.GetOutputDesc("y");
    std::vector<int64_t> expected_output_shape = {4, 2};
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
