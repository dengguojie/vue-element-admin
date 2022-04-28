#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "selection_ops.h"


class NonMaxSuppressionBucketizeTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "non_max_suppression_bucketize test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "non_max_suppression_bucketize test TearDown" << std::endl;
    }
};

TEST_F(NonMaxSuppressionBucketizeTest, non_max_suppression_bucketize_test_case_1) {
    ge::op::NonMaxSuppressionBucketize op;
    ge::TensorDesc tensordesc_box;
    ge::TensorDesc tensordesc_score;
    ge::TensorDesc tensordesc_class;
    ge::TensorDesc tensordesc_num;
    ge::Shape box_shape({4, 100, 4});
    ge::Shape score_class_shape({4, 100});
    ge::Shape num_shape({4});
    tensordesc_box.SetDataType(ge::DT_FLOAT16);
    tensordesc_box.SetShape(box_shape);
    tensordesc_box.SetOriginShape(box_shape);
    tensordesc_score.SetDataType(ge::DT_FLOAT16);
    tensordesc_score.SetShape(score_class_shape);
    tensordesc_score.SetOriginShape(score_class_shape);
    tensordesc_class.SetDataType(ge::DT_FLOAT16);
    tensordesc_class.SetShape(score_class_shape);
    tensordesc_class.SetOriginShape(score_class_shape);
    tensordesc_num.SetDataType(ge::DT_INT32);
    tensordesc_num.SetShape(num_shape);
    tensordesc_num.SetOriginShape(num_shape);

    op.UpdateInputDesc("input_nmsed_boxes", tensordesc_box);
    op.UpdateInputDesc("input_nmsed_score", tensordesc_score);
    op.UpdateInputDesc("input_nmsed_class", tensordesc_class);
    op.UpdateInputDesc("input_nmsed_num", tensordesc_num);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_boxes_desc = op.GetOutputDescByName("output_nmsed_boxes");
    EXPECT_EQ(output_boxes_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_boxes_shape = {4, -1, 4};
    EXPECT_EQ(output_boxes_desc.GetShape().GetDims(), expected_boxes_shape);

    auto output_score_desc = op.GetOutputDescByName("output_nmsed_score");
    EXPECT_EQ(output_score_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_score_shape = {4, -1};
    EXPECT_EQ(output_score_desc.GetShape().GetDims(), expected_score_shape);

    auto output_class_desc = op.GetOutputDescByName("output_nmsed_class");
    EXPECT_EQ(output_class_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_class_shape = {4, -1};
    EXPECT_EQ(output_class_desc.GetShape().GetDims(), expected_class_shape);
}