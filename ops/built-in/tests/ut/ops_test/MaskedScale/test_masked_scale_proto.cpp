#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class MaskedScaleTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MaskedScale SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MaskedScale TearDown" << std::endl;
    }
};

TEST_F(MaskedScaleTest, masked_scale_test_case_1) {
    ge::op::MaskedScale masked_scale_op;

    ge::TensorDesc tensorDesc1;
    ge::Shape shape1({2, 3, 4});
    tensorDesc1.SetDataType(ge::DT_FLOAT16);
    tensorDesc1.SetShape(shape1);
    masked_scale_op.UpdateInputDesc("x", tensorDesc1);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({2, 3, 4});
    tensorDesc2.SetDataType(ge::DT_FLOAT16);
    tensorDesc2.SetShape(shape2);
    masked_scale_op.UpdateInputDesc("mask", tensorDesc2);
	
    masked_scale_op.SetAttr("value", 0.5f);

    auto ret = masked_scale_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = masked_scale_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MaskedScaleTest, masked_scale_test_case_2) {
    ge::op::MaskedScale masked_scale_op;

    ge::TensorDesc tensorDesc1;
    ge::Shape shape1({2, 3, 4});
    tensorDesc1.SetDataType(ge::DT_FLOAT);
    tensorDesc1.SetShape(shape1);
    masked_scale_op.UpdateInputDesc("x", tensorDesc1);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({2, 3, 4});
    tensorDesc2.SetDataType(ge::DT_FLOAT16);
    tensorDesc2.SetShape(shape2);
    masked_scale_op.UpdateInputDesc("mask", tensorDesc2);
	
    masked_scale_op.SetAttr("value", 0.5f);

    auto ret = masked_scale_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = masked_scale_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MaskedScaleTest, masked_scale_test_case_3) {
    ge::op::MaskedScale masked_scale_op;

    ge::TensorDesc tensorDesc1;
    ge::Shape shape1({2, 3, 4});
    tensorDesc1.SetDataType(ge::DT_FLOAT);
    tensorDesc1.SetShape(shape1);
    masked_scale_op.UpdateInputDesc("x", tensorDesc1);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({2, 3, 4});
    tensorDesc2.SetDataType(ge::DT_FLOAT);
    tensorDesc2.SetShape(shape2);
    masked_scale_op.UpdateInputDesc("mask", tensorDesc2);
	
    masked_scale_op.SetAttr("value", 0.5f);

    auto ret = masked_scale_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = masked_scale_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MaskedScaleTest, masked_scale_test_case_4) {
    ge::op::MaskedScale masked_scale_op;

    ge::TensorDesc tensorDesc1;
    ge::Shape shape1({2, 3, 4});
    tensorDesc1.SetDataType(ge::DT_FLOAT);
    tensorDesc1.SetShape(shape1);
    masked_scale_op.UpdateInputDesc("x", tensorDesc1);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({2, 3, 4});
    tensorDesc2.SetDataType(ge::DT_INT8);
    tensorDesc2.SetShape(shape2);
    masked_scale_op.UpdateInputDesc("mask", tensorDesc2);
	
    masked_scale_op.SetAttr("value", 0.5f);

    auto ret = masked_scale_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = masked_scale_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MaskedScaleTest, masked_scale_test_case_5) {
    ge::op::MaskedScale masked_scale_op;

    ge::TensorDesc tensorDesc1;
    ge::Shape shape1({2, 3, 4});
    tensorDesc1.SetDataType(ge::DT_FLOAT);
    tensorDesc1.SetShape(shape1);
    masked_scale_op.UpdateInputDesc("x", tensorDesc1);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({3, 4});
    tensorDesc2.SetDataType(ge::DT_INT8);
    tensorDesc2.SetShape(shape2);
    masked_scale_op.UpdateInputDesc("mask", tensorDesc2);
	
    masked_scale_op.SetAttr("value", 0.5f);

    auto ret = masked_scale_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
