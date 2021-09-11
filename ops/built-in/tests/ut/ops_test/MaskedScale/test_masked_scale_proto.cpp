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
    tensorDesc1.SetOriginShape(shape1);
    masked_scale_op.UpdateInputDesc("x", tensorDesc1);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({2, 3, 4});
    tensorDesc2.SetDataType(ge::DT_FLOAT16);
    tensorDesc2.SetShape(shape2);
    tensorDesc2.SetOriginShape(shape2);
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
    tensorDesc1.SetOriginShape(shape1);
    masked_scale_op.UpdateInputDesc("x", tensorDesc1);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({2, 3, 4});
    tensorDesc2.SetDataType(ge::DT_FLOAT16);
    tensorDesc2.SetShape(shape2);
    tensorDesc2.SetOriginShape(shape2);
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
    tensorDesc1.SetOriginShape(shape1);
    masked_scale_op.UpdateInputDesc("x", tensorDesc1);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({2, 3, 4});
    tensorDesc2.SetDataType(ge::DT_FLOAT);
    tensorDesc2.SetShape(shape2);
    tensorDesc2.SetOriginShape(shape2);
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
    tensorDesc1.SetOriginShape(shape1);
    masked_scale_op.UpdateInputDesc("x", tensorDesc1);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({2, 3, 4});
    tensorDesc2.SetDataType(ge::DT_INT8);
    tensorDesc2.SetShape(shape2);
    tensorDesc2.SetOriginShape(shape2);
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
    tensorDesc1.SetOriginShape(shape1);
    masked_scale_op.UpdateInputDesc("x", tensorDesc1);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({3, 4});
    tensorDesc2.SetDataType(ge::DT_INT8);
    tensorDesc2.SetShape(shape2);
    tensorDesc2.SetOriginShape(shape2);
    masked_scale_op.UpdateInputDesc("mask", tensorDesc2);
	
    masked_scale_op.SetAttr("value", 0.5f);

    auto ret = masked_scale_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaskedScaleTest, masked_scale_infershape_dynamic_test) {
    ge::op::MaskedScale masked_scale_op;
    std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 100}};
    auto tensor_desc = create_desc_shape_range({-1},
                                               ge::DT_FLOAT16, ge::FORMAT_ND,
                                               {64},
                                               ge::FORMAT_ND, shape_range);
    masked_scale_op.UpdateInputDesc("x", tensor_desc);
    masked_scale_op.UpdateInputDesc("mask", tensor_desc);

    float value = 1.0;
    masked_scale_op.SetAttr("value", value);

    auto status = masked_scale_op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = masked_scale_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = masked_scale_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {-1};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
    std::vector<std::pair<int64_t,int64_t>> output_shape_range;
    EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
        {2, 100},
    };
    EXPECT_EQ(output_shape_range, expected_shape_range);

}