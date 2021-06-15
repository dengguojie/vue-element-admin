#include <gtest/gtest.h>
#include <vector>
#include "selection_ops.h"

class MaskedFillRangeTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "masked_fill_range test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "masked_fill_range test TearDown" << std::endl;
    }
};

TEST_F(MaskedFillRangeTest, masked_fill_range_test_case_1) {
    ge::op::MaskedFillRange masked_fill_range_op;

    ge::TensorDesc tensor_x;
    ge::Shape shape_x({2, 3, 4});
    tensor_x.SetDataType(ge::DT_FLOAT16);
    tensor_x.SetShape(shape_x);

    ge::TensorDesc tensor_start;
    ge::Shape shape_start({1, 2});
    tensor_start.SetDataType(ge::DT_INT32);
    tensor_start.SetShape(shape_start);

    ge::TensorDesc tensor_end;
    ge::Shape shape_end({1, 2});
    tensor_end.SetDataType(ge::DT_INT32);
    tensor_end.SetShape(shape_end);

    ge::TensorDesc tensor_value;
    ge::Shape shape_value({1});
    tensor_value.SetDataType(ge::DT_FLOAT16);
    tensor_value.SetShape(shape_value);

    masked_fill_range_op.SetAttr("axis", 0);

    // update op input here
    masked_fill_range_op.UpdateInputDesc("x", tensor_x);
    masked_fill_range_op.UpdateInputDesc("start", tensor_start);
    masked_fill_range_op.UpdateInputDesc("end", tensor_end);
    masked_fill_range_op.UpdateInputDesc("value", tensor_value);
    masked_fill_range_op.UpdateOutputDesc("value", tensor_value);

    // call InferShapeAndType function here
    auto ret = masked_fill_range_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // compare dtype and shape of op output
    auto output_desc = masked_fill_range_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MaskedFillRangeTest, masked_fill_range_test_case_2) {
    ge::op::MaskedFillRange masked_fill_range_op;

    ge::TensorDesc tensor_x;
    ge::Shape shape_x({2, 3, 4});
    tensor_x.SetDataType(ge::DT_FLOAT);
    tensor_x.SetShape(shape_x);

    ge::TensorDesc tensor_start;
    ge::Shape shape_start({1, 2});
    tensor_start.SetDataType(ge::DT_INT32);
    tensor_start.SetShape(shape_start);

    ge::TensorDesc tensor_end;
    ge::Shape shape_end({1, 2});
    tensor_end.SetDataType(ge::DT_INT32);
    tensor_end.SetShape(shape_end);

    ge::TensorDesc tensor_value;
    ge::Shape shape_value({1});
    tensor_value.SetDataType(ge::DT_FLOAT16);
    tensor_value.SetShape(shape_value);

    masked_fill_range_op.SetAttr("axis", 0);

    // update op input here
    masked_fill_range_op.UpdateInputDesc("x", tensor_x);
    masked_fill_range_op.UpdateInputDesc("start", tensor_start);
    masked_fill_range_op.UpdateInputDesc("end", tensor_end);
    masked_fill_range_op.UpdateInputDesc("value", tensor_value);

    // call InferShapeAndType function here
    auto ret = masked_fill_range_op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaskedFillRangeTest, masked_fill_range_test_case_3) {
    ge::op::MaskedFillRange masked_fill_range_op;

    ge::TensorDesc tensor_x;
    ge::Shape shape_x({2, 3, 4});
    tensor_x.SetDataType(ge::DT_FLOAT16);
    tensor_x.SetShape(shape_x);

    ge::TensorDesc tensor_start;
    ge::Shape shape_start({1, 2});
    tensor_start.SetDataType(ge::DT_FLOAT16);
    tensor_start.SetShape(shape_start);

    ge::TensorDesc tensor_end;
    ge::Shape shape_end({1, 2});
    tensor_end.SetDataType(ge::DT_INT32);
    tensor_end.SetShape(shape_end);

    ge::TensorDesc tensor_value;
    ge::Shape shape_value({1});
    tensor_value.SetDataType(ge::DT_FLOAT16);
    tensor_value.SetShape(shape_value);

    masked_fill_range_op.SetAttr("axis", 0);

    // update op input here
    masked_fill_range_op.UpdateInputDesc("x", tensor_x);
    masked_fill_range_op.UpdateInputDesc("start", tensor_start);
    masked_fill_range_op.UpdateInputDesc("end", tensor_end);
    masked_fill_range_op.UpdateInputDesc("value", tensor_value);

    // call InferShapeAndType function here
    auto ret = masked_fill_range_op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaskedFillRangeTest, masked_fill_range_test_case_4) {
    ge::op::MaskedFillRange masked_fill_range_op;

    ge::TensorDesc tensor_x;
    ge::Shape shape_x({2, 3, 4});
    tensor_x.SetDataType(ge::DT_FLOAT16);
    tensor_x.SetShape(shape_x);

    ge::TensorDesc tensor_start;
    ge::Shape shape_start({1, 2});
    tensor_start.SetDataType(ge::DT_INT32);
    tensor_start.SetShape(shape_start);

    ge::TensorDesc tensor_end;
    ge::Shape shape_end({1, 2});
    tensor_end.SetDataType(ge::DT_INT32);
    tensor_end.SetShape(shape_end);

    ge::TensorDesc tensor_value;
    ge::Shape shape_value({1});
    tensor_value.SetDataType(ge::DT_FLOAT16);
    tensor_value.SetShape(shape_value);

    masked_fill_range_op.SetAttr("axis", 5);

    // update op input here
    masked_fill_range_op.UpdateInputDesc("x", tensor_x);
    masked_fill_range_op.UpdateInputDesc("start", tensor_start);
    masked_fill_range_op.UpdateInputDesc("end", tensor_end);
    masked_fill_range_op.UpdateInputDesc("value", tensor_value);

    // call InferShapeAndType function here
    auto ret = masked_fill_range_op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaskedFillRangeTest, masked_fill_range_test_case_5) {
    ge::op::MaskedFillRange masked_fill_range_op;

    ge::TensorDesc tensor_x;
    ge::Shape shape_x({2, 3, 4});
    tensor_x.SetDataType(ge::DT_FLOAT16);
    tensor_x.SetShape(shape_x);

    ge::TensorDesc tensor_start;
    ge::Shape shape_start({1, 2});
    tensor_start.SetDataType(ge::DT_INT32);
    tensor_start.SetShape(shape_start);

    ge::TensorDesc tensor_end;
    ge::Shape shape_end({1, 2});
    tensor_end.SetDataType(ge::DT_INT32);
    tensor_end.SetShape(shape_end);

    ge::TensorDesc tensor_value;
    ge::Shape shape_value({1});
    tensor_value.SetDataType(ge::DT_FLOAT16);
    tensor_value.SetShape(shape_value);

    masked_fill_range_op.SetAttr("axis", 1);

    // update op input here
    masked_fill_range_op.UpdateInputDesc("x", tensor_x);
    masked_fill_range_op.UpdateInputDesc("start", tensor_start);
    masked_fill_range_op.UpdateInputDesc("end", tensor_end);
    masked_fill_range_op.UpdateInputDesc("value", tensor_value);

    // call InferShapeAndType function here
    auto ret = masked_fill_range_op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
