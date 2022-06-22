#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "selection_ops.h"
#include "common/utils/ut_op_common.h"
#include "array_ops.h"

// ----------------DynSeqOuter-------------------
class DynSeqOuterProtoTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "DynSeqOuterProtoTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DynSeqOuterProtoTest TearDown" << std::endl;
  }
};


TEST_F(DynSeqOuterProtoTest, DynSeqOuterProtoTest_0) {
    ge::op::DynSeqOuter op;

    ge::TensorDesc alpha_desc;
    ge::Shape xShape({41, 512});
    alpha_desc.SetDataType(ge::DT_FLOAT);
    alpha_desc.SetShape(xShape);
    alpha_desc.SetOriginShape(xShape);

    ge::TensorDesc offset_desc;
    ge::Shape yShape({8});
    offset_desc.SetDataType(ge::DT_INT32);
    offset_desc.SetShape(yShape);
    offset_desc.SetOriginShape(yShape);

    op.UpdateInputDesc("x1", alpha_desc);
    op.UpdateInputDesc("x2", alpha_desc);
    op.UpdateInputDesc("seq_len1", offset_desc);
    op.UpdateInputDesc("seq_len2", offset_desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(DynSeqOuterProtoTest, DynSeqOuterProtoTest_normal_case) {
    ge::op::DynSeqOuter op;
    std::vector<std::pair<int64_t, int64_t>> offset_range = {{1, -1}};
    std::vector<std::pair<int64_t, int64_t>> x_range = {{1, -1}, {1, -1}};
    std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{1, -1}, {1, -1}};

    ge::TensorDesc alpha_desc;
    ge::Shape xShape({-1, -1});
    alpha_desc.SetDataType(ge::DT_FLOAT);
    alpha_desc.SetShape(xShape);
    alpha_desc.SetOriginShape(xShape);
    alpha_desc.SetShapeRange(x_range);

    ge::TensorDesc offset_desc;
    ge::Shape yShape({-1});
    offset_desc.SetDataType(ge::DT_INT32);
    offset_desc.SetShape(yShape);
    offset_desc.SetOriginShape(yShape);
    offset_desc.SetShapeRange(offset_range);

    op.UpdateInputDesc("x1", alpha_desc);
    op.UpdateInputDesc("x2", alpha_desc);
    op.UpdateInputDesc("seq_len1", offset_desc);
    op.UpdateInputDesc("seq_len2", offset_desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    vector<std::pair<int64_t, int64_t>> output_range;
    EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
    EXPECT_EQ(output_range, expected_shape_range);
}

TEST_F(DynSeqOuterProtoTest, DynSeqOuterProtoTest_range) {
    ge::op::DynSeqOuter op;
    std::vector<std::pair<int64_t, int64_t>> offset_range = {{1, 100}};
    std::vector<std::pair<int64_t, int64_t>> x_range = {{1, 100}, {512, 512}};
    std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{1, -1}, {512, 512}};

    ge::TensorDesc alpha_desc;
    ge::Shape xShape({-1, 512});
    alpha_desc.SetDataType(ge::DT_FLOAT);
    alpha_desc.SetShape(xShape);
    alpha_desc.SetOriginShape(xShape);
    alpha_desc.SetShapeRange(x_range);

    ge::TensorDesc offset_desc;
    ge::Shape yShape({-1});
    offset_desc.SetDataType(ge::DT_INT32);
    offset_desc.SetShape(yShape);
    offset_desc.SetOriginShape(yShape);
    offset_desc.SetShapeRange(offset_range);

    op.UpdateInputDesc("x1", alpha_desc);
    op.UpdateInputDesc("x2", alpha_desc);
    op.UpdateInputDesc("seq_len1", offset_desc);
    op.UpdateInputDesc("seq_len2", offset_desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    vector<std::pair<int64_t, int64_t>> output_range;
    EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
    EXPECT_EQ(output_range, expected_shape_range);
}

TEST_F(DynSeqOuterProtoTest, DynSeqOuterProtoTest_2) {
    ge::op::DynSeqOuter op;

    ge::TensorDesc alpha_desc;
    ge::Shape xShape({8, 512});
    alpha_desc.SetDataType(ge::DT_FLOAT);
    alpha_desc.SetShape(xShape);
    alpha_desc.SetOriginShape(xShape);

    ge::TensorDesc offset_desc;
    ge::Shape yShape({8});
    offset_desc.SetDataType(ge::DT_INT32);
    offset_desc.SetShape(yShape);
    offset_desc.SetOriginShape(yShape);

    ge::Tensor constTensor;
    ge::TensorDesc constDesc(ge::Shape({8}), ge::FORMAT_ND, ge::DT_INT32);
    constDesc.SetSize(8 * sizeof(int32_t));
    constTensor.SetTensorDesc(constDesc);
    int32_t constData[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    constTensor.SetData((uint8_t *)constData, 8 * sizeof(int32_t));
    auto shape = ge::op::Constant().set_attr_value(constTensor);

    op.set_input_seq_len1(shape);
    op.set_input_seq_len2(shape);
 
    op.UpdateInputDesc("x1", alpha_desc);
    op.UpdateInputDesc("x2", alpha_desc);
    op.UpdateInputDesc("seq_len1", offset_desc);
    op.UpdateInputDesc("seq_len2", offset_desc);

    vector<bool> input_const = {false, false, true, true};
    std::vector<vector<int64_t>> expect_shapes_vector = {{8, 512}};

    CommonInferShapeOperatorWithConst(op, input_const, {}, expect_shapes_vector);
}