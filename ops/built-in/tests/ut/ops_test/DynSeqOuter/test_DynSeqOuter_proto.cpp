#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "selection_ops.h"

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

TEST_F(DynSeqOuterProtoTest, DynSeqOuterProtoTest_1) {
    ge::op::DynSeqOuter op;
    std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, -1}};

    ge::TensorDesc alpha_desc;
    ge::Shape xShape({-1, 512});
    alpha_desc.SetDataType(ge::DT_FLOAT);
    alpha_desc.SetShape(xShape);
    alpha_desc.SetOriginShape(xShape);
    alpha_desc.SetShapeRange(shape_range);

    ge::TensorDesc offset_desc;
    ge::Shape yShape({-1});
    offset_desc.SetDataType(ge::DT_INT32);
    offset_desc.SetShape(yShape);
    offset_desc.SetOriginShape(yShape);
    offset_desc.SetShapeRange(shape_range);

    op.UpdateInputDesc("x1", alpha_desc);
    op.UpdateInputDesc("x2", alpha_desc);
    op.UpdateInputDesc("seq_len1", offset_desc);
    op.UpdateInputDesc("seq_len2", offset_desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}