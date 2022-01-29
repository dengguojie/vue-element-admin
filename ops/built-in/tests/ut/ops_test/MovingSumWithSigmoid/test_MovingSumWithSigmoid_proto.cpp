#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "selection_ops.h"

// ----------------MovingSumWithSigmoid-------------------
class MovingSumWithSigmoidProtoTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "MovingSumWithSigmoid Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MovingSumWithSigmoid Proto Test TearDown" << std::endl;
  }
};


TEST_F(MovingSumWithSigmoidProtoTest, MovingSumWithSigmoidProtoTest_0) {
    ge::op::MovingSumWithSigmoid op;

    ge::TensorDesc alpha_desc;
    ge::Shape xShape({51200});
    alpha_desc.SetDataType(ge::DT_FLOAT);
    alpha_desc.SetShape(xShape);
    alpha_desc.SetOriginShape(xShape);

    ge::TensorDesc frame_desc;
    ge::Shape yShape({1});
    frame_desc.SetDataType(ge::DT_INT32);
    frame_desc.SetShape(yShape);
    frame_desc.SetOriginShape(yShape);

    op.UpdateInputDesc("alpha", alpha_desc);
    op.UpdateInputDesc("energy", alpha_desc);
    op.UpdateInputDesc("beam_size", frame_desc);
    op.UpdateInputDesc("frame_size", frame_desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MovingSumWithSigmoidProtoTest, MovingSumWithSigmoidProtoTest_1) {
    ge::op::MovingSumWithSigmoid op;
    std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, -1}};

    ge::TensorDesc alpha_desc;
    ge::Shape xShape({-1});
    alpha_desc.SetDataType(ge::DT_FLOAT);
    alpha_desc.SetShape(xShape);
    alpha_desc.SetOriginShape(xShape);
    alpha_desc.SetShapeRange(shape_range);

    ge::TensorDesc frame_desc;
    ge::Shape yShape({-1});
    frame_desc.SetDataType(ge::DT_INT32);
    frame_desc.SetShape(yShape);
    frame_desc.SetOriginShape(yShape);
    frame_desc.SetShapeRange(shape_range);

    op.UpdateInputDesc("alpha", alpha_desc);
    op.UpdateInputDesc("energy", alpha_desc);
    op.UpdateInputDesc("beam_size", frame_desc);
    op.UpdateInputDesc("frame_size", frame_desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}