#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "nn_training_ops.h"

class applyadagradv2d : public testing::Test {
  protected:
    static void SetUpTestCase() {
        std::cout << "applyadagradv2d SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "applyadagradv2d TearDown" << std::endl;
    }
};

TEST_F(applyadagradv2d, applyadagradv2d_infershape) {
    ge::op::ApplyAdagradV2D op;
    auto tensor_desc = create_desc_shape_range({-1, -1},
    ge::DT_FLOAT, ge::FORMAT_ND, {16, 16},  ge::FORMAT_ND, {{1, 16}, {1, 16}});

    op.UpdateInputDesc("var", tensor_desc);
    op.UpdateInputDesc("accum", tensor_desc);
    op.UpdateInputDesc("lr", tensor_desc);
    op.UpdateInputDesc("grad", tensor_desc);
    op.SetAttr("epsilon", (float)0.001);
    op.SetAttr("update_slots", true);
    op.SetAttr("use_locking", false);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto var_desc = op.GetOutputDesc("var");
    EXPECT_EQ(var_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_var_output_shape = {-1, -1};
    EXPECT_EQ(var_desc.GetShape().GetDims(), expected_var_output_shape);

    std::vector<std::pair<int64_t, int64_t>> var_output_shape_range;
    EXPECT_EQ(var_desc.GetShapeRange(var_output_shape_range), ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t, int64_t>> expected_var_shape_range = {{1, 16}, {1, 16}};
    EXPECT_EQ(var_output_shape_range, expected_var_shape_range);

    auto accum_desc = op.GetOutputDesc("accum");
    EXPECT_EQ(accum_desc.GetDataType(), ge::DT_FLOAT);
    
    std::vector<int64_t> expected_accum_output_shape = {-1, -1};
    EXPECT_EQ(accum_desc.GetShape().GetDims(), expected_accum_output_shape);

    std::vector<std::pair<int64_t, int64_t>> accum_output_shape_range;
    EXPECT_EQ(accum_desc.GetShapeRange(accum_output_shape_range), ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t, int64_t>> expected_accum_shape_range = {{1, 16}, {1, 16}};
    EXPECT_EQ(accum_output_shape_range, expected_accum_shape_range);
}