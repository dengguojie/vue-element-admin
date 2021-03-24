#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "nn_training_ops.h"

class appadamaxd : public testing::Test {
  protected:
    static void SetUpTestCase() {
        std::cout << "appadamaxd SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "appadamaxd TearDown" << std::endl;
    }
};

TEST_F(appadamaxd, applyadagradv2d_infershape) {
    ge::op::ApplyAdaMaxD op;
    auto tensor_desc = create_desc_shape_range({-1, -1},
    ge::DT_FLOAT, ge::FORMAT_ND, {16, 16},  ge::FORMAT_ND, {{1, 16}, {1, 16}});

    op.UpdateInputDesc("var", tensor_desc);
    op.UpdateInputDesc("m", tensor_desc);
    op.UpdateInputDesc("v", tensor_desc);
    op.UpdateInputDesc("beta1_power", create_desc({1, }, ge::DT_FLOAT));
    op.UpdateInputDesc("lr", create_desc({1, }, ge::DT_FLOAT));
    op.UpdateInputDesc("beta1", create_desc({1,}, ge::DT_FLOAT));
    op.UpdateInputDesc("beta2", create_desc({1, }, ge::DT_FLOAT));
    op.UpdateInputDesc("epsilon", create_desc({1, }, ge::DT_FLOAT));
    op.UpdateInputDesc("grad", tensor_desc);
    op.SetAttr("use_locking", false);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto var_desc = op.GetOutputDesc("var");
    EXPECT_EQ(var_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_var_output_shape = {-1, -1};
    EXPECT_EQ(var_desc.GetShape().GetDims(), expected_var_output_shape);

    std::vector<std::pair<int64_t, int64_t>> var_output_shape_range;
    EXPECT_EQ(var_desc.GetShapeRange(var_output_shape_range), ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t, int64_t>> expected_var_shape_range = {{1, 16}, {1, 16}};
    EXPECT_EQ(var_output_shape_range, expected_var_shape_range);

    auto m_desc = op.GetOutputDesc("m");
    EXPECT_EQ(m_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_m_output_shape = {-1, -1};
    EXPECT_EQ(m_desc.GetShape().GetDims(), expected_m_output_shape);

    std::vector<std::pair<int64_t, int64_t>> m_output_shape_range;
    EXPECT_EQ(m_desc.GetShapeRange(m_output_shape_range), ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t, int64_t>> expected_m_shape_range = {{1, 16}, {1, 16}};
    EXPECT_EQ(m_output_shape_range, expected_m_shape_range);

    auto v_desc = op.GetOutputDesc("v");
    EXPECT_EQ(v_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_v_output_shape = {-1, -1};
    EXPECT_EQ(v_desc.GetShape().GetDims(), expected_v_output_shape);

    std::vector<std::pair<int64_t, int64_t>> v_output_shape_range;
    EXPECT_EQ(v_desc.GetShapeRange(v_output_shape_range), ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t, int64_t>> expected_v_shape_range = {{1, 16}, {1, 16}};
    EXPECT_EQ(expected_v_shape_range, expected_v_shape_range);
}