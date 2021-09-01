#include <gtest/gtest.h>
#include <vector>
#include "rnn.h"

class DynamicGruV2Test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "dynamic_gru_v2 test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "dynamic_gru_v2 test TearDown" << std::endl;
    }
};

TEST_F(DynamicGruV2Test, dynamic_gru_v2_test_case_1_failed) {
    int t = 3;
    int batch = 16;
    int input_size = 32;
    ge::op::DynamicGRUV2 op;

    ge::TensorDesc x_desc;
    ge::Shape x_shape({1, t, batch, input_size});
    x_desc.SetDataType(ge::DT_FLOAT16);
    x_desc.SetShape(x_shape);
    op.UpdateInputDesc("x", x_desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(DynamicGruV2Test, dynamic_gru_v2_test_case_2) {
    int t = 3;
    int batch = 16;
    int input_size = 32;
    int hidden_size = 48;
    ge::op::DynamicGRUV2 op;

    ge::TensorDesc x_desc;
    ge::Shape x_shape({t, batch, input_size});
    x_desc.SetDataType(ge::DT_FLOAT16);
    x_desc.SetShape(x_shape);
    op.UpdateInputDesc("x", x_desc);

    ge::TensorDesc w_input_desc;
    ge::Shape w_input_shape({input_size, 3 * hidden_size});
    w_input_desc.SetDataType(ge::DT_FLOAT16);
    w_input_desc.SetShape(w_input_shape);
    op.UpdateInputDesc("weight_input", w_input_desc);

    ge::TensorDesc w_hidden_desc;
    ge::Shape w_hidden_shape({hidden_size, 3 * hidden_size});
    w_hidden_desc.SetDataType(ge::DT_FLOAT16);
    w_hidden_desc.SetShape(w_hidden_shape);
    op.UpdateInputDesc("weight_hidden", w_hidden_desc);

    ge::TensorDesc b_input_desc;
    ge::Shape b_input_shape({3 * hidden_size,});
    b_input_desc.SetDataType(ge::DT_FLOAT16);
    b_input_desc.SetShape(b_input_shape);
    op.UpdateInputDesc("bias_input", b_input_desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {t, batch, hidden_size};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(op.GetInputDesc("weight_input").GetFormat(), ge::FORMAT_HWCN);
    EXPECT_EQ(op.GetInputDesc("weight_hidden").GetFormat(), ge::FORMAT_HWCN);
}

TEST_F(DynamicGruV2Test, dynamic_gru_v2_test_case_3) {
    int t = 3;
    int batch = 16;
    int input_size = 32;
    int hidden_size = 48;
    ge::op::DynamicGRUV2 op;

    ge::TensorDesc x_desc;
    ge::Shape x_shape({t, batch, input_size});
    x_desc.SetDataType(ge::DT_FLOAT16);
    x_desc.SetShape(x_shape);
    op.UpdateInputDesc("x", x_desc);

    ge::TensorDesc w_input_desc;
    ge::Shape w_input_shape({input_size, 3 * hidden_size});
    w_input_desc.SetDataType(ge::DT_FLOAT16);
    w_input_desc.SetShape(w_input_shape);
    op.UpdateInputDesc("weight_input", w_input_desc);

    ge::TensorDesc w_hidden_desc;
    ge::Shape w_hidden_shape({hidden_size, 3 * hidden_size});
    w_hidden_desc.SetDataType(ge::DT_FLOAT16);
    w_hidden_desc.SetShape(w_hidden_shape);
    op.UpdateInputDesc("weight_hidden", w_hidden_desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {t, batch, hidden_size};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(op.GetInputDesc("weight_input").GetFormat(), ge::FORMAT_HWCN);
    EXPECT_EQ(op.GetInputDesc("weight_hidden").GetFormat(), ge::FORMAT_HWCN);
}

TEST_F(DynamicGruV2Test, dynamic_gru_v2_test_case_4) {
    int t = 1;
    int batch = 16;
    int input_size = 15;
    int hidden_size = 15;
    ge::op::DynamicGRUV2 op;

    ge::TensorDesc x_desc;
    ge::Shape x_shape({t, batch, input_size});
    x_desc.SetDataType(ge::DT_FLOAT16);
    x_desc.SetShape(x_shape);
    op.UpdateInputDesc("x", x_desc);

    ge::TensorDesc w_input_desc;
    ge::Shape w_input_shape({input_size, 3 * hidden_size});
    w_input_desc.SetDataType(ge::DT_FLOAT16);
    w_input_desc.SetShape(w_input_shape);
    op.UpdateInputDesc("weight_input", w_input_desc);

    ge::TensorDesc w_hidden_desc;
    ge::Shape w_hidden_shape({hidden_size, 3 * hidden_size});
    w_hidden_desc.SetDataType(ge::DT_FLOAT16);
    w_hidden_desc.SetShape(w_hidden_shape);
    op.UpdateInputDesc("weight_hidden", w_hidden_desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {t, batch, hidden_size};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(op.GetInputDesc("weight_input").GetFormat(), ge::FORMAT_ND);
    EXPECT_EQ(op.GetInputDesc("weight_hidden").GetFormat(), ge::FORMAT_ND);
}