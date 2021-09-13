#include <gtest/gtest.h>
#include <vector>
#include "rnn.h"

class GruV2HiddenGradTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "gru_v2_hidden_grad test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "gru_v2_hidden_grad test TearDown" << std::endl;
    }
};

TEST_F(GruV2HiddenGradTest, gru_v2_hidden_grad_test_case_1) {
    int t = 3;
    int batch = 16;
    int input_dim = 32;
    int output_dim = 48;
    ge::op::DynamicGRUV2Grad gruGradOp;
    //set input param
    ge::TensorDesc tensorDesc_x;
    ge::Shape shape_x({t, batch, input_dim});
    tensorDesc_x.SetDataType(ge::DT_FLOAT16);
    tensorDesc_x.SetShape(shape_x);
    tensorDesc_x.SetOriginShape(shape_x);

    ge::TensorDesc tensorDesc_weight_ih;
    ge::Shape shape_weight_ih({input_dim, output_dim*3});
    tensorDesc_weight_ih.SetDataType(ge::DT_FLOAT16);
    tensorDesc_weight_ih.SetShape(shape_weight_ih);
    tensorDesc_weight_ih.SetOriginShape(shape_weight_ih);

    ge::TensorDesc tensorDesc_weight_hh;
    ge::Shape shape_weight_hh({output_dim, output_dim*3});
    tensorDesc_weight_hh.SetDataType(ge::DT_FLOAT16);
    tensorDesc_weight_hh.SetShape(shape_weight_hh);
    tensorDesc_weight_hh.SetOriginShape(shape_weight_hh);

    ge::TensorDesc tensorDesc_init_h;
    ge::Shape shape_init_h({batch, output_dim});
    tensorDesc_init_h.SetDataType(ge::DT_FLOAT16);
    tensorDesc_init_h.SetShape(shape_init_h);
    tensorDesc_init_h.SetOriginShape(shape_init_h);

    ge::Shape shape_h({t, batch, output_dim});
    ge::TensorDesc tensorDesc_h;
    tensorDesc_h.SetDataType(ge::DT_FLOAT16);
    tensorDesc_h.SetShape(shape_h);
    tensorDesc_h.SetOriginShape(shape_h);

    ge::TensorDesc tensorDesc_y;
    tensorDesc_y.SetDataType(ge::DT_FLOAT16);
    tensorDesc_y.SetShape(shape_h);
    tensorDesc_y.SetOriginShape(shape_h);

    ge::TensorDesc tensorDesc_dy;
    tensorDesc_dy.SetDataType(ge::DT_FLOAT16);
    tensorDesc_dy.SetShape(shape_h);
    tensorDesc_dy.SetOriginShape(shape_h);

    ge::TensorDesc tensorDesc_dh;
    tensorDesc_dh.SetDataType(ge::DT_FLOAT16);
    tensorDesc_dh.SetShape(shape_h);
    tensorDesc_dh.SetOriginShape(shape_h);

    ge::TensorDesc tensorDesc_update;
    tensorDesc_update.SetDataType(ge::DT_FLOAT16);
    tensorDesc_update.SetShape(shape_h);
    tensorDesc_update.SetOriginShape(shape_h);

    ge::TensorDesc tensorDesc_reset;
    tensorDesc_reset.SetDataType(ge::DT_FLOAT16);
    tensorDesc_reset.SetShape(shape_h);
    tensorDesc_reset.SetOriginShape(shape_h);

    ge::TensorDesc tensorDesc_new;
    tensorDesc_new.SetDataType(ge::DT_FLOAT16);
    tensorDesc_new.SetShape(shape_h);
    tensorDesc_new.SetOriginShape(shape_h);

    ge::TensorDesc tensorDesc_hidden_new;
    tensorDesc_hidden_new.SetDataType(ge::DT_FLOAT16);
    tensorDesc_hidden_new.SetShape(shape_h);
    tensorDesc_hidden_new.SetOriginShape(shape_h);

    gruGradOp.UpdateInputDesc("x", tensorDesc_x);
    gruGradOp.UpdateInputDesc("weight_input", tensorDesc_weight_ih);
    gruGradOp.UpdateInputDesc("weight_hidden", tensorDesc_weight_hh);
    gruGradOp.UpdateInputDesc("y", tensorDesc_y);
    gruGradOp.UpdateInputDesc("init_h", tensorDesc_init_h);
    gruGradOp.UpdateInputDesc("h", tensorDesc_h);
    gruGradOp.UpdateInputDesc("dy", tensorDesc_dy);
    gruGradOp.UpdateInputDesc("dh", tensorDesc_dh);
    gruGradOp.UpdateInputDesc("update", tensorDesc_update);
    gruGradOp.UpdateInputDesc("reset", tensorDesc_reset);
    gruGradOp.UpdateInputDesc("new", tensorDesc_new);
    gruGradOp.UpdateInputDesc("hidden_new", tensorDesc_hidden_new);

    auto ret = gruGradOp.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto outputDwx = gruGradOp.GetOutputDesc("dw_input");
    EXPECT_EQ(outputDwx.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_outputDwx_shape = {input_dim, output_dim*3};
    EXPECT_EQ(outputDwx.GetShape().GetDims(), expected_outputDwx_shape);

    auto outputDwh = gruGradOp.GetOutputDesc("dw_hidden");
    EXPECT_EQ(outputDwh.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_outputDwh_shape = {output_dim, output_dim*3};
    EXPECT_EQ(outputDwh.GetShape().GetDims(), expected_outputDwh_shape);

    std::vector<int64_t> expected_outputDb_shape = {output_dim*3};
    auto outputDbx = gruGradOp.GetOutputDesc("db_input");
    EXPECT_EQ(outputDbx.GetDataType(), ge::DT_FLOAT16);
    EXPECT_EQ(outputDbx.GetShape().GetDims(), expected_outputDb_shape);

    auto outputDbh = gruGradOp.GetOutputDesc("db_hidden");
    EXPECT_EQ(outputDbh.GetDataType(), ge::DT_FLOAT16);
    EXPECT_EQ(outputDbh.GetShape().GetDims(), expected_outputDb_shape);

    auto outputDx = gruGradOp.GetOutputDesc("dx");
    EXPECT_EQ(outputDx.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_outputDx_shape = {t, batch, input_dim};
    EXPECT_EQ(outputDx.GetShape().GetDims(), expected_outputDx_shape);

    auto output_dh_prev = gruGradOp.GetOutputDesc("dh_prev");
    EXPECT_EQ(output_dh_prev.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_dh_prev_shape = {batch, output_dim};
    EXPECT_EQ(output_dh_prev.GetShape().GetDims(), expected_dh_prev_shape);
}

TEST_F(GruV2HiddenGradTest, gru_v2_hidden_grad_test_case_2) {
    int t = 3;
    int batch = 16;
    int input_dim = 32;
    int output_dim = 48;
    ge::op::DynamicGRUV2Grad gruGradOp;
    //set input param
    ge::TensorDesc tensorDesc_x;
    ge::Shape shape_x({t, batch, input_dim, output_dim});
    tensorDesc_x.SetDataType(ge::DT_FLOAT16);
    tensorDesc_x.SetShape(shape_x);

    ge::TensorDesc tensorDesc_weight_ih;
    ge::Shape shape_weight_ih({input_dim, output_dim*3});
    tensorDesc_weight_ih.SetDataType(ge::DT_FLOAT16);
    tensorDesc_weight_ih.SetShape(shape_weight_ih);

    ge::TensorDesc tensorDesc_weight_hh;
    ge::Shape shape_weight_hh({output_dim, output_dim*3});
    tensorDesc_weight_hh.SetDataType(ge::DT_FLOAT16);
    tensorDesc_weight_hh.SetShape(shape_weight_hh);

    ge::TensorDesc tensorDesc_init_h;
    ge::Shape shape_init_h({batch, output_dim});
    tensorDesc_init_h.SetDataType(ge::DT_FLOAT16);
    tensorDesc_init_h.SetShape(shape_init_h);

    ge::Shape shape_h({t, batch, output_dim});
    ge::TensorDesc tensorDesc_h;
    tensorDesc_h.SetDataType(ge::DT_FLOAT16);
    tensorDesc_h.SetShape(shape_h);

    ge::TensorDesc tensorDesc_y;
    tensorDesc_y.SetDataType(ge::DT_FLOAT16);
    tensorDesc_y.SetShape(shape_h);

    ge::TensorDesc tensorDesc_dy;
    tensorDesc_dy.SetDataType(ge::DT_FLOAT16);
    tensorDesc_dy.SetShape(shape_h);

    ge::TensorDesc tensorDesc_dh;
    tensorDesc_dh.SetDataType(ge::DT_FLOAT16);
    tensorDesc_dh.SetShape(shape_h);

    ge::TensorDesc tensorDesc_update;
    tensorDesc_update.SetDataType(ge::DT_FLOAT16);
    tensorDesc_update.SetShape(shape_h);

    ge::TensorDesc tensorDesc_reset;
    tensorDesc_reset.SetDataType(ge::DT_FLOAT16);
    tensorDesc_reset.SetShape(shape_h);

    ge::TensorDesc tensorDesc_new;
    tensorDesc_new.SetDataType(ge::DT_FLOAT16);
    tensorDesc_new.SetShape(shape_h);

    ge::TensorDesc tensorDesc_hidden_new;
    tensorDesc_hidden_new.SetDataType(ge::DT_FLOAT16);
    tensorDesc_hidden_new.SetShape(shape_h);

    gruGradOp.UpdateInputDesc("x", tensorDesc_x);
    gruGradOp.UpdateInputDesc("weight_input", tensorDesc_weight_ih);
    gruGradOp.UpdateInputDesc("weight_hidden", tensorDesc_weight_hh);
    gruGradOp.UpdateInputDesc("y", tensorDesc_y);
    gruGradOp.UpdateInputDesc("init_h", tensorDesc_init_h);
    gruGradOp.UpdateInputDesc("h", tensorDesc_h);
    gruGradOp.UpdateInputDesc("dy", tensorDesc_dy);
    gruGradOp.UpdateInputDesc("dh", tensorDesc_dh);
    gruGradOp.UpdateInputDesc("update", tensorDesc_update);
    gruGradOp.UpdateInputDesc("reset", tensorDesc_reset);
    gruGradOp.UpdateInputDesc("new", tensorDesc_new);
    gruGradOp.UpdateInputDesc("hidden_new", tensorDesc_hidden_new);

    auto ret = gruGradOp.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}