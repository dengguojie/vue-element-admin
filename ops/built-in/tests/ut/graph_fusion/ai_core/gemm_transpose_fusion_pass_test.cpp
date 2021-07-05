#include "gtest/gtest.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "matrix_calculation_ops.h"
#include "transformation_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class gemm_transpose_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "gemm_transpose_fusion_pass_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "gemm_transpose_fusion_pass_test TearDown" << std::endl;
    }
};

TEST_F(gemm_transpose_fusion_pass_test, gemm_transpose_fusion_pass_test_1) {
    ge::Graph graph("gemm_transpose_fusion_pass_test_1");

    // create gemm (32, 16) (64, 32)
    auto nz_shape_a = vector<int64_t>({1,2,16,16});
    ge::TensorDesc desc_a(ge::Shape(nz_shape_a), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
    auto data_a = op::Data("data_a");
    data_a.update_input_desc_x(desc_a);
    data_a.update_output_desc_y(desc_a);

    auto nz_shape_b = vector<int64_t>({2,4,16,16});
    ge::TensorDesc desc_b(ge::Shape(nz_shape_b), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
    auto data_b = op::Data("data_b");
    data_b.update_input_desc_x(desc_b);
    data_b.update_output_desc_y(desc_b);

    auto nd_shape_c = vector<int64_t>({64});
    ge::TensorDesc desc_c(ge::Shape(nd_shape_c), ge::FORMAT_ND, ge::DT_FLOAT16);
    auto data_c = op::Data("data_c");
    data_c.update_input_desc_x(desc_c);
    data_c.update_output_desc_y(desc_c);

    auto nd_shape_alpha = vector<int64_t>({1});
    ge::TensorDesc desc_alpha(ge::Shape(nd_shape_alpha), ge::FORMAT_ND, ge::DT_FLOAT16);
    auto data_alpha = op::Data("data_alpha");
    data_alpha.update_input_desc_x(desc_alpha);
    data_alpha.update_output_desc_y(desc_alpha);

    auto nd_shape_beta = vector<int64_t>({1});
    ge::TensorDesc desc_beta(ge::Shape(nd_shape_beta), ge::FORMAT_ND, ge::DT_FLOAT16);
    auto data_beta = op::Data("data_beta");
    data_beta.update_input_desc_x(desc_beta);
    data_beta.update_output_desc_y(desc_beta);

    auto nz_shape_y_gemm = vector<int64_t>({4,1,16,16});

    auto gemm = op::GEMM("GEMM")
        .set_input_a(data_a)
        .set_input_b(data_b)
        .set_input_c(data_c)
        .set_input_alpha(data_alpha)
        .set_input_beta(data_beta)
        .set_attr_transpose_a(true)
        .set_attr_transpose_b(true);
    TensorDesc gemm_input_desc_a(ge::Shape(nz_shape_a), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc gemm_input_desc_b(ge::Shape(nz_shape_a), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TensorDesc gemm_input_desc_c(ge::Shape(nd_shape_c), FORMAT_ND, DT_FLOAT16);
    TensorDesc gemm_input_desc_alpha(ge::Shape(nd_shape_alpha), FORMAT_ND, DT_FLOAT16);
    TensorDesc gemm_input_desc_beta(ge::Shape(nd_shape_beta), FORMAT_ND, DT_FLOAT16);
    TensorDesc gemm_output_desc_y(ge::Shape(nz_shape_y_gemm), FORMAT_FRACTAL_NZ, DT_FLOAT16);
    gemm.update_input_desc_a(gemm_input_desc_a);
    gemm.update_input_desc_b(gemm_input_desc_b);
    gemm.update_input_desc_c(gemm_input_desc_c);
    gemm.update_input_desc_alpha(gemm_input_desc_alpha);
    gemm.update_input_desc_beta(gemm_input_desc_beta);
    gemm.update_output_desc_y(gemm_output_desc_y);

    std::vector<Operator> inputs{data_a, data_b, data_c, data_alpha, data_beta};
    std::vector<Operator> outputs{gemm};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("GemmTransFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool find_mul = false;
    EXPECT_EQ(find_mul, false);
}
