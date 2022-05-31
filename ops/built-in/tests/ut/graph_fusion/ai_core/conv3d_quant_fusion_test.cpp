#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_calculation_ops.h"
#include "quantize_ops.h"
#include "nonlinear_fuc_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "fp16_t.hpp"
#define private public
#include "common/util/platform_info.h"

using namespace ge;
using namespace op;

class conv3d_quant_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "conv3d_quant_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "conv3d_quant_fusion_test TearDown" << std::endl;
    }
};

TEST_F(conv3d_quant_fusion_test, conv3d_quant_fusion_test_1) {
    ge::Graph graph("conv3d_quant_fusion_test_1");

    auto x_shape = vector<int64_t>({1, 32, 240, 352, 16});
    ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    auto data_x = op::Data("data_x");
    data_x.update_input_desc_x(x_desc);
    data_x.update_output_desc_y(x_desc);

    TensorDesc filter_desc(ge::Shape({3,3,3,16,16}), FORMAT_DHWCN, DT_FLOAT16);
    Tensor weighttensor1(filter_desc);
    auto data_filter = op::Const().set_attr_value(weighttensor1);

    auto conv3d = op::Conv3D("conv3d")
        .set_input_x(data_x)
        .set_input_filter(data_filter)
        .set_attr_strides({1, 1, 1, 1, 1})
        .set_attr_pads({1, 1, 1, 1, 1, 1})
        .set_attr_dilations({1, 1, 1, 1, 1})
        .set_attr_groups({1})
        .set_attr_data_format("NDHWC");

    TensorDesc conv3d_input_desc_x(ge::Shape(), FORMAT_NDHWC, DT_FLOAT16);
    TensorDesc conv3d_input_desc_filter(ge::Shape(), FORMAT_DHWCN, DT_FLOAT16);
    TensorDesc conv3d_output_desc_y(ge::Shape(), FORMAT_NDHWC, DT_FLOAT16);
    conv3d.update_input_desc_x(conv3d_input_desc_x);
    conv3d.update_input_desc_filter(conv3d_input_desc_filter);
    conv3d.update_output_desc_y(conv3d_output_desc_y);

    auto x2_shape = vector<int64_t>({1, 32, 240, 352, 16});
    ge::TensorDesc x2_desc(ge::Shape(x2_shape), ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    auto data_x2 = op::Data("data_x2");
    data_x.update_input_desc_x(x_desc);
    data_x.update_output_desc_y(x_desc);

    ge::Shape shape({16});
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT);
    float deqScale = 1.0;
    ge::Tensor scaleTensor(tensorDesc, reinterpret_cast<uint8_t*>(&deqScale), sizeof(float));
    auto constOp = op::Const("deq_scale").set_attr_value(scaleTensor);
    auto dequantOp = op::AscendDequant("dequant");
    dequantOp.set_input_x(conv3d)
             .set_input_deq_scale(constOp)
             .set_attr_dtype(DT_FLOAT);

    std::vector<Operator> inputs{data_x, data_x2};
    std::vector<Operator> outputs{dequantOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
}
