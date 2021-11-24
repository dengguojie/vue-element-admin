#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_calculation_ops.h"
#include "split_combination_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "fp16_t.hpp"

using namespace ge;

namespace fe {

class splitv_conv2d_concat_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "splitV_conv2d_concat_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "splitV_conv2d_concat_test TearDown" << std::endl;
  }

  /************************************
   *
   *               splitV
   *       filter    /\     filter
   *           \  /       \ /
   *         conv2d ... conv2d
   *              \       /
   *                 \/
   *              concat(v2)
   *                 |
   *
   *************************************/
  void BuildGraph1(ComputeGraphPtr &compute_graph, int32_t splitv_dim, int32_t splitv_num, vector<int32_t> sizes_split) {
    ge::Graph graph("test");
    auto input0 = op::Data("input0");
    ge::Shape shape_x({1, 7, 7, 36});
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC, DT_FLOAT16);
    input0.update_input_desc_x(tensorDescX);
    input0.update_output_desc_y(tensorDescX);

    auto splitv_dim_op = op::Const("splitv_dim");
    Tensor axis;
    int32_t * dataValue = new int32_t[1];
    * dataValue = 3;
    axis.SetTensorDesc(TensorDesc(ge::Shape({1}), FORMAT_NHWC, DT_INT32));
    axis.SetData((uint8_t*)dataValue, 4);
    splitv_dim_op.set_attr_value(axis);
    splitv_dim_op.update_output_desc_y(TensorDesc(ge::Shape({1}), FORMAT_NHWC, DT_INT32));

    auto size_splits_op = op::Const("size_splits");
    Tensor axis2;
    int32_t* size_splits_data = new int32_t[3];
    for (size_t i = 0; i < 3; i++) {
        *(size_splits_data + i) = sizes_split[i];
    }

    axis2.SetTensorDesc(TensorDesc(ge::Shape({3}), FORMAT_NHWC, DT_INT32));
    axis2.SetData((uint8_t*)size_splits_data, 3*sizeof(int32_t));
    size_splits_op.set_attr_value(axis2);
    size_splits_op.update_output_desc_y(TensorDesc(ge::Shape({3}), FORMAT_NHWC, DT_INT32));

    auto splitv_layer = op::SplitV("splitV");
    splitv_layer.set_input_x(input0)
                .set_input_size_splits(size_splits_op)
                .set_input_split_dim(splitv_dim_op)
                .create_dynamic_output_y(splitv_num)
                .set_attr_num_split(splitv_num);

    TensorDesc filterDesc0(ge::Shape({3,3,size_splits_data[0],16}), FORMAT_NHWC, DT_FLOAT16);
    filterDesc0.SetOriginFormat(FORMAT_HWCN);
    auto filter_0 = op::Const("filter_0");
    Tensor filter0;
    fp16_t * filterValue0 = new fp16_t[3*3*size_splits_data[0]*16];
    filter0.SetTensorDesc(filterDesc0);
    filter0.SetData((uint8_t*)filterValue0, 2*3*3*size_splits_data[0]*16);
    filter_0.set_attr_value(filter0);
    filter_0.update_output_desc_y(filterDesc0);

    auto conv2d_0_layer = op::Conv2D("conv2d_0");
    conv2d_0_layer.set_input_x(splitv_layer, "y0")
                  .set_input_filter(filter_0)
                  .set_attr_strides({0,1,1,0})
                  .set_attr_pads({0,0,0,0});

    TensorDesc filterDesc1(ge::Shape({3,3,size_splits_data[1],16}), FORMAT_NHWC, DT_FLOAT16);
    filterDesc1.SetOriginFormat(FORMAT_HWCN);
    auto filter_1 = op::Const("filter_1");
    Tensor filter1;
    fp16_t * filterValue1 = new fp16_t[3*3*size_splits_data[1]*16];
    filter1.SetTensorDesc(filterDesc1);
    filter1.SetData((uint8_t*)filterValue1, 2*3*3*size_splits_data[1]*16);
    filter_1.set_attr_value(filter1);
    filter_1.update_output_desc_y(filterDesc1);

    auto conv2d_1_layer = op::Conv2D("conv2d_1");
    conv2d_1_layer.set_input_x(splitv_layer, "y1")
                  .set_input_filter(filter_1)
                  .set_attr_strides({0,1,1,0})
                  .set_attr_pads({0,0,0,0});

    TensorDesc filterDesc2(ge::Shape({3,3,size_splits_data[2],16}), FORMAT_NHWC, DT_FLOAT16);
    filterDesc2.SetOriginFormat(FORMAT_HWCN);
    auto filter_2 = op::Const("filter_2");
    Tensor filter2;
    fp16_t * filterValue2 = new fp16_t[3*3*size_splits_data[2]*16];
    filter2.SetTensorDesc(filterDesc2);
    filter2.SetData((uint8_t*)filterValue2, 2*3*3*size_splits_data[2]*16);
    filter_2.set_attr_value(filter2);
    filter_2.update_output_desc_y(filterDesc2);

    auto conv2d_2_layer = op::Conv2D("conv2d_2");
    conv2d_2_layer.set_input_x(splitv_layer, "y2")
                  .set_input_filter(filter_2)
                  .set_attr_strides({0,1,1,0})
                  .set_attr_pads({0,0,0,0});

    auto concat_dim = op::Const("concat_dim");
    concat_dim.set_attr_value(axis);
    concat_dim.update_output_desc_y(TensorDesc(ge::Shape({1}), FORMAT_NHWC, DT_INT32));

    auto concat_layer = op::Concat("concat");
    concat_layer.set_input_concat_dim(concat_dim)
                .create_dynamic_input_x(3)
                .set_dynamic_input_x(0, conv2d_0_layer)
                .set_dynamic_input_x(1, conv2d_1_layer)
                .set_dynamic_input_x(2, conv2d_1_layer)
                .set_attr_N(3);

    delete[] size_splits_data;
    delete[] dataValue;
    delete[] filterValue0;
    delete[] filterValue1;
    delete[] filterValue2;
    std::vector<Operator> inputs{input0, splitv_dim_op, size_splits_op, filter_0, filter_1, filter_2, concat_dim};
    std::vector<Operator> outputs{concat_layer};
    graph.SetInputs(inputs).SetOutputs(outputs);
    compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  }
};

TEST_F(splitv_conv2d_concat_test, splitv_conv2d_concat_test_1) {
  ge::ComputeGraphPtr compute_graph;
  BuildGraph1(compute_graph, 3, 3, {10, 14, 12});
  FusionPassTestUtils::InferShapeAndType(compute_graph);
  FusionPassTestUtils::RunGraphFusionPass("ASplitConv2dConcatPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph);
  bool findSplit = false;
  bool findConcat = false;
  for (auto node: compute_graph->GetAllNodes()) {
    if (node->GetType() == "SplitV") {
        findSplit = true;
    }
    if (node->GetType() == "Concat") {
        findConcat = true;
    }
  }
  EXPECT_EQ(findSplit, true);
  EXPECT_EQ(findConcat, true);
}

} // namespace fe
