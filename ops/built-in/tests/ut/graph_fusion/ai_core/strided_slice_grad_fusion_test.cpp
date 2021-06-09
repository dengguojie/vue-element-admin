#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "selection_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class strided_slice_grad_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "strided_slice SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "strided_slice TearDown" << std::endl;
    }
};

TEST_F(strided_slice_grad_fusion_test, strided_slice_grad_fusion_test_1) {
  ge::Graph graph("strided_slice_grad_fusion_test_1");
  ge::Tensor shape_tensor;
  std::vector<int64_t> shape_vec{3};
  ge::Shape shape_shape(shape_vec);
  ge::TensorDesc shape_desc(shape_shape, FORMAT_ND, DT_INT32);
  int32_t shape_size = shape_desc.GetShape().GetShapeSize();
  shape_desc.SetSize(shape_size * sizeof(int32_t));
  shape_tensor.SetTensorDesc(shape_desc);
  int32_t* shape_data = new int32_t[shape_size];
  *(shape_data + 0) = 0;
  *(shape_data + 1) = 0;
  *(shape_data + 2) = 0;
  shape_tensor.SetData((uint8_t*)shape_data, shape_size * sizeof(int32_t));
  delete [] shape_data;

  ge::Tensor begin_tensor;
  std::vector<int64_t> begin_vec{3};
  ge::Shape begin_shape(begin_vec);
  ge::TensorDesc begin_desc(begin_shape, FORMAT_ND, DT_INT32);
  int32_t begin_size = begin_desc.GetShape().GetShapeSize();
  begin_desc.SetSize(begin_size * sizeof(int32_t));
  begin_tensor.SetTensorDesc(begin_desc);
  int32_t* begin_data = nullptr;
  begin_data = new int32_t[begin_size];
  *(begin_data + 0) = 0;
  *(begin_data + 1) = 0;
  *(begin_data + 2) = 0;
  begin_tensor.SetData((uint8_t*)begin_data, begin_size * sizeof(int32_t));
  delete [] begin_data;

  ge::Tensor end_tensor;
  std::vector<int64_t> end_vec{3};
  ge::Shape end_shape(end_vec);
  ge::TensorDesc end_desc(end_shape, FORMAT_ND, DT_INT32);
  int32_t end_size = end_desc.GetShape().GetShapeSize();
  end_desc.SetSize(end_size * sizeof(int32_t));
  end_tensor.SetTensorDesc(end_desc);
  int32_t* end_data = nullptr;
  end_data = new int32_t[end_size];
  *(end_data + 0) = 3;
  *(end_data + 1) = 2;
  *(end_data + 2) = 4;
  end_tensor.SetData((uint8_t*)end_data, end_size * sizeof(int32_t));
  delete [] end_data;

  ge::Tensor strides_tensor;
  std::vector<int64_t> strides_vec{3};
  ge::Shape strides_shape(strides_vec);
  ge::TensorDesc strides_desc(strides_shape, FORMAT_ND, DT_INT32);
  int32_t strides_size = strides_desc.GetShape().GetShapeSize();
  strides_desc.SetSize(strides_size * sizeof(int32_t));
  strides_tensor.SetTensorDesc(strides_desc);
  int32_t* strides_data = nullptr;
  strides_data = new int32_t[strides_size];
  *(strides_data + 0) = 1;
  *(strides_data + 1) = 1;
  *(strides_data + 2) = 1;
  strides_tensor.SetData((uint8_t*)strides_data, strides_size * sizeof(int32_t));
  delete [] strides_data;

  auto shape = op::Constant().set_attr_value(shape_tensor);
  auto begin = op::Constant().set_attr_value(begin_tensor);
  auto end = op::Constant().set_attr_value(end_tensor);
  auto strides = op::Constant().set_attr_value(strides_tensor);
  auto data_dy =  op::Data().set_attr_index(0);
  auto strided_slice_grad_op = op::StridedSliceGrad("strided_slice_grad_op")
                                   .set_input_shape(shape)
                                   .set_input_begin(begin)
                                   .set_input_end(end)
                                   .set_input_strides(strides)
                                   .set_input_dy(data_dy)
                                   .set_attr_begin_mask(0)
                                   .set_attr_end_mask(0)
                                   .set_attr_ellipsis_mask(0)
                                   .set_attr_new_axis_mask(0)
                                   .set_attr_shrink_axis_mask(0);

  shape.update_output_desc_y(shape_desc);
  begin.update_output_desc_y(begin_desc);
  end.update_output_desc_y(end_desc);
  strides.update_output_desc_y(strides_desc);

  std::vector<int64_t> data_x_vec{10,12,12};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_INT32);
  data_dy.update_input_desc_x(data_x_desc);
  data_dy.update_output_desc_y(data_x_desc);
  std::vector<int64_t> dy_vec{3,2,2};
  ge::Shape output_shape(dy_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_ND, DT_INT32);
  strided_slice_grad_op.update_input_desc_dy(output_desc);
  strided_slice_grad_op.update_output_desc_output(data_x_desc);
  std::vector<Operator> inputs{shape, begin, end, strides, data_dy};
  std::vector<Operator> outputs{strided_slice_grad_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("StridedSliceGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "StridedSliceGradD") {
            findOp = true;
        }
    }
  EXPECT_EQ(findOp, false);
}

TEST_F(strided_slice_grad_fusion_test, strided_slice_grad_fusion_test_2) {
  ge::Graph graph("strided_slice_grad_fusion_test_1");
  ge::Tensor shape_tensor;
  std::vector<int64_t> shape_vec{3};
  ge::Shape shape_shape(shape_vec);
  ge::TensorDesc shape_desc(shape_shape, FORMAT_ND, DT_INT32);
  int32_t shape_size = shape_desc.GetShape().GetShapeSize();
  shape_desc.SetSize(shape_size * sizeof(int32_t));
  shape_tensor.SetTensorDesc(shape_desc);
  int32_t* shape_data = new int32_t[shape_size];
  *(shape_data + 0) = 0;
  *(shape_data + 1) = 0;
  *(shape_data + 2) = 0;
  shape_tensor.SetData((uint8_t*)shape_data, shape_size * sizeof(int32_t));
  delete [] shape_data;

  ge::Tensor begin_tensor;
  std::vector<int64_t> begin_vec{3};
  ge::Shape begin_shape(begin_vec);
  ge::TensorDesc begin_desc(begin_shape, FORMAT_ND, DT_INT32);
  int32_t begin_size = begin_desc.GetShape().GetShapeSize();
  begin_desc.SetSize(begin_size * sizeof(int32_t));
  begin_tensor.SetTensorDesc(begin_desc);
  int32_t* begin_data = nullptr;
  begin_data = new int32_t[begin_size];
  *(begin_data + 0) = 0;
  *(begin_data + 1) = 0;
  *(begin_data + 2) = 0;
  begin_tensor.SetData((uint8_t*)begin_data, begin_size * sizeof(int32_t));
  delete [] begin_data;

  ge::Tensor end_tensor;
  std::vector<int64_t> end_vec{3};
  ge::Shape end_shape(end_vec);
  ge::TensorDesc end_desc(end_shape, FORMAT_ND, DT_INT32);
  int32_t end_size = end_desc.GetShape().GetShapeSize();
  end_desc.SetSize(end_size * sizeof(int32_t));
  end_tensor.SetTensorDesc(end_desc);
  int32_t* end_data = nullptr;
  end_data = new int32_t[end_size];
  *(end_data + 0) = 3;
  *(end_data + 1) = 2;
  *(end_data + 2) = 4;
  end_tensor.SetData((uint8_t*)end_data, end_size * sizeof(int32_t));
  delete [] end_data;

  ge::Tensor strides_tensor;
  std::vector<int64_t> strides_vec{3};
  ge::Shape strides_shape(strides_vec);
  ge::TensorDesc strides_desc(strides_shape, FORMAT_ND, DT_INT32);
  int32_t strides_size = strides_desc.GetShape().GetShapeSize();
  strides_desc.SetSize(strides_size * sizeof(int32_t));
  strides_tensor.SetTensorDesc(strides_desc);
  int32_t* strides_data = nullptr;
  strides_data = new int32_t[strides_size];
  *(strides_data + 0) = 1;
  *(strides_data + 1) = 1;
  *(strides_data + 2) = 1;
  strides_tensor.SetData((uint8_t*)strides_data, strides_size * sizeof(int32_t));
  delete [] strides_data;

  auto shape = op::Constant().set_attr_value(shape_tensor);
  auto begin = op::Constant().set_attr_value(begin_tensor);
  auto end = op::Constant().set_attr_value(end_tensor);
  auto strides = op::Constant().set_attr_value(strides_tensor);
  auto data_dy =  op::Data().set_attr_index(0);
  auto strided_slice_grad_op = op::StridedSliceGrad("strided_slice_grad_op")
      .set_input_shape(shape)
      .set_input_begin(begin)
      .set_input_end(end)
      .set_input_strides(strides)
      .set_input_dy(data_dy)
      .set_attr_begin_mask(0)
      .set_attr_end_mask(0)
      .set_attr_ellipsis_mask(0)
      .set_attr_new_axis_mask(0)
      .set_attr_shrink_axis_mask(1);

  shape.update_output_desc_y(shape_desc);
  begin.update_output_desc_y(begin_desc);
  end.update_output_desc_y(end_desc);
  strides.update_output_desc_y(strides_desc);

  std::vector<int64_t> data_x_vec{10,12,12};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_INT32);
  data_dy.update_input_desc_x(data_x_desc);
  data_dy.update_output_desc_y(data_x_desc);
  std::vector<int64_t> dy_vec{3,2,2};
  ge::Shape output_shape(dy_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_ND, DT_INT32);
  strided_slice_grad_op.update_input_desc_dy(output_desc);
  strided_slice_grad_op.update_output_desc_output(data_x_desc);
  std::vector<Operator> inputs{shape, begin, end, strides, data_dy};
  std::vector<Operator> outputs{strided_slice_grad_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("StridedSliceGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "StridedSliceGradD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, false);
}

TEST_F(strided_slice_grad_fusion_test, strided_slice_grad_fusion_test_3) {
  ge::Graph graph("strided_slice_grad_fusion_test_1");
  ge::Tensor shape_tensor;
  std::vector<int64_t> shape_vec{3};
  ge::Shape shape_shape(shape_vec);
  ge::TensorDesc shape_desc(shape_shape, FORMAT_ND, DT_INT32);
  int32_t shape_size = shape_desc.GetShape().GetShapeSize();
  shape_desc.SetSize(shape_size * sizeof(int32_t));
  shape_tensor.SetTensorDesc(shape_desc);
  int32_t* shape_data = new int32_t[shape_size];
  *(shape_data + 0) = 0;
  *(shape_data + 1) = 0;
  *(shape_data + 2) = 0;
  shape_tensor.SetData((uint8_t*)shape_data, shape_size * sizeof(int32_t));
  delete [] shape_data;

  ge::Tensor begin_tensor;
  std::vector<int64_t> begin_vec{3};
  ge::Shape begin_shape(begin_vec);
  ge::TensorDesc begin_desc(begin_shape, FORMAT_ND, DT_INT32);
  int32_t begin_size = begin_desc.GetShape().GetShapeSize();
  begin_desc.SetSize(begin_size * sizeof(int32_t));
  begin_tensor.SetTensorDesc(begin_desc);
  int32_t* begin_data = nullptr;
  begin_data = new int32_t[begin_size];
  *(begin_data + 0) = 0;
  *(begin_data + 1) = 0;
  *(begin_data + 2) = 0;
  begin_tensor.SetData((uint8_t*)begin_data, begin_size * sizeof(int32_t));
  delete [] begin_data;

  ge::Tensor end_tensor;
  std::vector<int64_t> end_vec{3};
  ge::Shape end_shape(end_vec);
  ge::TensorDesc end_desc(end_shape, FORMAT_ND, DT_INT32);
  int32_t end_size = end_desc.GetShape().GetShapeSize();
  end_desc.SetSize(end_size * sizeof(int32_t));
  end_tensor.SetTensorDesc(end_desc);
  int32_t* end_data = nullptr;
  end_data = new int32_t[end_size];
  *(end_data + 0) = 3;
  *(end_data + 1) = 2;
  *(end_data + 2) = 4;
  end_tensor.SetData((uint8_t*)end_data, end_size * sizeof(int32_t));
  delete [] end_data;

  ge::Tensor strides_tensor;
  std::vector<int64_t> strides_vec{3};
  ge::Shape strides_shape(strides_vec);
  ge::TensorDesc strides_desc(strides_shape, FORMAT_ND, DT_INT32);
  int32_t strides_size = strides_desc.GetShape().GetShapeSize();
  strides_desc.SetSize(strides_size * sizeof(int32_t));
  strides_tensor.SetTensorDesc(strides_desc);
  int32_t* strides_data = nullptr;
  strides_data = new int32_t[strides_size];
  *(strides_data + 0) = 1;
  *(strides_data + 1) = 1;
  *(strides_data + 2) = 2;
  strides_tensor.SetData((uint8_t*)strides_data, strides_size * sizeof(int32_t));
  delete [] strides_data;

  auto shape = op::Constant().set_attr_value(shape_tensor);
  auto begin = op::Constant().set_attr_value(begin_tensor);
  auto end = op::Constant().set_attr_value(end_tensor);
  auto strides = op::Constant().set_attr_value(strides_tensor);
  auto data_dy =  op::Data().set_attr_index(0);
  auto strided_slice_grad_op = op::StridedSliceGrad("strided_slice_grad_op")
      .set_input_shape(shape)
      .set_input_begin(begin)
      .set_input_end(end)
      .set_input_strides(strides)
      .set_input_dy(data_dy)
      .set_attr_begin_mask(0)
      .set_attr_end_mask(0)
      .set_attr_ellipsis_mask(0)
      .set_attr_new_axis_mask(0)
      .set_attr_shrink_axis_mask(1);

  shape.update_output_desc_y(shape_desc);
  begin.update_output_desc_y(begin_desc);
  end.update_output_desc_y(end_desc);
  strides.update_output_desc_y(strides_desc);

  std::vector<int64_t> data_x_vec{10,12,12};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_INT32);
  data_dy.update_input_desc_x(data_x_desc);
  data_dy.update_output_desc_y(data_x_desc);
  std::vector<int64_t> dy_vec{3,2,2};
  ge::Shape output_shape(dy_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_ND, DT_INT32);
  strided_slice_grad_op.update_input_desc_dy(output_desc);
  strided_slice_grad_op.update_output_desc_output(data_x_desc);
  std::vector<Operator> inputs{shape, begin, end, strides, data_dy};
  std::vector<Operator> outputs{strided_slice_grad_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("StridedSliceGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "StridedSliceGradD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, false);
}

