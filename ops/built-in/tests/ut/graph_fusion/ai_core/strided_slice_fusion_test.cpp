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

class strided_slice_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "strided_slice SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "strided_slice TearDown" << std::endl;
    }
};
TEST_F(strided_slice_fusion_test, strided_slice_fusion_test_1) {
  ge::Graph graph("strided_slice_fusion_test_1");
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

  auto begin = op::Constant().set_attr_value(begin_tensor);
  auto end = op::Constant().set_attr_value(end_tensor);
  auto strides = op::Constant().set_attr_value(strides_tensor);
  auto data_x =  op::Data().set_attr_index(0);
  auto strided_slice_op = op::StridedSlice("strided_slice_op")
                          .set_input_x(data_x)
                          .set_input_begin(begin)
                          .set_input_end(end)
                          .set_input_strides(strides)
                          .set_attr_begin_mask(0)
                          .set_attr_end_mask(0)
                          .set_attr_ellipsis_mask(0)
                          .set_attr_new_axis_mask(0)
                          .set_attr_shrink_axis_mask(0);

  begin.update_output_desc_y(begin_desc);
  end.update_output_desc_y(end_desc);
  strides.update_output_desc_y(strides_desc);

  std::vector<int64_t> data_x_vec{10,12,12};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_INT32);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);
  std::vector<int64_t> output_vec{3,2,2};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_ND, DT_INT32);
  strided_slice_op.update_input_desc_x(output_desc);
  strided_slice_op.update_output_desc_y(output_desc);
  std::vector<Operator> inputs{data_x, begin, end, strides};
  std::vector<Operator> outputs{strided_slice_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "strided_slice_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrStridedSliceFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  vector<int64_t> expectShape{3,2,2};
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "StridedSliceD") {
            findOp = true;
        }
    }
  EXPECT_EQ(findOp, true);
}


TEST_F(strided_slice_fusion_test, strided_slice_fusion_test_2) {
  ge::Graph graph("strided_slice_fusion_test_2");
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

  auto begin = op::Constant().set_attr_value(begin_tensor);
  auto end = op::Constant().set_attr_value(end_tensor);
  auto strides = op::Constant().set_attr_value(strides_tensor);
  auto data_x =  op::Data().set_attr_index(0);
  auto strided_slice_op = op::StridedSlice("strided_slice_op")
                          .set_input_x(data_x)
                          .set_input_begin(begin)
                          .set_input_end(end)
                          .set_input_strides(strides)
                          .set_attr_begin_mask(0)
                          .set_attr_end_mask(0)
                          .set_attr_ellipsis_mask(0)
                          .set_attr_new_axis_mask(0)
                          .set_attr_shrink_axis_mask(4);

  begin.update_output_desc_y(begin_desc);
  end.update_output_desc_y(end_desc);
  strides.update_output_desc_y(strides_desc);

  std::vector<int64_t> data_x_vec{10,12,12};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_INT32);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);
  std::vector<int64_t> output_vec{3,2};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_ND, DT_INT32);
  strided_slice_op.update_input_desc_x(output_desc);
  strided_slice_op.update_output_desc_y(output_desc);
  std::vector<Operator> inputs{data_x, begin, end, strides};
  std::vector<Operator> outputs{strided_slice_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "strided_slice_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrStridedSliceFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  vector<int64_t> expectShape{3,2};
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "StridedSliceD") {
            findOp = true;
        }
    }
  EXPECT_EQ(findOp, false);
}


TEST_F(strided_slice_fusion_test, strided_slice_fusion_test_3) {
  ge::Graph graph("strided_slice_fusion_test_3");
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
  *(begin_data + 2) = 1;
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
  *(end_data + 2) = 3;
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
  *(strides_data + 0) = 2;
  *(strides_data + 1) = 2;
  *(strides_data + 2) = 1;
  strides_tensor.SetData((uint8_t*)strides_data, strides_size * sizeof(int32_t));
  delete [] strides_data;

  auto begin = op::Constant().set_attr_value(begin_tensor);
  auto end = op::Constant().set_attr_value(end_tensor);
  auto strides = op::Constant().set_attr_value(strides_tensor);
  auto data_x =  op::Data().set_attr_index(0);
  auto strided_slice_op = op::StridedSlice("strided_slice_op")
                          .set_input_x(data_x)
                          .set_input_begin(begin)
                          .set_input_end(end)
                          .set_input_strides(strides)
                          .set_attr_begin_mask(0)
                          .set_attr_end_mask(3)
                          .set_attr_ellipsis_mask(1)
                          .set_attr_new_axis_mask(6)
                          .set_attr_shrink_axis_mask(5);

  begin.update_output_desc_y(begin_desc);
  end.update_output_desc_y(end_desc);
  strides.update_output_desc_y(strides_desc);

  std::vector<int64_t> data_x_vec{3,2,2,2};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_INT32);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);
  std::vector<int64_t> output_vec{3,2,2,2,1,1};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_ND, DT_INT32);
  strided_slice_op.update_input_desc_x(output_desc);
  strided_slice_op.update_output_desc_y(output_desc);
  std::vector<Operator> inputs{data_x, begin, end, strides};
  std::vector<Operator> outputs{strided_slice_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "strided_slice_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrStridedSliceFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  vector<int64_t> expectShape{3,2,2,2,1,1};
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "StridedSliceD") {
            findOp = true;
        }
    }
  EXPECT_EQ(findOp, false);
}

TEST_F(strided_slice_fusion_test, strided_slice_fusion_test_4) {
  ge::Graph graph("strided_slice_fusion_test_4");
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
  *(end_data + 0) = 0;
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

  auto begin = op::Constant().set_attr_value(begin_tensor);
  auto end = op::Constant().set_attr_value(end_tensor);
  auto strides = op::Constant().set_attr_value(strides_tensor);
  auto data_x =  op::Data().set_attr_index(0);
  auto strided_slice_op = op::StridedSlice("strided_slice_op")
                          .set_input_x(data_x)
                          .set_input_begin(begin)
                          .set_input_end(end)
                          .set_input_strides(strides)
                          .set_attr_begin_mask(0)
                          .set_attr_end_mask(0)
                          .set_attr_ellipsis_mask(0)
                          .set_attr_new_axis_mask(0)
                          .set_attr_shrink_axis_mask(0);

  begin.update_output_desc_y(begin_desc);
  end.update_output_desc_y(end_desc);
  strides.update_output_desc_y(strides_desc);

  std::vector<int64_t> data_x_vec{10,12,12};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_INT32);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);
  std::vector<int64_t> output_vec{0,2,4};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_ND, DT_INT32);
  strided_slice_op.update_input_desc_x(output_desc);
  strided_slice_op.update_output_desc_y(output_desc);
  std::vector<Operator> inputs{data_x, begin, end, strides};
  std::vector<Operator> outputs{strided_slice_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "strided_slice_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrStridedSliceFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "StridedSliceD") {
            findOp = true;
        }
    }
  EXPECT_EQ(findOp, false);
}

TEST_F(strided_slice_fusion_test, strided_slice_fusion_test_5) {
  ge::Graph graph("strided_slice_fusion_test_5");
  ge::Tensor begin_tensor;
  std::vector<int64_t> begin_vec{2};
  ge::Shape begin_shape(begin_vec);
  ge::TensorDesc begin_desc(begin_shape, FORMAT_ND, DT_INT32);
  int32_t begin_size = begin_desc.GetShape().GetShapeSize();
  begin_desc.SetSize(begin_size * sizeof(int32_t));
  begin_tensor.SetTensorDesc(begin_desc);
  int32_t* begin_data = nullptr;
  begin_data = new int32_t[begin_size];
  *(begin_data + 0) = 0;
  *(begin_data + 1) = 0;
  begin_tensor.SetData((uint8_t*)begin_data, begin_size * sizeof(int32_t));
  delete [] begin_data;

  ge::Tensor end_tensor;
  std::vector<int64_t> end_vec{2};
  ge::Shape end_shape(end_vec);
  ge::TensorDesc end_desc(end_shape, FORMAT_ND, DT_INT32);
  int32_t end_size = end_desc.GetShape().GetShapeSize();
  end_desc.SetSize(end_size * sizeof(int32_t));
  end_tensor.SetTensorDesc(end_desc);
  int32_t* end_data = nullptr;
  end_data = new int32_t[end_size];
  *(end_data + 0) = 1;
  *(end_data + 1) = 3;
  end_tensor.SetData((uint8_t*)end_data, end_size * sizeof(int32_t));
  delete [] end_data;

  ge::Tensor strides_tensor;
  std::vector<int64_t> strides_vec{2};
  ge::Shape strides_shape(strides_vec);
  ge::TensorDesc strides_desc(strides_shape, FORMAT_ND, DT_INT32);
  int32_t strides_size = strides_desc.GetShape().GetShapeSize();
  strides_desc.SetSize(strides_size * sizeof(int32_t));
  strides_tensor.SetTensorDesc(strides_desc);
  int32_t* strides_data = nullptr;
  strides_data = new int32_t[strides_size];
  *(strides_data + 0) = 1;
  *(strides_data + 1) = 1;
  strides_tensor.SetData((uint8_t*)strides_data, strides_size * sizeof(int32_t));
  delete [] strides_data;

  auto begin = op::Constant().set_attr_value(begin_tensor);
  auto end = op::Constant().set_attr_value(end_tensor);
  auto strides = op::Constant().set_attr_value(strides_tensor);
  auto data_x =  op::Data().set_attr_index(0);
  auto strided_slice_op = op::StridedSlice("strided_slice_op")
                          .set_input_x(data_x)
                          .set_input_begin(begin)
                          .set_input_end(end)
                          .set_input_strides(strides)
                          .set_attr_begin_mask(0)
                          .set_attr_end_mask(0)
                          .set_attr_ellipsis_mask(0)
                          .set_attr_new_axis_mask(1)
                          .set_attr_shrink_axis_mask(3);

  begin.update_output_desc_y(begin_desc);
  end.update_output_desc_y(end_desc);
  strides.update_output_desc_y(strides_desc);

  std::vector<int64_t> data_x_vec{4};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_INT32);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);
  std::vector<int64_t> output_vec{1};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_ND, DT_INT32);
  strided_slice_op.update_input_desc_x(output_desc);
  strided_slice_op.update_output_desc_y(output_desc);
  std::vector<Operator> inputs{data_x, begin, end, strides};
  std::vector<Operator> outputs{strided_slice_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "strided_slice_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrStridedSliceFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "StridedSliceD") {
            findOp = true;
        }
    }
  EXPECT_EQ(findOp, true);
}

TEST_F(strided_slice_fusion_test, strided_slice_fusion_test_6) {
  ge::Graph graph("strided_slice_fusion_test_6");
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
  *(end_data + 0) = 1;
  *(end_data + 1) = 2;
  *(end_data + 2) = 4;
  end_tensor.SetData((uint8_t*)end_data, end_size * sizeof(int32_t));
  delete [] end_data;

  auto begin = op::Constant().set_attr_value(begin_tensor);
  auto end = op::Constant().set_attr_value(end_tensor);
  auto strides = op::Data().set_attr_index(0);
  auto data_x =  op::Data().set_attr_index(0);
  auto strided_slice_op = op::StridedSlice("strided_slice_op")
                          .set_input_x(data_x)
                          .set_input_begin(begin)
                          .set_input_end(end)
                          .set_input_strides(strides)
                          .set_attr_begin_mask(0)
                          .set_attr_end_mask(0)
                          .set_attr_ellipsis_mask(0)
                          .set_attr_new_axis_mask(0)
                          .set_attr_shrink_axis_mask(0);

  begin.update_output_desc_y(begin_desc);
  end.update_output_desc_y(end_desc);

  std::vector<int64_t> data_x_vec{10,12,12};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_INT32);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);
  std::vector<int64_t>strides_vec{3};
  ge::Shape strides_shape(strides_vec);
  ge::TensorDesc strides_desc(strides_shape, FORMAT_ND, DT_INT32);
  strides.update_input_desc_x(strides_desc);
  strides.update_output_desc_y(strides_desc);
  std::vector<int64_t> output_vec{1,2,4};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_ND, DT_INT32);
  strided_slice_op.update_input_desc_x(output_desc);
  strided_slice_op.update_output_desc_y(output_desc);
  std::vector<Operator> inputs{data_x, begin, end, strides};
  std::vector<Operator> outputs{strided_slice_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "strided_slice_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrStridedSliceFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "StridedSliceD") {
            findOp = true;
        }
    }
  EXPECT_EQ(findOp, false);
}

TEST_F(strided_slice_fusion_test, strided_slice_fusion_test_7) {
  ge::Graph graph("strided_slice_fusion_test_7");
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

  auto end = op::Constant().set_attr_value(end_tensor);
  auto strides = op::Constant().set_attr_value(strides_tensor);
  auto begin =  op::Data().set_attr_index(0);
  auto data_x =  op::Data().set_attr_index(0);
  auto strided_slice_op = op::StridedSlice("strided_slice_op")
                          .set_input_x(data_x)
                          .set_input_begin(begin)
                          .set_input_end(end)
                          .set_input_strides(strides)
                          .set_attr_begin_mask(0)
                          .set_attr_end_mask(0)
                          .set_attr_ellipsis_mask(0)
                          .set_attr_new_axis_mask(0)
                          .set_attr_shrink_axis_mask(0);

  end.update_output_desc_y(end_desc);
  strides.update_output_desc_y(strides_desc);

  std::vector<int64_t> data_x_vec{10,12,12};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_INT32);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);
  std::vector<int64_t> begin_vec{3};
  ge::Shape begin_shape(begin_vec);
  ge::TensorDesc begin_desc(begin_shape, FORMAT_ND, DT_INT32);
  begin.update_input_desc_x(begin_desc);
  begin.update_output_desc_y(begin_desc);
  std::vector<int64_t> output_vec{3,2,4};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_ND, DT_INT32);
  strided_slice_op.update_input_desc_x(output_desc);
  strided_slice_op.update_output_desc_y(output_desc);
  std::vector<Operator> inputs{data_x, begin, end, strides};
  std::vector<Operator> outputs{strided_slice_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "strided_slice_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrStridedSliceFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  vector<int64_t> expectShape{3,2};
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "StridedSliceD") {
            findOp = true;
        }
    }
  EXPECT_EQ(findOp, false);
}

TEST_F(strided_slice_fusion_test, strided_slice_fusion_test_8) {
  ge::Graph graph("strided_slice_fusion_test_8");
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

  auto begin = op::Constant().set_attr_value(begin_tensor);
  auto strides = op::Constant().set_attr_value(strides_tensor);
  auto end =  op::Data().set_attr_index(0);
  auto data_x =  op::Data().set_attr_index(0);
  auto strided_slice_op = op::StridedSlice("strided_slice_op")
                          .set_input_x(data_x)
                          .set_input_begin(begin)
                          .set_input_end(end)
                          .set_input_strides(strides)
                          .set_attr_begin_mask(0)
                          .set_attr_end_mask(0)
                          .set_attr_ellipsis_mask(0)
                          .set_attr_new_axis_mask(0)
                          .set_attr_shrink_axis_mask(0);

  begin.update_output_desc_y(begin_desc);
  strides.update_output_desc_y(strides_desc);

  std::vector<int64_t> data_x_vec{10,12,12};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_INT32);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);
  std::vector<int64_t> end_vec{3};
  ge::Shape end_shape(end_vec);
  ge::TensorDesc end_desc(end_shape, FORMAT_ND, DT_INT32);
  end.update_input_desc_x(end_desc);
  end.update_output_desc_y(end_desc);
  std::vector<int64_t> output_vec{3,2,2};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_ND, DT_INT32);
  strided_slice_op.update_input_desc_x(output_desc);
  strided_slice_op.update_output_desc_y(output_desc);
  std::vector<Operator> inputs{data_x, begin, end, strides};
  std::vector<Operator> outputs{strided_slice_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "strided_slice_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrStridedSliceFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  vector<int64_t> expectShape{3,2};
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "StridedSliceD") {
            findOp = true;
        }
    }
  EXPECT_EQ(findOp, false);
}



TEST_F(strided_slice_fusion_test, strided_slice_fusion_test_9) {
  ge::Graph graph("strided_slice_fusion_test_9");
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
  *(end_data + 1) = 3;
  *(end_data + 2) = 3;
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

  auto begin = op::Constant().set_attr_value(begin_tensor);
  auto end = op::Constant().set_attr_value(end_tensor);
  auto strides = op::Constant().set_attr_value(strides_tensor);
  auto data_x =  op::Data().set_attr_index(0);
  auto strided_slice_op = op::StridedSlice("strided_slice_op")
                          .set_input_x(data_x)
                          .set_input_begin(begin)
                          .set_input_end(end)
                          .set_input_strides(strides)
                          .set_attr_begin_mask(0)
                          .set_attr_end_mask(0)
                          .set_attr_ellipsis_mask(0)
                          .set_attr_new_axis_mask(0)
                          .set_attr_shrink_axis_mask(0);

  begin.update_output_desc_y(begin_desc);
  end.update_output_desc_y(end_desc);
  strides.update_output_desc_y(strides_desc);

  std::vector<int64_t> data_x_vec{10,12,12};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_INT32);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);
  std::vector<int64_t> output_vec{3,3,3};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_ND, DT_INT32);
  strided_slice_op.update_input_desc_x(output_desc);
  strided_slice_op.update_output_desc_y(output_desc);
  std::vector<Operator> inputs{data_x, begin, end, strides};
  std::vector<Operator> outputs{strided_slice_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "strided_slice_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrStridedSliceFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "StridedSliceD") {
            findOp = true;
        }
    }
  EXPECT_EQ(findOp, true);
}


TEST_F(strided_slice_fusion_test, strided_slice_fusion_test_10) {
  ge::Graph graph("strided_slice_fusion_test_10");
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
  *(end_data + 1) = 3;
  *(end_data + 2) = 3;
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

  auto begin = op::Constant().set_attr_value(begin_tensor);
  auto end = op::Constant().set_attr_value(end_tensor);
  auto strides = op::Constant().set_attr_value(strides_tensor);
  auto data_x =  op::Data().set_attr_index(0);
  auto strided_slice_op = op::StridedSlice("strided_slice_op")
                          .set_input_x(data_x)
                          .set_input_begin(begin)
                          .set_input_end(end)
                          .set_input_strides(strides)
                          .set_attr_end_mask(0)
                          .set_attr_ellipsis_mask(0)
                          .set_attr_new_axis_mask(0)
                          .set_attr_shrink_axis_mask(0);

  begin.update_output_desc_y(begin_desc);
  end.update_output_desc_y(end_desc);
  strides.update_output_desc_y(strides_desc);

  std::vector<int64_t> data_x_vec{10,12,12};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_INT32);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);
  std::vector<int64_t> output_vec{3,3,3};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_ND, DT_INT32);
  strided_slice_op.update_input_desc_x(output_desc);
  strided_slice_op.update_output_desc_y(output_desc);
  std::vector<Operator> inputs{data_x, begin, end, strides};
  std::vector<Operator> outputs{strided_slice_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "strided_slice_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrStridedSliceFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "StridedSliceD") {
            findOp = true;
        }
    }
  EXPECT_EQ(findOp, true);
}

TEST_F(strided_slice_fusion_test, strided_slice_fusion_test_11) {
  ge::Graph graph("strided_slice_fusion_test_11");
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
  *(end_data + 1) = 3;
  *(end_data + 2) = 3;
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

  auto begin = op::Constant().set_attr_value(begin_tensor);
  auto end = op::Constant().set_attr_value(end_tensor);
  auto strides = op::Constant().set_attr_value(strides_tensor);
  auto data_x =  op::Data().set_attr_index(0);
  auto strided_slice_op = op::StridedSlice("strided_slice_op")
                          .set_input_x(data_x)
                          .set_input_begin(begin)
                          .set_input_end(end)
                          .set_input_strides(strides)
                          .set_attr_begin_mask(0)
                          .set_attr_end_mask(0)
                          .set_attr_new_axis_mask(0)
                          .set_attr_shrink_axis_mask(0);

  begin.update_output_desc_y(begin_desc);
  end.update_output_desc_y(end_desc);
  strides.update_output_desc_y(strides_desc);

  std::vector<int64_t> data_x_vec{10,12,12};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_INT32);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);
  std::vector<int64_t> output_vec{3,3,3};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_ND, DT_INT32);
  strided_slice_op.update_input_desc_x(output_desc);
  strided_slice_op.update_output_desc_y(output_desc);
  std::vector<Operator> inputs{data_x, begin, end, strides};
  std::vector<Operator> outputs{strided_slice_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "strided_slice_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrStridedSliceFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "StridedSliceD") {
            findOp = true;
        }
    }
  EXPECT_EQ(findOp, true);
}

TEST_F(strided_slice_fusion_test, strided_slice_fusion_test_12) {
  ge::Graph graph("strided_slice_fusion_test_12");
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
  *(end_data + 1) = 3;
  *(end_data + 2) = 3;
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

  auto begin = op::Constant().set_attr_value(begin_tensor);
  auto end = op::Constant().set_attr_value(end_tensor);
  auto strides = op::Constant().set_attr_value(strides_tensor);
  auto data_x =  op::Data().set_attr_index(0);
  auto strided_slice_op = op::StridedSlice("strided_slice_op")
                          .set_input_x(data_x)
                          .set_input_begin(begin)
                          .set_input_end(end)
                          .set_input_strides(strides)
                          .set_attr_begin_mask(0)
                          .set_attr_end_mask(0)
                          .set_attr_ellipsis_mask(0)
                          .set_attr_shrink_axis_mask(0);

  begin.update_output_desc_y(begin_desc);
  end.update_output_desc_y(end_desc);
  strides.update_output_desc_y(strides_desc);

  std::vector<int64_t> data_x_vec{10,12,12};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_INT32);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);
  std::vector<int64_t> output_vec{3,3,3};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_ND, DT_INT32);
  strided_slice_op.update_input_desc_x(output_desc);
  strided_slice_op.update_output_desc_y(output_desc);
  std::vector<Operator> inputs{data_x, begin, end, strides};
  std::vector<Operator> outputs{strided_slice_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "strided_slice_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrStridedSliceFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "StridedSliceD") {
            findOp = true;
        }
    }
  EXPECT_EQ(findOp, true);
}

TEST_F(strided_slice_fusion_test, strided_slice_fusion_test_13) {
  ge::Graph graph("strided_slice_fusion_test_13");
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
  *(begin_data + 2) = 4;
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
  *(end_data + 2) = 0;
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
  *(strides_data + 2) = -2;
  strides_tensor.SetData((uint8_t*)strides_data, strides_size * sizeof(int32_t));
  delete [] strides_data;

  auto begin = op::Constant().set_attr_value(begin_tensor);
  auto end = op::Constant().set_attr_value(end_tensor);
  auto strides = op::Constant().set_attr_value(strides_tensor);
  auto data_x =  op::Data().set_attr_index(0);
  auto strided_slice_op = op::StridedSlice("strided_slice_op")
                          .set_input_x(data_x)
                          .set_input_begin(begin)
                          .set_input_end(end)
                          .set_input_strides(strides)
                          .set_attr_begin_mask(0)
                          .set_attr_end_mask(0)
                          .set_attr_ellipsis_mask(0)
                          .set_attr_new_axis_mask(0)
                          .set_attr_shrink_axis_mask(0);

  begin.update_output_desc_y(begin_desc);
  end.update_output_desc_y(end_desc);
  strides.update_output_desc_y(strides_desc);

  std::vector<int64_t> data_x_vec{10, 12, 12};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_INT32);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);
  std::vector<int64_t> output_vec{3, 2, 2};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_ND, DT_INT32);
  strided_slice_op.update_input_desc_x(output_desc);
  strided_slice_op.update_output_desc_y(output_desc);
  std::vector<Operator> inputs{data_x, begin, end, strides};
  std::vector<Operator> outputs{strided_slice_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "strided_slice_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrStridedSliceFusion",
                                              fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  vector<int64_t> expectShape{3, 2, 0};
  for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "StridedSliceD") {
            findOp = true;
        }
    }
  EXPECT_EQ(findOp, false);
}

TEST_F(strided_slice_fusion_test, strided_slice_fusion_test_14) {
  ge::Graph graph("strided_slice_fusion_test_14");
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

  auto begin = op::Constant().set_attr_value(begin_tensor);
  auto end = op::Constant().set_attr_value(end_tensor);
  auto strides = op::Constant().set_attr_value(strides_tensor);
  auto data_x =  op::Data().set_attr_index(0);
  auto strided_slice_op = op::StridedSlice("strided_slice_op")
                          .set_input_x(data_x)
                          .set_input_begin(begin)
                          .set_input_end(end)
                          .set_input_strides(strides)
                          .set_attr_begin_mask(0)
                          .set_attr_end_mask(0)
                          .set_attr_ellipsis_mask(0)
                          .set_attr_new_axis_mask(0)
                          .set_attr_shrink_axis_mask(0);

  begin.update_output_desc_y(begin_desc);
  end.update_output_desc_y(end_desc);
  strides.update_output_desc_y(strides_desc);

  std::vector<int64_t> data_x_vec{10,12,12};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, FORMAT_ND, DT_INT8);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);
  std::vector<int64_t> output_vec{3,2,2};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, FORMAT_ND, DT_INT8);
  strided_slice_op.update_input_desc_x(output_desc);
  strided_slice_op.update_output_desc_y(output_desc);
  std::vector<Operator> inputs{data_x, begin, end, strides};
  std::vector<Operator> outputs{strided_slice_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "strided_slice_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrStridedSliceFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  vector<int64_t> expectShape{3,2,2};
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "StridedSliceD") {
            findOp = true;
        }
    }
  EXPECT_EQ(findOp, true);
}
