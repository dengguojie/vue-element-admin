#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "gtest/gtest.h"
#include "kernel_run_context_facker.h"
#include "matrix_calculation_ops.h"
#include "register/op_impl_registry.h"
#include "error_util.h"

class Conv2DBackpropInputRuntimeProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "Conv2DBackpropInputRuntimeProtoTest Proto Test SetUp" << std::endl; }

  static void TearDownTestCase() {
    std::cout << "Conv2DBackpropInputRuntimeProtoTest Proto Test TearDown" << std::endl;
  }
};

TEST_F(Conv2DBackpropInputRuntimeProtoTest, basic) {
  vector<int64_t> strides({1, 3, 1, 1});
  vector<int64_t> pads({0, 0, 0, 0});
  vector<int64_t> dilations({1, 1, 1, 1});
  int64_t groups = 1;
  string data_format("NHWC");

  vector<int64_t> input_size = {28, 96, 96, 2};
  gert::StorageShape input_size_shape = {{28, 96, 96, 2}, {28, 96, 96, 2}};
  gert::StorageShape filter_shape = {{62, 2, 2, 2}, {62, 2, 2, 2}};
  gert::StorageShape out_backprop_shape = {{28, 32, 95, 62}, {28, 32, 95, 62}};
  gert::StorageShape output_shape = {{}, {}};

  size_t total_size = 0;
  auto tensor_holder =
      gert::Tensor::CreateFollowing(input_size_shape.GetStorageShape().GetDimNum(), ge::DT_INT64, total_size);
  auto tensor = reinterpret_cast<gert::Tensor *>(tensor_holder.get());
  tensor->MutableStorageShape() = input_size_shape.MutableStorageShape();
  tensor->MutableOriginShape() = input_size_shape.MutableOriginShape();
  tensor->SetOriginFormat(ge::FORMAT_NHWC);
  tensor->SetStorageFormat(ge::FORMAT_NHWC);
  (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), input_size.data(),
                 input_size.size() * sizeof(int64_t));

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({tensor, &filter_shape, &out_backprop_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                                {"groups", ge::AnyValue::CreateFrom<int64_t>(groups)},
                                {"data_format", ge::AnyValue::CreateFrom<std::string>(data_format)}})
                    .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                    .Build();

  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DBackpropInput")->infer_shape;
  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(ge::Shape2String(*output), "[28, 96, 96, 2]");
}
