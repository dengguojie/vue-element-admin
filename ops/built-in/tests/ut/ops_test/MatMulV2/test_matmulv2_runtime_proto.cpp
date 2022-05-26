#include "gtest/gtest.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "error_util.h"

class MatMulV2RuntimeInferShape : public testing::Test {
};

TEST_F(MatMulV2RuntimeInferShape, basic) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV2")->infer_shape;

  gert::StorageShape x1_shape = {{32, 64}, {4, 2, 16, 16}};
  gert::StorageShape x2_shape = {{64, 128}, {8, 4, 16, 16}};
  gert::StorageShape bias_shape = {{128}, {128}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"transpose_x1", ge::AnyValue::CreateFrom<bool>(false)},
                                {"transpose_x2", ge::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(ge::Shape2String(*output), "[32, 128]");
}
