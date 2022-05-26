#include "gtest/gtest.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "error_util.h"

class BatchMatMulRuntimeInferShape : public testing::Test {
};

TEST_F(BatchMatMulRuntimeInferShape, basic) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMul")->infer_shape;

  gert::StorageShape x1_shape = {{4, 8, 16, 32, 64}, {4, 8, 16, 4, 2, 16, 16}};
  gert::StorageShape x2_shape = {{16, 64, 64}, {16, 4, 4, 16, 16}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"adj_x1", ge::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", ge::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(ge::Shape2String(*output), "[4, 8, 16, 32, 64]");
}
