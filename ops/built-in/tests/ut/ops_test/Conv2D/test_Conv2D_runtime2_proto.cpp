#include <vector>
#include "gtest/gtest.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "error_util.h"

class Conv2DRuntimeInferShape : public testing::Test {};

// Expect successful cases
TEST_F(Conv2DRuntimeInferShape, basic1)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")}
            })
        .InputShapes({&xShape, &wShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(ge::Shape2String(*output), "[1, 64, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, basic2)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{3, 90, 100, 78}, {}};
    gert::StorageShape wShape = {{66, 30, 5, 5}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 4})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 2})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")}
            })
        .InputShapes({&xShape, &wShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(ge::Shape2String(*output), "[3, 66, 46, 18]");
}

TEST_F(Conv2DRuntimeInferShape, xShapeNHWC)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 16, 16, 32}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NHWC")}
            })
        .InputShapes({&xShape, &wShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(ge::Shape2String(*output), "[1, 16, 16, 64]");
}

TEST_F(Conv2DRuntimeInferShape, wShapeNHWC)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{64, 3, 3, 32}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")}
            })
        .InputShapes({&xShape, &wShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(ge::Shape2String(*output), "[1, 64, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, wShapeHWCN)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{3, 3, 32, 64}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_HWCN, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")}
            })
        .InputShapes({&xShape, &wShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(ge::Shape2String(*output), "[1, 64, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, bias1D)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape biasShape = {{64}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")}
            })
        .InputShapes({&xShape, &wShape, &biasShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(ge::Shape2String(*output), "[1, 64, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, bias4DNCHW)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape biasShape = {{1, 64, 1, 1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")}
            })
        .InputShapes({&xShape, &wShape, &biasShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(ge::Shape2String(*output), "[1, 64, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, bias4DNHWC)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape biasShape = {{1, 1, 1, 64}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")}
            })
        .InputShapes({&xShape, &wShape, &biasShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(ge::Shape2String(*output), "[1, 64, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, paddingSAME)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -1, -1, -1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")},
            {"offset_x", ge::AnyValue::CreateFrom<int64_t>(0)},
            {"padding", ge::AnyValue::CreateFrom<std::string>("SAME")}
            })
        .InputShapes({&xShape, &wShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(ge::Shape2String(*output), "[1, 64, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, paddingVALID)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -1, -1, -1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")},
            {"offset_x", ge::AnyValue::CreateFrom<int64_t>(0)},
            {"padding", ge::AnyValue::CreateFrom<std::string>("VALID")}
            })
        .InputShapes({&xShape, &wShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(ge::Shape2String(*output), "[1, 64, 14, 14]");
}

TEST_F(Conv2DRuntimeInferShape, autoPadSAME_UPPER)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -1, -1, -1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")},
            {"offset_x", ge::AnyValue::CreateFrom<int64_t>(0)},
            {"padding", ge::AnyValue::CreateFrom<std::string>("EXPLICIT")},
            {"auto_pad", ge::AnyValue::CreateFrom<std::string>("SAME_UPPER")}
            })
        .InputShapes({&xShape, &wShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(ge::Shape2String(*output), "[1, 64, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, autoPadSAME_LOWER)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -1, -1, -1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")},
            {"offset_x", ge::AnyValue::CreateFrom<int64_t>(0)},
            {"padding", ge::AnyValue::CreateFrom<std::string>("EXPLICIT")},
            {"auto_pad", ge::AnyValue::CreateFrom<std::string>("SAME_LOWER")}
            })
        .InputShapes({&xShape, &wShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(ge::Shape2String(*output), "[1, 64, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, autoPadNOTSET)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")},
            {"offset_x", ge::AnyValue::CreateFrom<int64_t>(0)},
            {"padding", ge::AnyValue::CreateFrom<std::string>("EXPLICIT")},
            {"auto_pad", ge::AnyValue::CreateFrom<std::string>("NOTSET")}
            })
        .InputShapes({&xShape, &wShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(ge::Shape2String(*output), "[1, 64, 14, 14]");
}

TEST_F(Conv2DRuntimeInferShape, autoPadVALID)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -1, -1, -1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")},
            {"offset_x", ge::AnyValue::CreateFrom<int64_t>(0)},
            {"padding", ge::AnyValue::CreateFrom<std::string>("EXPLICIT")},
            {"auto_pad", ge::AnyValue::CreateFrom<std::string>("VALID")}
            })
        .InputShapes({&xShape, &wShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(ge::Shape2String(*output), "[1, 64, 14, 14]");
}

// Expect failed cases
TEST_F(Conv2DRuntimeInferShape, xShapeCHWN) // unsupported
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{32, 16, 16, 1}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_CHWN, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_CHWN, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("CHWN")}
            })
        .InputShapes({&xShape, &wShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, wShapeCHWN) // unsupported
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{32, 3, 3, 64}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_CHWN, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")}
            })
        .InputShapes({&xShape, &wShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, bias4DCHWN) // unsupported
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape biasShape = {{64, 1, 1, 1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_CHWN, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")}
            })
        .InputShapes({&xShape, &wShape, &biasShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, yShapeCHWN) // unsupported
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_CHWN, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")}
            })
        .InputShapes({&xShape, &wShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, bias2D) // unsupported
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape biasShape = {{64, 2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")}
            })
        .InputShapes({&xShape, &wShape, &biasShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, invalidBiasChannel)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape biasShape = {{1, 1, 1, 1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")}
            })
        .InputShapes({&xShape, &wShape, &biasShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, invalidBiasOtherDims)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape biasShape = {{1, 64, 1, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")}
            })
        .InputShapes({&xShape, &wShape, &biasShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, negativeStrides)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape biasShape = {{64}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -1, -1, -1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")}
            })
        .InputShapes({&xShape, &wShape, &biasShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, negativeDilations)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape biasShape = {{64}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -1, -1, -1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")}
            })
        .InputShapes({&xShape, &wShape, &biasShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, invalidGroups1)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(2)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")}
            })
        .InputShapes({&xShape, &wShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, invalidGroups2)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 9, 16, 16}, {}};
    gert::StorageShape wShape = {{32, 3, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(3)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")}
            })
        .InputShapes({&xShape, &wShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, invalidGroups3)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 9, 16, 16}, {}};
    gert::StorageShape wShape = {{32, 4, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")}
            })
        .InputShapes({&xShape, &wShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, unsupportedStridesDimNum)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")}
            })
        .InputShapes({&xShape, &wShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, unsupportedDilationsDimNum)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")}
            })
        .InputShapes({&xShape, &wShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, negativePads)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, -11})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")}
            })
        .InputShapes({&xShape, &wShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, unsupportedPadsDimNum)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")},
            {"offset_x", ge::AnyValue::CreateFrom<int64_t>(0)},
            {"padding", ge::AnyValue::CreateFrom<std::string>("EXPLICIT")}
            })
        .InputShapes({&xShape, &wShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, unsupportedPadding)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, -11})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")},
            {"offset_x", ge::AnyValue::CreateFrom<int64_t>(0)},
            {"padding", ge::AnyValue::CreateFrom<std::string>("INVALID_PADDING")}
            })
        .InputShapes({&xShape, &wShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, unsupportedAutoPad)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, -11})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")},
            {"offset_x", ge::AnyValue::CreateFrom<int64_t>(0)},
            {"padding", ge::AnyValue::CreateFrom<std::string>("EXPLICIT")},
            {"auto_pad", ge::AnyValue::CreateFrom<std::string>("INVALID_AUTO_PAD")}
            })
        .InputShapes({&xShape, &wShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, negativeInputWithPadsDilations)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 3, 3}, {}};
    gert::StorageShape wShape = {{64, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 3, 3})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")}
            })
        .InputShapes({&xShape, &wShape})
        .OutputShapes({&yShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}