#ifndef private
#define private public
#define protected public
#endif
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include <mockcpp/ChainingMockHelper.h>

#include "node_def_builder.h"
#include "cpu_kernel_utils.h"

using namespace std;
using namespace aicpu;
using aicpu::DataType;

namespace {
const char *Test = "Test";
}

namespace aicpu {
class TestKernel : public CpuKernel {
public:
    ~TestKernel() = default;

protected:
    uint32_t Compute(CpuKernelContext &ctx) override
    {
        string op = ctx.GetOpType();
        cout << "TestKernel:" << op << " success." << endl;
        Tensor *input = ctx.Input(0);
        if (input == nullptr) {
            return KERNEL_STATUS_PARAM_INVALID;
        }
        uint32_t inputSize = ctx.GetInputsSize();
        cout << "TestKernel:" << op << " input size:" << inputSize << endl;
        cout << "TestKernel:" << op << " input[0] data addr:" << input->GetData() << "data size:" <<
            input->GetDataSize() << endl;


        Tensor *output = ctx.Output(0);
        if (output == nullptr) {
            return KERNEL_STATUS_PARAM_INVALID;
        }
        uint32_t outputSize = ctx.GetOutputsSize();
        cout << "TestKernel:" << op << " output size:" << outputSize << endl;

        AttrValue *s = ctx.GetAttr("s");
        if (s == nullptr) {
            return KERNEL_STATUS_PARAM_INVALID;
        }
        cout << "TestKernel:" << op << " attr s:" << s->GetString() << endl;

        AttrValue *list_s = ctx.GetAttr("list_s");
        if (list_s == nullptr) {
            return KERNEL_STATUS_PARAM_INVALID;
        }
        cout << "TestKernel:" << op << " attr list_s size:" << list_s->GetListString().size() << " size:" <<
            list_s->ListStringSize() << " list_s[0]:" << list_s->GetListString()[0] << endl;

        AttrValue *i = ctx.GetAttr("i");
        if (i == nullptr) {
            return KERNEL_STATUS_PARAM_INVALID;
        }
        cout << "TestKernel:" << op << " attr i:" << i->GetInt() << endl;

        AttrValue *list_i = ctx.GetAttr("list_i");
        if (list_i == nullptr) {
            return KERNEL_STATUS_PARAM_INVALID;
        }
        cout << "TestKernel:" << op << " attr list_i size:" << list_i->GetListInt().size() << " size:" <<
            list_i->ListIntSize() << " list_i[0]:" << list_i->GetListInt()[0] << endl;

        AttrValue *f = ctx.GetAttr("f");
        if (i == nullptr) {
            return KERNEL_STATUS_PARAM_INVALID;
        }
        cout << "TestKernel:" << op << " attr f:" << f->GetFloat() << endl;

        AttrValue *list_f = ctx.GetAttr("list_f");
        if (list_f == nullptr) {
            return KERNEL_STATUS_PARAM_INVALID;
        }
        cout << "TestKernel:" << op << " attr list_f size:" << list_f->GetListFloat().size() << " size:" <<
            list_f->ListFloatSize() << " list_f[0]:" << list_f->GetListFloat()[0] << endl;

        AttrValue *b = ctx.GetAttr("b");
        if (b == nullptr) {
            return KERNEL_STATUS_PARAM_INVALID;
        }
        cout << "TestKernel:" << op << " attr b:" << b->GetBool() << endl;

        AttrValue *list_b = ctx.GetAttr("list_b");
        if (list_b == nullptr) {
            return KERNEL_STATUS_PARAM_INVALID;
        }
        cout << "TestKernel:" << op << " attr list_b size:" << list_b->GetListBool().size() << " size:" <<
            list_b->ListBoolSize() << " list_b[0]:" << list_b->GetListBool()[0] << endl;

        AttrValue *type = ctx.GetAttr("type");
        if (type == nullptr) {
            return KERNEL_STATUS_PARAM_INVALID;
        }
        cout << "TestKernel:" << op << " attr type:" << type->GetDataType() << endl;

        AttrValue *list_type = ctx.GetAttr("list_type");
        if (list_type == nullptr) {
            return KERNEL_STATUS_PARAM_INVALID;
        }
        cout << "TestKernel:" << op << " attr list_type size:" << list_type->GetListDataType().size() << " size:" <<
            list_type->ListDataTypeSize() << " list_type[0]:" << list_type->GetListDataType()[0] << endl;

        AttrValue *shape = ctx.GetAttr("shape");
        if (shape == nullptr) {
            return KERNEL_STATUS_PARAM_INVALID;
        }
        cout << "TestKernel:" << op << " attr shape:" << shape->GetTensorShape()->GetDimSizes().size() << endl;

        AttrValue *list_shape = ctx.GetAttr("list_shape");
        if (list_shape == nullptr) {
            return KERNEL_STATUS_PARAM_INVALID;
        }
        cout << "TestKernel:" << op << " attr list_shape size:" << list_shape->GetListTensorShape().size() <<
            " size:" << list_shape->ListTensorShapeSize() << " list_shape[0] dims size:" <<
            list_shape->GetListTensorShape()[0].GetDims() << endl;

        AttrValue *tensor = ctx.GetAttr("tensor");
        if (tensor == nullptr) {
            return KERNEL_STATUS_PARAM_INVALID;
        }
        cout << "TestKernel:" << op << " attr tensor:" << tensor->GetTensor()->GetDataType() << endl;

        AttrValue *list_tensor = ctx.GetAttr("list_tensor");
        if (list_tensor == nullptr) {
            return KERNEL_STATUS_PARAM_INVALID;
        }
        cout << "TestKernel:" << op << " attr list_tensor size:" << list_tensor->GetListTensor().size() << " size:" <<
            list_tensor->ListTensorSize() << " list_tensor[0] data type:" <<
            list_tensor->GetListTensor()[0].GetDataType() << endl;

        return KERNEL_STATUS_OK;
    }
};
REGISTER_CPU_KERNEL(Test, TestKernel);
} // namespace aicpu

class TEST_KERNEL_STest : public testing::Test {
protected:
    virtual void SetUp() {}

    virtual void TearDown()
    {
        GlobalMockObject::verify();
    }

private:
};

TEST_F(TEST_KERNEL_STest, Host)
{
    cout << "Test Kernel Begin." << endl;
    uint32_t input[5] = {96, 96, 15, 5, 4};
    uint32_t output[5] = {0};

    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    vector<string> list_s = { "aaa", "bbb" };
    vector<int64_t> list_i = { 2, 5 };
    vector<float> list_f = { 2.1, 5.2 };
    vector<bool> list_b = { true, false };
    aicpu::DataType type = DT_INT32;
    vector<aicpu::DataType> list_type = { aicpu::DataType::DT_FLOAT, aicpu::DataType::DT_INT32 };
    vector<int64_t> shape = { 3, 1, 3, 5 };
    std::vector<std::vector<int64_t>> shapeLists = { { 1, 2, 3 }, { 2, 2, 2 } };

    auto tensor = CpuKernelUtils::CreateTensor();
    tensor->SetDataType(DT_INT32);
    vector<Tensor *> list_tensor;
    auto tmp_tensor = CpuKernelUtils::CreateTensor();
    list_tensor.push_back(tmp_tensor.get());

    NodeDefBuilder(nodeDef.get(), "Test", "Test")
        .Input({ "x", DT_UINT32, { 5, 1 }, (void *)input })
        .Output({ "y", DT_UINT32, { 1 }, (void *)output })
        .Attr("sorted", true)
        .Attr("s", "aaa")
        .Attr("list_s", list_s)
        .Attr("i", 1)
        .Attr("b", true)
        .Attr("type", type)
        .Attr("f", (float)1)
        .Attr("list_i", list_i)
        .Attr("list_f", list_f)
        .Attr("list_b", list_b)
        .Attr("list_type", list_type)
        .Attr("shape", shape, "shape")
        .Attr("list_shape", shapeLists, "shape_list")
        .Attr("tensor", tensor.get())
        .Attr("list_tensor", list_tensor);

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    cout << "Test Kernel " << nodeDef->GetOpType() << " Finish. " << endl;
}

TEST_F(TEST_KERNEL_STest, Device)
{
    uint32_t input[5] = {96, 96, 15, 5, 4};
    uint32_t output[5] = {0};

    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    vector<string> list_s = { "aaa", "bbb" };
    vector<int64_t> list_i = { 2, 5 };
    vector<float> list_f = { 2.1, 5.2 };
    vector<bool> list_b = { true, false };
    aicpu::DataType type = DT_INT32;
    vector<aicpu::DataType> list_type = { aicpu::DataType::DT_FLOAT, aicpu::DataType::DT_INT32 };
    vector<int64_t> shape = { 3, 1, 3, 5 };
    std::vector<std::vector<int64_t>> shapeLists = { { 1, 2, 3 }, { 2, 2, 2 } };

    auto tensor = CpuKernelUtils::CreateTensor();
    tensor->SetDataType(DT_INT32);
    vector<Tensor *> list_tensor;
    auto tmp_tensor = CpuKernelUtils::CreateTensor();
    list_tensor.push_back(tmp_tensor.get());

    NodeDefBuilder(nodeDef.get(), "Test", "Test")
        .Input({ "x", DT_UINT32, { 5, 1 }, (void *)input })
        .Output({ "y", DT_UINT32, { 1 }, (void *)output })
        .Attr("sorted", true)
        .Attr("s", "aaa")
        .Attr("list_s", list_s)
        .Attr("i", 1)
        .Attr("b", true)
        .Attr("type", type)
        .Attr("f", (float)1)
        .Attr("list_i", list_i)
        .Attr("list_f", list_f)
        .Attr("list_b", list_b)
        .Attr("list_type", list_type)
        .Attr("shape", shape, "shape")
        .Attr("list_shape", shapeLists, "shape_list")
        .Attr("tensor", tensor.get())
        .Attr("list_tensor", list_tensor);

    std::string nodeDefStr;
    nodeDef->SerializeToString(nodeDefStr);


    std::vector<void *> io_addrs;
    io_addrs.push_back(reinterpret_cast<void *>(input));
    io_addrs.push_back(reinterpret_cast<void *>(output));
    uint32_t param_len = static_cast<uint32_t>(sizeof(AicpuParamHead) + sizeof(uint32_t) + nodeDefStr.length());
    param_len += 2 * static_cast<uint32_t>(sizeof(uint64_t));
    AicpuParamHead paramHead = { param_len, 2 };
    char *parambase = (char *)malloc(param_len * sizeof(char));
    char *param = parambase;

    memcpy(parambase, reinterpret_cast<const char *>(&paramHead), sizeof(AicpuParamHead));
    param += sizeof(AicpuParamHead);
    memcpy(param, reinterpret_cast<const char *>(io_addrs.data()),
        io_addrs.size() * static_cast<uint32_t>(sizeof(uint64_t)));
    param += io_addrs.size() * static_cast<uint32_t>(sizeof(uint64_t));
    uint32_t nodeDefLen = nodeDefStr.length();
    memcpy(param, reinterpret_cast<const char *>(&nodeDefLen), sizeof(uint32_t));
    param += sizeof(uint32_t);
    memcpy(param, static_cast<const char *>(nodeDefStr.data()), nodeDefStr.length());

    uint32_t ret = RunCpuKernel(parambase);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);

    free(parambase);
}