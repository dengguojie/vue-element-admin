#ifndef private
#define private public
#define protected public
#endif
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include <mockcpp/ChainingMockHelper.h>
#include "cpu_kernel.h"
#include "status.h"
#include "cpu_kernel_register.h"
#include "aicpu_task_struct.h"
#include "device_cpu_kernel.h"
#include "cpu_types.h"
#include "cpu_kernel_utils.h"

using namespace std;
using namespace aicpu;

class TEST_SPARSETODENSE_UTest : public testing::Test {
protected:
    virtual void SetUp()
    {
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
    }

private:
};

TEST_F(TEST_SPARSETODENSE_UTest, Host1)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int64_t indiceData[3] = {0,1,1};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {1,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT64);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(3 * sizeof(int64_t));

    //set shape
    int64_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT64);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(3 * sizeof(int64_t));

   //set value
    int16_t valueData[1] = {1};
    auto valueTensor = nodeDef->AddInputs();
    EXPECT_NE(valueTensor, nullptr);

    auto valueShape = valueTensor->GetTensorShape();
    std::vector<int64_t> valueDims = {1};
    valueShape->SetDimSizes(valueDims);
    valueTensor->SetDataType(DT_INT16);
    valueTensor->SetData(valueData);
    valueTensor->SetDataSize(sizeof(int16_t));

   //set default value
    int16_t DValueData[1] = {5};
    auto DValueTensor = nodeDef->AddInputs();
    EXPECT_NE(DValueTensor, nullptr);

    auto DValueShape = DValueTensor->GetTensorShape();
    std::vector<int64_t> DValueDims = {};
    DValueShape->SetDimSizes(DValueDims);
    DValueTensor->SetDataType(DT_INT16);
    DValueTensor->SetData(DValueData);
    DValueTensor->SetDataSize(sizeof(int16_t));

    const int32_t ind = 8;
    int16_t output[ind] = {0};

    //set output
    auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);
    auto outputShape = outputTensor->GetTensorShape();
    std::vector<int64_t> outputShapeDims = {2, 2, 2};
    outputShape->SetDimSizes(outputShapeDims);

    outputTensor->SetDataType(DT_INT16);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(ind * sizeof(int16_t));

    auto validIndice = CpuKernelUtils::CreateAttrValue();
    validIndice->SetBool(true);
    nodeDef->AddAttrs("validate_indices", validIndice.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UTest, Host2)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int64_t indiceData[3] = {0,1,1};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {1,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT64);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(3 * sizeof(int64_t));

    //set shape
    int64_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3,1};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT64);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(3 * sizeof(int64_t));

   //set value
    int16_t valueData[1] = {1};
    auto valueTensor = nodeDef->AddInputs();
    EXPECT_NE(valueTensor, nullptr);

    auto valueShape = valueTensor->GetTensorShape();
    std::vector<int64_t> valueDims = {1};
    valueShape->SetDimSizes(valueDims);
    valueTensor->SetDataType(DT_INT16);
    valueTensor->SetData(valueData);
    valueTensor->SetDataSize(sizeof(int16_t));

   //set default value
    int16_t DValueData[1] = {5};
    auto DValueTensor = nodeDef->AddInputs();
    EXPECT_NE(DValueTensor, nullptr);

    auto DValueShape = DValueTensor->GetTensorShape();
    std::vector<int64_t> DValueDims = {};
    DValueShape->SetDimSizes(DValueDims);
    DValueTensor->SetDataType(DT_INT16);
    DValueTensor->SetData(DValueData);
    DValueTensor->SetDataSize(sizeof(int16_t));

    const int32_t ind = 8;
    int16_t output[ind] = {0};

    //set output
    auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);
    auto outputShape = outputTensor->GetTensorShape();
    std::vector<int64_t> outputShapeDims = {2, 2, 2};
    outputShape->SetDimSizes(outputShapeDims);

    outputTensor->SetDataType(DT_INT16);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(ind * sizeof(int16_t));

    auto validIndice = CpuKernelUtils::CreateAttrValue();
    validIndice->SetBool(true);
    nodeDef->AddAttrs("validate_indices", validIndice.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_NE(ret, KERNEL_STATUS_OK);
}


TEST_F(TEST_SPARSETODENSE_UTest, Host3)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int64_t indiceData[3] = {0,1,1};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {1,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT64);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(3 * sizeof(int64_t));

    //set shape
    int64_t shapeData[4] = {2,2,2,1};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {4};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT64);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(4 * sizeof(int64_t));

   //set value
    int16_t valueData[1] = {1};
    auto valueTensor = nodeDef->AddInputs();
    EXPECT_NE(valueTensor, nullptr);

    auto valueShape = valueTensor->GetTensorShape();
    std::vector<int64_t> valueDims = {1};
    valueShape->SetDimSizes(valueDims);
    valueTensor->SetDataType(DT_INT16);
    valueTensor->SetData(valueData);
    valueTensor->SetDataSize(sizeof(int16_t));

   //set default value
    int16_t DValueData[1] = {5};
    auto DValueTensor = nodeDef->AddInputs();
    EXPECT_NE(DValueTensor, nullptr);

    auto DValueShape = DValueTensor->GetTensorShape();
    std::vector<int64_t> DValueDims = {};
    DValueShape->SetDimSizes(DValueDims);
    DValueTensor->SetDataType(DT_INT16);
    DValueTensor->SetData(DValueData);
    DValueTensor->SetDataSize(sizeof(int16_t));

    const int32_t ind = 8;
    int16_t output[ind] = {0};

    //set output
    auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);
    auto outputShape = outputTensor->GetTensorShape();
    std::vector<int64_t> outputShapeDims = {2, 2, 2};
    outputShape->SetDimSizes(outputShapeDims);

    outputTensor->SetDataType(DT_INT16);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(ind * sizeof(int16_t));

    auto validIndice = CpuKernelUtils::CreateAttrValue();
    validIndice->SetBool(true);
    nodeDef->AddAttrs("validate_indices", validIndice.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_NE(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UTest, Host4)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int64_t indiceData[3] = {0,1,1};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {1,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT64);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(2 * sizeof(int64_t));

    //set shape
    int64_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT64);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(3 * sizeof(int64_t));

   //set value
    int16_t valueData[1] = {1};
    auto valueTensor = nodeDef->AddInputs();
    EXPECT_NE(valueTensor, nullptr);

    auto valueShape = valueTensor->GetTensorShape();
    std::vector<int64_t> valueDims = {1};
    valueShape->SetDimSizes(valueDims);
    valueTensor->SetDataType(DT_INT16);
    valueTensor->SetData(valueData);
    valueTensor->SetDataSize(sizeof(int16_t));

   //set default value
    int16_t DValueData[1] = {5};
    auto DValueTensor = nodeDef->AddInputs();
    EXPECT_NE(DValueTensor, nullptr);

    auto DValueShape = DValueTensor->GetTensorShape();
    std::vector<int64_t> DValueDims = {};
    DValueShape->SetDimSizes(DValueDims);
    DValueTensor->SetDataType(DT_INT16);
    DValueTensor->SetData(DValueData);
    DValueTensor->SetDataSize(sizeof(int16_t));

    const int32_t ind = 8;
    int16_t output[ind] = {0};

    //set output
    auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);
    auto outputShape = outputTensor->GetTensorShape();
    std::vector<int64_t> outputShapeDims = {2, 2, 2};
    outputShape->SetDimSizes(outputShapeDims);

    outputTensor->SetDataType(DT_INT16);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(ind * sizeof(int16_t));

    auto validIndice = CpuKernelUtils::CreateAttrValue();
    validIndice->SetBool(true);
    nodeDef->AddAttrs("validate_indices", validIndice.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UTest, Host5)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int64_t indiceData[3] = {0,1,1};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {1,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT64);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(3 * sizeof(int64_t));

    //set shape
    int64_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT64);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(3 * sizeof(int64_t));

   //set value
    int16_t valueData[1] = {1};
    auto valueTensor = nodeDef->AddInputs();
    EXPECT_NE(valueTensor, nullptr);

    auto valueShape = valueTensor->GetTensorShape();
    std::vector<int64_t> valueDims = {1,1};
    valueShape->SetDimSizes(valueDims);
    valueTensor->SetDataType(DT_INT16);
    valueTensor->SetData(valueData);
    valueTensor->SetDataSize(sizeof(int16_t));

   //set default value
    int16_t DValueData[1] = {5};
    auto DValueTensor = nodeDef->AddInputs();
    EXPECT_NE(DValueTensor, nullptr);

    auto DValueShape = DValueTensor->GetTensorShape();
    std::vector<int64_t> DValueDims = {};
    DValueShape->SetDimSizes(DValueDims);
    DValueTensor->SetDataType(DT_INT16);
    DValueTensor->SetData(DValueData);
    DValueTensor->SetDataSize(sizeof(int16_t));

    const int32_t ind = 8;
    int16_t output[ind] = {0};

    //set output
    auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);
    auto outputShape = outputTensor->GetTensorShape();
    std::vector<int64_t> outputShapeDims = {2, 2, 2};
    outputShape->SetDimSizes(outputShapeDims);

    outputTensor->SetDataType(DT_INT16);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(ind * sizeof(int16_t));

    auto validIndice = CpuKernelUtils::CreateAttrValue();
    validIndice->SetBool(true);
    nodeDef->AddAttrs("validate_indices", validIndice.get());
    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_NE(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UTest, Host6)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int64_t indiceData[3] = {0,1,1};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {1,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT64);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(3 * sizeof(int64_t));

    //set shape
    int64_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT64);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(3 * sizeof(int64_t));

   //set value
    int16_t valueData[1] = {2};
    auto valueTensor = nodeDef->AddInputs();
    EXPECT_NE(valueTensor, nullptr);

    auto valueShape = valueTensor->GetTensorShape();
    std::vector<int64_t> valueDims = {3};
    valueShape->SetDimSizes(valueDims);
    valueTensor->SetDataType(DT_INT16);
    valueTensor->SetData(valueData);
    valueTensor->SetDataSize(sizeof(int16_t));

   //set default value
    int16_t DValueData[1] = {5};
    auto DValueTensor = nodeDef->AddInputs();
    EXPECT_NE(DValueTensor, nullptr);

    auto DValueShape = DValueTensor->GetTensorShape();
    std::vector<int64_t> DValueDims = {};
    DValueShape->SetDimSizes(DValueDims);
    DValueTensor->SetDataType(DT_INT16);
    DValueTensor->SetData(DValueData);
    DValueTensor->SetDataSize(sizeof(int16_t));

    const int32_t ind = 8;
    int16_t output[ind] = {0};

    //set output
    auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);
    auto outputShape = outputTensor->GetTensorShape();
    std::vector<int64_t> outputShapeDims = {2, 2, 2};
    outputShape->SetDimSizes(outputShapeDims);

    outputTensor->SetDataType(DT_INT16);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(ind * sizeof(int16_t));

    auto validIndice = CpuKernelUtils::CreateAttrValue();
    validIndice->SetBool(true);
    nodeDef->AddAttrs("validate_indices", validIndice.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_NE(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UTest, Host7)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int64_t indiceData[3] = {0,1,1};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {1,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT64);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(3 * sizeof(int64_t));

    //set shape
    int64_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT64);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(3 * sizeof(int64_t));

   //set value
    int16_t valueData[1] = {1};
    auto valueTensor = nodeDef->AddInputs();
    EXPECT_NE(valueTensor, nullptr);

    auto valueShape = valueTensor->GetTensorShape();
    std::vector<int64_t> valueDims = {1};
    valueShape->SetDimSizes(valueDims);
    valueTensor->SetDataType(DT_INT16);
    valueTensor->SetData(valueData);
    valueTensor->SetDataSize(sizeof(int16_t));

   //set default value
    int16_t DValueData[1] = {5};
    auto DValueTensor = nodeDef->AddInputs();
    EXPECT_NE(DValueTensor, nullptr);

    auto DValueShape = DValueTensor->GetTensorShape();
    std::vector<int64_t> DValueDims = {1};
    DValueShape->SetDimSizes(DValueDims);
    DValueTensor->SetDataType(DT_INT16);
    DValueTensor->SetData(DValueData);
    DValueTensor->SetDataSize(sizeof(int16_t));

    const int32_t ind = 8;
    int16_t output[ind] = {0};

    //set output
    auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);
    auto outputShape = outputTensor->GetTensorShape();
    std::vector<int64_t> outputShapeDims = {2, 2, 2};
    outputShape->SetDimSizes(outputShapeDims);

    outputTensor->SetDataType(DT_INT16);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(ind * sizeof(int16_t));

    auto validIndice = CpuKernelUtils::CreateAttrValue();
    validIndice->SetBool(true);
    nodeDef->AddAttrs("validate_indices", validIndice.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_NE(ret, KERNEL_STATUS_OK);
}


TEST_F(TEST_SPARSETODENSE_UTest, Host8)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int32_t indiceData[6] = {1,1,1,1,1,1};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {2,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT32);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(6 * sizeof(int32_t));

    //set shape
    int32_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT32);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(3 * sizeof(int32_t));

   //set value
    int16_t valueData[2] = {1,1};
    auto valueTensor = nodeDef->AddInputs();
    EXPECT_NE(valueTensor, nullptr);

    auto valueShape = valueTensor->GetTensorShape();
    std::vector<int64_t> valueDims = {2};
    valueShape->SetDimSizes(valueDims);
    valueTensor->SetDataType(DT_INT16);
    valueTensor->SetData(valueData);
    valueTensor->SetDataSize(2*sizeof(int16_t));

   //set default value
    int16_t DValueData[1] = {5};
    auto DValueTensor = nodeDef->AddInputs();
    EXPECT_NE(DValueTensor, nullptr);

    auto DValueShape = DValueTensor->GetTensorShape();
    std::vector<int64_t> DValueDims = {};
    DValueShape->SetDimSizes(DValueDims);
    DValueTensor->SetDataType(DT_INT16);
    DValueTensor->SetData(DValueData);
    DValueTensor->SetDataSize(sizeof(int16_t));

    const int32_t ind = 8;
    int16_t output[ind] = {0};

    //set output
    auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);
    auto outputShape = outputTensor->GetTensorShape();
    std::vector<int64_t> outputShapeDims = {2, 2, 2};
    outputShape->SetDimSizes(outputShapeDims);

    outputTensor->SetDataType(DT_INT16);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(ind * sizeof(int16_t));

    auto validIndice = CpuKernelUtils::CreateAttrValue();
    validIndice->SetBool(true);
    nodeDef->AddAttrs("validate_indices", validIndice.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_NE(ret, KERNEL_STATUS_OK);
}


TEST_F(TEST_SPARSETODENSE_UTest, Host9)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int64_t indiceData[3] = {0,1,1};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {1, 2, 3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT64);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(3 * sizeof(int64_t));

    //set shape
    int32_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT32);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(3 * sizeof(int32_t));

   //set value
    int16_t valueData[1] = {1};
    auto valueTensor = nodeDef->AddInputs();
    EXPECT_NE(valueTensor, nullptr);

    auto valueShape = valueTensor->GetTensorShape();
    std::vector<int64_t> valueDims = {1};
    valueShape->SetDimSizes(valueDims);
    valueTensor->SetDataType(DT_INT16);
    valueTensor->SetData(valueData);
    valueTensor->SetDataSize(sizeof(int16_t));

   //set default value
    int16_t DValueData[1] = {5};
    auto DValueTensor = nodeDef->AddInputs();
    EXPECT_NE(DValueTensor, nullptr);

    auto DValueShape = DValueTensor->GetTensorShape();
    std::vector<int64_t> DValueDims = {};
    DValueShape->SetDimSizes(DValueDims);
    DValueTensor->SetDataType(DT_INT16);
    DValueTensor->SetData(DValueData);
    DValueTensor->SetDataSize(sizeof(int16_t));

    const int32_t ind = 8;
    int16_t output[ind] = {0};

    //set output
    auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);
    auto outputShape = outputTensor->GetTensorShape();
    std::vector<int64_t> outputShapeDims = {2, 2, 2};
    outputShape->SetDimSizes(outputShapeDims);

    outputTensor->SetDataType(DT_INT16);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(ind * sizeof(int16_t));

    auto validIndice = CpuKernelUtils::CreateAttrValue();
    validIndice->SetBool(true);
    nodeDef->AddAttrs("validate_indices", validIndice.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_NE(ret, KERNEL_STATUS_OK);
}


TEST_F(TEST_SPARSETODENSE_UTest, Host10)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int64_t indiceData[6] = {1,1,1,1,1,1};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {2,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT64);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(6 * sizeof(int64_t));

    //set shape
    int32_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT32);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(3 * sizeof(int32_t));

   //set value
    int16_t valueData[3] = {1, 1, 1};
    auto valueTensor = nodeDef->AddInputs();
    EXPECT_NE(valueTensor, nullptr);

    auto valueShape = valueTensor->GetTensorShape();
    std::vector<int64_t> valueDims = {3};
    valueShape->SetDimSizes(valueDims);
    valueTensor->SetDataType(DT_INT16);
    valueTensor->SetData(valueData);
    valueTensor->SetDataSize(3*sizeof(int16_t));

   //set default value
    int16_t DValueData[1] = {5};
    auto DValueTensor = nodeDef->AddInputs();
    EXPECT_NE(DValueTensor, nullptr);

    auto DValueShape = DValueTensor->GetTensorShape();
    std::vector<int64_t> DValueDims = {};
    DValueShape->SetDimSizes(DValueDims);
    DValueTensor->SetDataType(DT_INT16);
    DValueTensor->SetData(DValueData);
    DValueTensor->SetDataSize(sizeof(int16_t));

    const int32_t ind = 8;
    int16_t output[ind] = {0};

    //set output
    auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);
    auto outputShape = outputTensor->GetTensorShape();
    std::vector<int64_t> outputShapeDims = {2, 2, 2};
    outputShape->SetDimSizes(outputShapeDims);

    outputTensor->SetDataType(DT_INT16);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(ind * sizeof(int16_t));

    auto validIndice = CpuKernelUtils::CreateAttrValue();
    validIndice->SetBool(true);
    nodeDef->AddAttrs("validate_indices", validIndice.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_NE(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UTest, Host11)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    float indiceData[6] = {0,0,0,0,0,0};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {2,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_FLOAT);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(6 * sizeof(float));

    //set shape
    int32_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT32);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(3 * sizeof(int32_t));

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_NE(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UTest, Host12)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int32_t indiceData[6] = {0,0,0,0,0,0};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {2,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT32);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(6 * sizeof(int32_t));

    //set shape
    int32_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT32);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(3 * sizeof(int32_t));

   //set value
    int16_t valueData[2] = {1,1};
    auto valueTensor = nodeDef->AddInputs();
    EXPECT_NE(valueTensor, nullptr);

    auto valueShape = valueTensor->GetTensorShape();
    std::vector<int64_t> valueDims = {2};
    valueShape->SetDimSizes(valueDims);
    valueTensor->SetDataType(DT_INT16);
    valueTensor->SetData(valueData);
    valueTensor->SetDataSize(2*sizeof(int16_t));

   //set default value
    int16_t DValueData[1] = {5};
    auto DValueTensor = nodeDef->AddInputs();
    EXPECT_NE(DValueTensor, nullptr);

    auto DValueShape = DValueTensor->GetTensorShape();
    std::vector<int64_t> DValueDims = {};
    DValueShape->SetDimSizes(DValueDims);
    DValueTensor->SetDataType(DT_INT16);
    DValueTensor->SetData(DValueData);
    DValueTensor->SetDataSize(sizeof(int16_t));

    const int32_t ind = 8;
    int16_t output[ind] = {0};

    //set output
    auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);
    auto outputShape = outputTensor->GetTensorShape();
    std::vector<int64_t> outputShapeDims = {2, 2, 2};
    outputShape->SetDimSizes(outputShapeDims);

    outputTensor->SetDataType(DT_INT16);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(ind * sizeof(int16_t));

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_NE(ret, KERNEL_STATUS_OK);
}


TEST_F(TEST_SPARSETODENSE_UTest, Host13)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int32_t indiceData[6] = {0,0,0,0,0,0};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {2,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT32);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(6 * sizeof(int32_t));

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_NE(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UTest, Host14)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int32_t indiceData[6] = {0,0,0,0,0,0};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {2,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT32);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(6 * sizeof(int32_t));

    //set shape
    int32_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT32);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(3 * sizeof(int32_t));

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_NE(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UTest, Host15)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int32_t indiceData[6] = {0,0,0,0,0,0};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {2,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT32);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(6 * sizeof(int32_t));

    //set shape
    int32_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT32);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(3 * sizeof(int32_t));

   //set value
    int16_t valueData[2] = {1,1};
    auto valueTensor = nodeDef->AddInputs();
    EXPECT_NE(valueTensor, nullptr);

    auto valueShape = valueTensor->GetTensorShape();
    std::vector<int64_t> valueDims = {2};
    valueShape->SetDimSizes(valueDims);
    valueTensor->SetDataType(DT_INT16);
    valueTensor->SetData(valueData);
    valueTensor->SetDataSize(2*sizeof(int16_t));

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_NE(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UTest, Host16)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int32_t indiceData[6] = {0,0,0,0,0,0};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {2,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT32);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(6 * sizeof(int32_t));

    //set shape
    int32_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT32);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(3 * sizeof(int32_t));

   //set value
    int16_t valueData[2] = {1,1};
    auto valueTensor = nodeDef->AddInputs();
    EXPECT_NE(valueTensor, nullptr);

    auto valueShape = valueTensor->GetTensorShape();
    std::vector<int64_t> valueDims = {2};
    valueShape->SetDimSizes(valueDims);
    valueTensor->SetDataType(DT_INT16);
    valueTensor->SetData(valueData);
    valueTensor->SetDataSize(2*sizeof(int16_t));

   //set default value
    int16_t DValueData[1] = {5};
    auto DValueTensor = nodeDef->AddInputs();
    EXPECT_NE(DValueTensor, nullptr);

    auto DValueShape = DValueTensor->GetTensorShape();
    std::vector<int64_t> DValueDims = {};
    DValueShape->SetDimSizes(DValueDims);
    DValueTensor->SetDataType(DT_INT16);
    DValueTensor->SetData(DValueData);
    DValueTensor->SetDataSize(sizeof(int16_t));

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_NE(ret, KERNEL_STATUS_OK);
}


TEST_F(TEST_SPARSETODENSE_UTest, Host17)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int64_t indiceData[3] = {0,1,1};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {1,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT64);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(3 * sizeof(int64_t));

    //set shape
    int64_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT64);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(3 * sizeof(int64_t));

   //set value
    int32_t valueData[1] = {1};
    auto valueTensor = nodeDef->AddInputs();
    EXPECT_NE(valueTensor, nullptr);

    auto valueShape = valueTensor->GetTensorShape();
    std::vector<int64_t> valueDims = {1};
    valueShape->SetDimSizes(valueDims);
    valueTensor->SetDataType(DT_INT32);
    valueTensor->SetData(valueData);
    valueTensor->SetDataSize(sizeof(int32_t));

   //set default value
    int32_t DValueData[1] = {5};
    auto DValueTensor = nodeDef->AddInputs();
    EXPECT_NE(DValueTensor, nullptr);

    auto DValueShape = DValueTensor->GetTensorShape();
    std::vector<int64_t> DValueDims = {};
    DValueShape->SetDimSizes(DValueDims);
    DValueTensor->SetDataType(DT_INT32);
    DValueTensor->SetData(DValueData);
    DValueTensor->SetDataSize(sizeof(int32_t));


    const int32_t ind = 8;
    int32_t output[ind] = {0};

    //set output
    auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);
    auto outputShape = outputTensor->GetTensorShape();
    std::vector<int64_t> outputShapeDims = {2, 2, 2};
    outputShape->SetDimSizes(outputShapeDims);

    outputTensor->SetDataType(DT_INT32);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(ind * sizeof(int32_t));

    auto validIndice = CpuKernelUtils::CreateAttrValue();
    validIndice->SetBool(false);
    nodeDef->AddAttrs("validate_indices", validIndice.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UTest, Host18)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int64_t indiceData[3] = {0,1,1};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {1,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT64);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(3 * sizeof(int64_t));

    //set shape
    int64_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT64);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(8 * sizeof(int64_t));

   //set value
    int64_t valueData[1] = {1};
    auto valueTensor = nodeDef->AddInputs();
    EXPECT_NE(valueTensor, nullptr);

    auto valueShape = valueTensor->GetTensorShape();
    std::vector<int64_t> valueDims = {1};
    valueShape->SetDimSizes(valueDims);
    valueTensor->SetDataType(DT_INT64);
    valueTensor->SetData(valueData);
    valueTensor->SetDataSize(sizeof(int64_t));

   //set default value
    int64_t DValueData[1] = {5};
    auto DValueTensor = nodeDef->AddInputs();
    EXPECT_NE(DValueTensor, nullptr);

    auto DValueShape = DValueTensor->GetTensorShape();
    std::vector<int64_t> DValueDims = {};
    DValueShape->SetDimSizes(DValueDims);
    DValueTensor->SetDataType(DT_INT64);
    DValueTensor->SetData(DValueData);
    DValueTensor->SetDataSize(sizeof(int64_t));

    const int32_t ind = 8;
    int64_t output[ind] = {0};

    //set output
    auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);
    auto outputShape = outputTensor->GetTensorShape();
    std::vector<int64_t> outputShapeDims = {2, 2, 2};
    outputShape->SetDimSizes(outputShapeDims);

    outputTensor->SetDataType(DT_INT64);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(ind * sizeof(int64_t));

    auto validIndice = CpuKernelUtils::CreateAttrValue();
    validIndice->SetBool(true);
    nodeDef->AddAttrs("validate_indices", validIndice.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}


TEST_F(TEST_SPARSETODENSE_UTest, Host19)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int64_t indiceData[3] = {0,1,1};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {1,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT64);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(3 * sizeof(int64_t));

    //set shape
    int64_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT64);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(8 * sizeof(int64_t));

   //set value
    int8_t valueData[1] = {1};
    auto valueTensor = nodeDef->AddInputs();
    EXPECT_NE(valueTensor, nullptr);

    auto valueShape = valueTensor->GetTensorShape();
    std::vector<int64_t> valueDims = {1};
    valueShape->SetDimSizes(valueDims);
    valueTensor->SetDataType(DT_INT8);
    valueTensor->SetData(valueData);
    valueTensor->SetDataSize(sizeof(int8_t));

   //set default value
    int8_t DValueData[1] = {5};
    auto DValueTensor = nodeDef->AddInputs();
    EXPECT_NE(DValueTensor, nullptr);

    auto DValueShape = DValueTensor->GetTensorShape();
    std::vector<int64_t> DValueDims = {};
    DValueShape->SetDimSizes(DValueDims);
    DValueTensor->SetDataType(DT_INT8);
    DValueTensor->SetData(DValueData);
    DValueTensor->SetDataSize(sizeof(int8_t));


    const int32_t ind = 8;
    int8_t output[ind] = {0};

    //set output
    auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);
    auto outputShape = outputTensor->GetTensorShape();
    std::vector<int64_t> outputShapeDims = {2, 2, 2};
    outputShape->SetDimSizes(outputShapeDims);

    outputTensor->SetDataType(DT_INT8);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(ind * sizeof(int8_t));

    auto validIndice = CpuKernelUtils::CreateAttrValue();
    validIndice->SetBool(true);
    nodeDef->AddAttrs("validate_indices", validIndice.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UTest, Host20)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int64_t indiceData[3] = {0,1,1};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {1,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT64);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(3 * sizeof(int64_t));

    //set shape
    int64_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT64);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(8 * sizeof(int64_t));

   //set value
    float valueData[1] = {1};
    auto valueTensor = nodeDef->AddInputs();
    EXPECT_NE(valueTensor, nullptr);

    auto valueShape = valueTensor->GetTensorShape();
    std::vector<int64_t> valueDims = {1};
    valueShape->SetDimSizes(valueDims);
    valueTensor->SetDataType(DT_FLOAT);
    valueTensor->SetData(valueData);
    valueTensor->SetDataSize(sizeof(float));

   //set default value
    float DValueData[1] = {5};
    auto DValueTensor = nodeDef->AddInputs();
    EXPECT_NE(DValueTensor, nullptr);

    auto DValueShape = DValueTensor->GetTensorShape();
    std::vector<int64_t> DValueDims = {};
    DValueShape->SetDimSizes(DValueDims);
    DValueTensor->SetDataType(DT_FLOAT);
    DValueTensor->SetData(DValueData);
    DValueTensor->SetDataSize(sizeof(float));


    const int32_t ind = 8;
    float output[ind] = {0};

    //set output
    auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);
    auto outputShape = outputTensor->GetTensorShape();
    std::vector<int64_t> outputShapeDims = {2, 2, 2};
    outputShape->SetDimSizes(outputShapeDims);

    outputTensor->SetDataType(DT_FLOAT);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(ind * sizeof(float));

    auto validIndice = CpuKernelUtils::CreateAttrValue();
    validIndice->SetBool(true);
    nodeDef->AddAttrs("validate_indices", validIndice.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}


TEST_F(TEST_SPARSETODENSE_UTest, Host21)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int64_t indiceData[3] = {0,1,1};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {1,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT64);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(3 * sizeof(int64_t));

    //set shape
    int64_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT64);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(8 * sizeof(int64_t));

   //set value
    uint16_t valueData[1] = {1};
    auto valueTensor = nodeDef->AddInputs();
    EXPECT_NE(valueTensor, nullptr);

    auto valueShape = valueTensor->GetTensorShape();
    std::vector<int64_t> valueDims = {1};
    valueShape->SetDimSizes(valueDims);
    valueTensor->SetDataType(DT_UINT16);
    valueTensor->SetData(valueData);
    valueTensor->SetDataSize(sizeof(uint16_t));

   //set default value
    uint16_t DValueData[1] = {5};
    auto DValueTensor = nodeDef->AddInputs();
    EXPECT_NE(DValueTensor, nullptr);

    auto DValueShape = DValueTensor->GetTensorShape();
    std::vector<int64_t> DValueDims = {};
    DValueShape->SetDimSizes(DValueDims);
    DValueTensor->SetDataType(DT_UINT16);
    DValueTensor->SetData(DValueData);
    DValueTensor->SetDataSize(sizeof(uint16_t));


    const int32_t ind = 8;
    uint16_t output[ind] = {0};

    //set output
    auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);
    auto outputShape = outputTensor->GetTensorShape();
    std::vector<int64_t> outputShapeDims = {2, 2, 2};
    outputShape->SetDimSizes(outputShapeDims);

    outputTensor->SetDataType(DT_UINT16);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(ind * sizeof(uint16_t));

    auto validIndice = CpuKernelUtils::CreateAttrValue();
    validIndice->SetBool(true);
    nodeDef->AddAttrs("validate_indices", validIndice.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UTest, Host22)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int64_t indiceData[3] = {0,1,1};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {1,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT64);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(3 * sizeof(int64_t));

    //set shape
    int64_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT64);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(8 * sizeof(int64_t));

   //set value
    double valueData[1] = {1};
    auto valueTensor = nodeDef->AddInputs();
    EXPECT_NE(valueTensor, nullptr);

    auto valueShape = valueTensor->GetTensorShape();
    std::vector<int64_t> valueDims = {1};
    valueShape->SetDimSizes(valueDims);
    valueTensor->SetDataType(DT_DOUBLE);
    valueTensor->SetData(valueData);
    valueTensor->SetDataSize(sizeof(double));

   //set default value
    double DValueData[1] = {5};
    auto DValueTensor = nodeDef->AddInputs();
    EXPECT_NE(DValueTensor, nullptr);

    auto DValueShape = DValueTensor->GetTensorShape();
    std::vector<int64_t> DValueDims = {};
    DValueShape->SetDimSizes(DValueDims);
    DValueTensor->SetDataType(DT_DOUBLE);
    DValueTensor->SetData(DValueData);
    DValueTensor->SetDataSize(sizeof(double));


    const int32_t ind = 8;
    double output[ind] = {0};

    //set output
    auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);
    auto outputShape = outputTensor->GetTensorShape();
    std::vector<int64_t> outputShapeDims = {2, 2, 2};
    outputShape->SetDimSizes(outputShapeDims);

    outputTensor->SetDataType(DT_DOUBLE);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(ind * sizeof(double));

    auto validIndice = CpuKernelUtils::CreateAttrValue();
    validIndice->SetBool(true);
    nodeDef->AddAttrs("validate_indices", validIndice.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UTest, Host23)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int64_t indiceData[3] = {0,1,1};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {1,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT64);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(3 * sizeof(int64_t));

    //set shape
    int64_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT64);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(8 * sizeof(int64_t));

   //set value
    bool valueData[1] = {1};
    auto valueTensor = nodeDef->AddInputs();
    EXPECT_NE(valueTensor, nullptr);

    auto valueShape = valueTensor->GetTensorShape();
    std::vector<int64_t> valueDims = {1};
    valueShape->SetDimSizes(valueDims);
    valueTensor->SetDataType(DT_BOOL);
    valueTensor->SetData(valueData);
    valueTensor->SetDataSize(sizeof(bool));

   //set default value
    bool DValueData[1] = {5};
    auto DValueTensor = nodeDef->AddInputs();
    EXPECT_NE(DValueTensor, nullptr);

    auto DValueShape = DValueTensor->GetTensorShape();
    std::vector<int64_t> DValueDims = {};
    DValueShape->SetDimSizes(DValueDims);
    DValueTensor->SetDataType(DT_BOOL);
    DValueTensor->SetData(DValueData);
    DValueTensor->SetDataSize(sizeof(bool));


    const int32_t ind = 8;
    bool output[ind] = {0};

    //set output
    auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);
    auto outputShape = outputTensor->GetTensorShape();
    std::vector<int64_t> outputShapeDims = {2, 2, 2};
    outputShape->SetDimSizes(outputShapeDims);

    outputTensor->SetDataType(DT_BOOL);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(ind * sizeof(bool));

    auto validIndice = CpuKernelUtils::CreateAttrValue();
    validIndice->SetBool(true);
    nodeDef->AddAttrs("validate_indices", validIndice.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UTest, Host24)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_NE(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UTest, Host25)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int64_t indiceData[3] = {0,1,1};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {1,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT64);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(3 * sizeof(int64_t));

    //set shape
    int64_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT64);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(8 * sizeof(int64_t));

   //set value
    uint32_t valueData[1] = {1};
    auto valueTensor = nodeDef->AddInputs();
    EXPECT_NE(valueTensor, nullptr);

    auto valueShape = valueTensor->GetTensorShape();
    std::vector<int64_t> valueDims = {1};
    valueShape->SetDimSizes(valueDims);
    valueTensor->SetDataType(DT_UINT32);
    valueTensor->SetData(valueData);
    valueTensor->SetDataSize(sizeof(uint32_t));

   //set default value
    uint32_t DValueData[1] = {5};
    auto DValueTensor = nodeDef->AddInputs();
    EXPECT_NE(DValueTensor, nullptr);

    auto DValueShape = DValueTensor->GetTensorShape();
    std::vector<int64_t> DValueDims = {};
    DValueShape->SetDimSizes(DValueDims);
    DValueTensor->SetDataType(DT_UINT32);
    DValueTensor->SetData(DValueData);
    DValueTensor->SetDataSize(sizeof(uint32_t));


    const int32_t ind = 8;
    uint32_t output[ind] = {0};

    //set output
    auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);
    auto outputShape = outputTensor->GetTensorShape();
    std::vector<int64_t> outputShapeDims = {2, 2, 2};
    outputShape->SetDimSizes(outputShapeDims);

    outputTensor->SetDataType(DT_UINT32);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(ind * sizeof(uint32_t));

    auto validIndice = CpuKernelUtils::CreateAttrValue();
    validIndice->SetBool(true);
    nodeDef->AddAttrs("validate_indices", validIndice.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_NE(ret, KERNEL_STATUS_OK);
}


TEST_F(TEST_SPARSETODENSE_UTest, Host26)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int64_t indiceData[3] = {0,1,1};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {1,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT64);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(3 * sizeof(int64_t));

    //set shape
    int64_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT64);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(8 * sizeof(int64_t));

   //set value
    int32_t valueData[1] = {1};
    auto valueTensor = nodeDef->AddInputs();
    EXPECT_NE(valueTensor, nullptr);

    auto valueShape = valueTensor->GetTensorShape();
    std::vector<int64_t> valueDims = {1};
    valueShape->SetDimSizes(valueDims);
    valueTensor->SetDataType(DT_INT32);
    valueTensor->SetData(valueData);
    valueTensor->SetDataSize(sizeof(int32_t));

   //set default value
    int32_t DValueData[1] = {5};
    auto DValueTensor = nodeDef->AddInputs();
    EXPECT_NE(DValueTensor, nullptr);

    auto DValueShape = DValueTensor->GetTensorShape();
    std::vector<int64_t> DValueDims = {};
    DValueShape->SetDimSizes(DValueDims);
    DValueTensor->SetDataType(DT_INT32);
    DValueTensor->SetData(DValueData);
    DValueTensor->SetDataSize(sizeof(int32_t));


    const int32_t ind = 8;
    int32_t output[ind] = {0};

    //set output
    auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);
    auto outputShape = outputTensor->GetTensorShape();
    std::vector<int64_t> outputShapeDims = {2, 2, 2};
    outputShape->SetDimSizes(outputShapeDims);

    outputTensor->SetDataType(DT_INT16);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(ind * sizeof(int32_t));

    auto validIndice = CpuKernelUtils::CreateAttrValue();
    validIndice->SetBool(true);
    nodeDef->AddAttrs("validate_indices", validIndice.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_NE(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UTest, Host27)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int64_t indiceData[3] = {0,1,1};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {1,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT64);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(3 * sizeof(int64_t));

    //set shape
    int64_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT64);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(8 * sizeof(int64_t));

   //set value
    int32_t valueData[1] = {1};
    auto valueTensor = nodeDef->AddInputs();
    EXPECT_NE(valueTensor, nullptr);

    auto valueShape = valueTensor->GetTensorShape();
    std::vector<int64_t> valueDims = {1};
    valueShape->SetDimSizes(valueDims);
    valueTensor->SetDataType(DT_INT32);
    valueTensor->SetData(valueData);
    valueTensor->SetDataSize(sizeof(int32_t));

   //set default value
    int32_t DValueData[1] = {5};
    auto DValueTensor = nodeDef->AddInputs();
    EXPECT_NE(DValueTensor, nullptr);

    auto DValueShape = DValueTensor->GetTensorShape();
    std::vector<int64_t> DValueDims = {};
    DValueShape->SetDimSizes(DValueDims);
    DValueTensor->SetDataType(DT_INT32);
    DValueTensor->SetData(DValueData);
    DValueTensor->SetDataSize(sizeof(int32_t));


    const int32_t ind = 8;
    int32_t output[ind] = {0};

    //set output
    auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);
    auto outputShape = outputTensor->GetTensorShape();
    std::vector<int64_t> outputShapeDims = {2, 2};
    outputShape->SetDimSizes(outputShapeDims);

    outputTensor->SetDataType(DT_INT32);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(ind * sizeof(int32_t));

    auto validIndice = CpuKernelUtils::CreateAttrValue();
    validIndice->SetBool(true);
    nodeDef->AddAttrs("validate_indices", validIndice.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_NE(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UTest, Host28)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int64_t indiceData[3] = {0,1,1};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {1,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT64);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(3 * sizeof(int64_t));

    //set shape
    int64_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT64);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(8 * sizeof(int64_t));

   //set value
    int32_t valueData[1] = {1};
    auto valueTensor = nodeDef->AddInputs();
    EXPECT_NE(valueTensor, nullptr);

    auto valueShape = valueTensor->GetTensorShape();
    std::vector<int64_t> valueDims = {1};
    valueShape->SetDimSizes(valueDims);
    valueTensor->SetDataType(DT_INT32);
    valueTensor->SetData(valueData);
    valueTensor->SetDataSize(sizeof(int32_t));

   //set default value
    int32_t DValueData[1] = {5};
    auto DValueTensor = nodeDef->AddInputs();
    EXPECT_NE(DValueTensor, nullptr);

    auto DValueShape = DValueTensor->GetTensorShape();
    std::vector<int64_t> DValueDims = {};
    DValueShape->SetDimSizes(DValueDims);
    DValueTensor->SetDataType(DT_INT32);
    DValueTensor->SetData(DValueData);
    DValueTensor->SetDataSize(sizeof(int32_t));


    const int32_t ind = 8;
    int32_t output[ind] = {0};

    //set output
    auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);
    auto outputShape = outputTensor->GetTensorShape();
    std::vector<int64_t> outputShapeDims = {1, 1, 1};
    outputShape->SetDimSizes(outputShapeDims);

    outputTensor->SetDataType(DT_INT32);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(ind * sizeof(int32_t));

    auto validIndice = CpuKernelUtils::CreateAttrValue();
    validIndice->SetBool(true);
    nodeDef->AddAttrs("validate_indices", validIndice.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_NE(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UTest, Host29)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SparseToDense");
    //set indice
    int32_t indiceData[3] = {0,1,1};

    auto indiceTensor = nodeDef->AddInputs();
    EXPECT_NE(indiceTensor, nullptr);

    auto indiceShape = indiceTensor->GetTensorShape();
    std::vector<int64_t> indiceShapeDims = {1,3};
    indiceShape->SetDimSizes(indiceShapeDims);
    indiceTensor->SetDataType(DT_INT32);
    indiceTensor->SetData(indiceData);
    indiceTensor->SetDataSize(3 * sizeof(int32_t));

    //set shape
    int64_t shapeData[3] = {2,2,2};
    auto shapeTensor = nodeDef->AddInputs();
    EXPECT_NE(shapeTensor, nullptr);

    auto shapeShape = shapeTensor->GetTensorShape();
    std::vector<int64_t> ShapeDims = {3};
    shapeShape->SetDimSizes(ShapeDims);
    shapeTensor->SetDataType(DT_INT64);
    shapeTensor->SetData(shapeData);
    shapeTensor->SetDataSize(3 * sizeof(int64_t));

   //set value
    uint8_t valueData[1] = {1};
    auto valueTensor = nodeDef->AddInputs();
    EXPECT_NE(valueTensor, nullptr);

    auto valueShape = valueTensor->GetTensorShape();
    std::vector<int64_t> valueDims = {1};
    valueShape->SetDimSizes(valueDims);
    valueTensor->SetDataType(DT_UINT8);
    valueTensor->SetData(valueData);
    valueTensor->SetDataSize(sizeof(uint8_t));

   //set default value
    uint8_t DValueData[1] = {5};
    auto DValueTensor = nodeDef->AddInputs();
    EXPECT_NE(DValueTensor, nullptr);

    auto DValueShape = DValueTensor->GetTensorShape();
    std::vector<int64_t> DValueDims = {};
    DValueShape->SetDimSizes(DValueDims);
    DValueTensor->SetDataType(DT_UINT8);
    DValueTensor->SetData(DValueData);
    DValueTensor->SetDataSize(sizeof(uint8_t));

    const int32_t ind = 8;
    uint8_t output[ind] = {0};

    //set output
    auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);
    auto outputShape = outputTensor->GetTensorShape();
    std::vector<int64_t> outputShapeDims = {2, 2, 2};
    outputShape->SetDimSizes(outputShapeDims);

    outputTensor->SetDataType(DT_UINT8);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(ind * sizeof(uint8_t));

    auto validIndice = CpuKernelUtils::CreateAttrValue();
    validIndice->SetBool(true);
    nodeDef->AddAttrs("validate_indices", validIndice.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}
