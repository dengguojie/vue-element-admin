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

namespace fe {

class same_input_conv2d_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "same input conv2d fusion SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "same input conv2d fusion TearDown" << std::endl;
    }

    /***************************************
    *
    *          x          
    *       /     \       
    *   conv2d0  conv2d1  
    *      |        |     
    *    relu      relu   
    *      |        |     
    *   conv2d2   conv2d3 
    *
    ****************************************/
    void CheckNodesCnt(ComputeGraphPtr &computeGraph, uint32_t conv2dNodeExpect,
        uint32_t splitNodeExpect, uint32_t constNodeExpect, uint32_t reluNodeExpect, uint32_t concatNodeExpect)
    {
        uint32_t conv2dNodeCnt = 0;
        uint32_t splitNodeCnt = 0;
        uint32_t constNodeCnt = 0;
        uint32_t reluNodeCnt = 0;
        uint32_t concatNodeCnt = 0;
        for (auto node: computeGraph->GetAllNodes()) {
            if (node->GetType() == "SplitV") {
                splitNodeCnt++;
            }
            if (node->GetType() == "Conv2D") {
                conv2dNodeCnt++;
            }
            if (node->GetType() == "Const") {
                constNodeCnt++;
            }
            if (node->GetType() == "Relu") {
                reluNodeCnt++;
            }
            if (node->GetType() == "ConcatV2") {
                concatNodeCnt++;
            }
        }
        EXPECT_EQ(splitNodeCnt, splitNodeExpect);
        EXPECT_EQ(conv2dNodeCnt, conv2dNodeExpect);
        EXPECT_EQ(concatNodeCnt, concatNodeExpect);
    }

    ge::op::Data BuildInputOp(std::string opName, const std::vector<int64_t>& inputDims, Format format, DataType dataType)
    {
        auto inputOp = op::Data(opName.c_str());
        ge::Shape inputShape(inputDims);
        ge::TensorDesc inputTensorDesc(inputShape, format, dataType);
        inputOp.update_input_desc_x(inputTensorDesc);
        inputOp.update_output_desc_y(inputTensorDesc);

        return inputOp;
    }

    ge::op::Conv2D BuildConv2dOp(std::string opName, ge::Operator& inputOp, ge::op::Const& filterOp, ge::op::Const& biasOp,
        bool hasBias, std::string format, const std::vector<int64_t>& strides, const std::vector<int64_t>& pads)
    {
        auto conv2dOp = op::Conv2D(opName.c_str());
        conv2dOp.set_input_x(inputOp);
        conv2dOp.set_input_filter(filterOp);
        if (hasBias) {
            conv2dOp.set_input_bias(biasOp);
        }
        conv2dOp.set_attr_data_format(format.c_str());
        conv2dOp.set_attr_strides(strides);
        conv2dOp.set_attr_pads(pads);

        return conv2dOp;
    }

    ge::op::Const BuildFilter(std::string opName, const std::vector<float>& filterValue, const std::vector<int64_t>& filterDims,
        Format format, DataType dataType)
    {
        Tensor filterTensor;
        std::vector<float> filterTensorVaule(filterValue);
        filterTensor.SetData(reinterpret_cast<uint8_t*>(filterTensorVaule.data()), filterTensorVaule.size() * sizeof(float));

        ge::Shape filterShape(filterDims);
        ge::TensorDesc filterTensorDesc(filterShape, format, dataType);
        filterTensor.SetTensorDesc(filterTensorDesc);

        auto filterOp = op::Const(opName.c_str());
        filterOp.set_attr_value(filterTensor);
        filterOp.update_output_desc_y(filterTensorDesc);

        return filterOp;
    }

    ge::op::Const BuildBias(std::string opName, const std::vector<float>& biasValue, const std::vector<int64_t>& biasDims,
        Format format, DataType dataType)
    {
        Tensor biasTensor;
        std::vector<float> biasTensorVaule(biasValue);
        biasTensor.SetData(reinterpret_cast<uint8_t*>(biasTensorVaule.data()), biasTensorVaule.size() * sizeof(float));

        ge::Shape biasShape(biasDims);
        ge::TensorDesc biasTensorDesc(biasShape, format, dataType);
        biasTensor.SetTensorDesc(biasTensorDesc);

        auto biasOp = op::Const(opName.c_str());
        biasOp.set_attr_value(biasTensor);
        biasOp.update_output_desc_y(biasTensorDesc);

        return biasOp;
    }

    ge::op::Relu BuildRelu(std::string opName, ge::Operator& inputOp)
    {
        auto reluOp = op::Relu(opName.c_str());
        reluOp.set_input_x(inputOp);

        return reluOp;
    }

    ge::op::AscendQuant BuildQuant(std::string opName, ge::Operator& inputOp)
    {
        float scale = 1.234;
        float offset = 0.0;
        auto quantOp = op::AscendQuant(opName.c_str());
        quantOp.set_input_x(inputOp)
               .set_attr_scale(scale)
               .set_attr_offset(offset);

        return quantOp;
    }

    ge::op::AscendDequant BuildDequant(std::string opName, ge::Operator& inputOp)
    {
        ge::Shape shape({32});
        ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT);

        float deqScale = 1.0;
        ge::Tensor scaleTensor(tensorDesc, reinterpret_cast<uint8_t*>(&deqScale), sizeof(float));
        auto constName = opName + "deq_scale";
        auto constOp = op::Const(constName.c_str()).set_attr_value(scaleTensor);

        auto dequantOp = op::AscendDequant(opName.c_str());
        dequantOp.set_input_x(inputOp)
                 .set_input_deq_scale(constOp)
                 .set_attr_dtype(DT_FLOAT);

        return dequantOp;
    }

    ge::op::AscendRequant BuildRequant(std::string opName, ge::Operator& inputOp)
    {
        ge::Shape shape({32});
        ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT);

        float deqScale = 1.0;
        ge::Tensor scaleTensor(tensorDesc, reinterpret_cast<uint8_t*>(&deqScale), sizeof(float));
        auto constName = opName + "req_scale";
        auto constOp = op::Const(constName.c_str()).set_attr_value(scaleTensor);
        auto requantOp = op::AscendRequant(opName.c_str());
        requantOp.set_input_x(inputOp)
                 .set_input_req_scale(constOp);

        return requantOp;
    }

    void BuildGraph(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildInputOp("input", {1, 32, 32, 32}, FORMAT_NHWC, DT_FLOAT);

        auto filterOp0 = BuildFilter("filter_0", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp0 = BuildBias("bias_0", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp0 = BuildConv2dOp("conv_0", inputOp, filterOp0, biasOp0, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp0 = BuildRelu("relu_0", conv2dOp0);

        auto filterOp1 = BuildFilter("filter_1", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp1 = BuildBias("bias_1", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp1 = BuildConv2dOp("conv_1", inputOp, filterOp1, biasOp1, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp1 = BuildRelu("relu_1", conv2dOp1);

        auto filterOp2 = BuildFilter("filter_2", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp2 = BuildBias("bias_2", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp2 = BuildConv2dOp("conv_2", reluOp0, filterOp2, biasOp2, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        auto filterOp3 = BuildFilter("filter_3", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp3 = BuildBias("bias_3", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp3 = BuildConv2dOp("conv_3", reluOp1, filterOp3, biasOp3, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        // create graph
        std::vector<Operator> inputs {inputOp, filterOp0, biasOp0, filterOp1, biasOp1, filterOp2, biasOp2, filterOp3, biasOp3};
        std::vector<Operator> outputs {conv2dOp2, conv2dOp3};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    void BuildGraphWithNoBias(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildInputOp("input", {1, 32, 32, 32}, FORMAT_NHWC, DT_FLOAT);

        auto filterOp0 = BuildFilter("filter_0", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp0 = BuildBias("bias_0", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp0 = BuildConv2dOp("conv_0", inputOp, filterOp0, biasOp0, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp0 = BuildRelu("relu_0", conv2dOp0);

        auto filterOp1 = BuildFilter("filter_1", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp1 = BuildBias("bias_1", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp1 = BuildConv2dOp("conv_1", inputOp, filterOp1, biasOp1, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp1 = BuildRelu("relu_1", conv2dOp1);

        auto filterOp2 = BuildFilter("filter_2", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp2 = BuildBias("bias_2", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp2 = BuildConv2dOp("conv_2", reluOp0, filterOp2, biasOp2, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        auto filterOp3 = BuildFilter("filter_3", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp3 = BuildBias("bias_3", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp3 = BuildConv2dOp("conv_3", reluOp1, filterOp3, biasOp3, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        // create graph
        std::vector<Operator> inputs {inputOp, filterOp0, filterOp1, filterOp2, filterOp3};
        std::vector<Operator> outputs {conv2dOp2, conv2dOp3};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    void BuildGraphWithOneBias(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildInputOp("input", {1, 32, 32, 32}, FORMAT_NHWC, DT_FLOAT);

        auto filterOp0 = BuildFilter("filter_0", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp0 = BuildBias("bias_0", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp0 = BuildConv2dOp("conv_0", inputOp, filterOp0, biasOp0, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp0 = BuildRelu("relu_0", conv2dOp0);

        auto filterOp1 = BuildFilter("filter_1", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp1 = BuildBias("bias_1", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp1 = BuildConv2dOp("conv_1", inputOp, filterOp1, biasOp1, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp1 = BuildRelu("relu_1", conv2dOp1);

        auto filterOp2 = BuildFilter("filter_2", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp2 = BuildBias("bias_2", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp2 = BuildConv2dOp("conv_2", reluOp0, filterOp2, biasOp2, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        auto filterOp3 = BuildFilter("filter_3", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp3 = BuildBias("bias_3", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp3 = BuildConv2dOp("conv_3", reluOp1, filterOp3, biasOp3, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        // create graph
        std::vector<Operator> inputs {inputOp, filterOp0, filterOp1, filterOp2, filterOp3, biasOp1};
        std::vector<Operator> outputs {conv2dOp2, conv2dOp3};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    void BuildGraphWithCommonBias(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildInputOp("input", {1, 32, 32, 32}, FORMAT_NHWC, DT_FLOAT);

        auto filterOp0 = BuildFilter("filter_0", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp0 = BuildBias("bias_0", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp0 = BuildConv2dOp("conv_0", inputOp, filterOp0, biasOp0, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp0 = BuildRelu("relu_0", conv2dOp0);

        auto filterOp1 = BuildFilter("filter_1", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto conv2dOp1 = BuildConv2dOp("conv_1", inputOp, filterOp1, biasOp0, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp1 = BuildRelu("relu_1", conv2dOp1);

        auto filterOp2 = BuildFilter("filter_2", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp2 = BuildBias("bias_2", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp2 = BuildConv2dOp("conv_2", reluOp0, filterOp2, biasOp2, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        auto filterOp3 = BuildFilter("filter_3",vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp3 = BuildBias("bias_3", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp3 = BuildConv2dOp("conv_3", reluOp1, filterOp3, biasOp3, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        // create graph
        std::vector<Operator> inputs {inputOp, filterOp0, filterOp1, filterOp2, filterOp3, biasOp0};
        std::vector<Operator> outputs {conv2dOp2, conv2dOp3};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    void BuildGraphWithCommonFilter(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildInputOp("input", {1, 32, 32, 32}, FORMAT_NHWC, DT_FLOAT);

        auto filterOp0 = BuildFilter("filter_0", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp0 = BuildBias("bias_0", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp0 = BuildConv2dOp("conv_0", inputOp, filterOp0, biasOp0, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp0 = BuildRelu("relu_0", conv2dOp0);

        // same filter with conv2d 0
        auto biasOp1 = BuildBias("bias_1", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp1 = BuildConv2dOp("conv_1", inputOp, filterOp0, biasOp1, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp1 = BuildRelu("relu_1", conv2dOp1);

        auto filterOp2 = BuildFilter("filter_2", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp2 = BuildBias("bias_2", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp2 = BuildConv2dOp("conv_2", reluOp0, filterOp2, biasOp2, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        auto filterOp3 = BuildFilter("filter_3", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp3 = BuildBias("bias_3", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp3 = BuildConv2dOp("conv_3", reluOp1, filterOp3, biasOp3, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        // create graph
        std::vector<Operator> inputs {inputOp, filterOp0, filterOp2, filterOp3, biasOp0, biasOp1};
        std::vector<Operator> outputs {conv2dOp2, conv2dOp3};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    void BuildGraphWithCommonFilterBias(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildInputOp("input", {1, 32, 32, 32}, FORMAT_NHWC, DT_FLOAT);

        auto filterOp0 = BuildFilter("filter_0", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp0 = BuildBias("bias_0", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp0 = BuildConv2dOp("conv_0", inputOp, filterOp0, biasOp0, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp0 = BuildRelu("relu_0", conv2dOp0);

        // same filter with conv2d 0
        // same bias with conv2d 0
        auto conv2dOp1 = BuildConv2dOp("conv_1", inputOp, filterOp0, biasOp0, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp1 = BuildRelu("relu_1", conv2dOp1);

        auto filterOp2 = BuildFilter("filter_2", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp2 = BuildBias("bias_2", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp2 = BuildConv2dOp("conv_2", reluOp0, filterOp2, biasOp2, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        auto filterOp3 = BuildFilter("filter_3", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp3 = BuildBias("bias_3", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp3 = BuildConv2dOp("conv_3", reluOp1, filterOp3, biasOp3, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        // create graph
        std::vector<Operator> inputs {inputOp, filterOp0, filterOp2, filterOp3, biasOp0};
        std::vector<Operator> outputs {conv2dOp2, conv2dOp3};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    void BuildGraphWithBiasLinkOtherNode(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildInputOp("input", {1, 32, 32, 32}, FORMAT_NHWC, DT_FLOAT);

        auto filterOp0 = BuildFilter("filter_0", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp0 = BuildBias("bias_0", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp0 = BuildConv2dOp("conv_0", inputOp, filterOp0, biasOp0, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp0 = BuildRelu("relu_0", conv2dOp0);

        auto filterOp1 = BuildFilter("filter_1", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp1 = BuildBias("bias_1", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp1 = BuildConv2dOp("conv_1", inputOp, filterOp1, biasOp1, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp1 = BuildRelu("relu_1", conv2dOp1);

        // other node relu
        auto reluOp2 = BuildRelu("relu_2", biasOp1);

        auto filterOp2 = BuildFilter("filter_2", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp2 = BuildBias("bias_2", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp2 = BuildConv2dOp("conv_2", reluOp0, filterOp2, biasOp2, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        auto filterOp3 = BuildFilter("filter_3", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp3 = BuildBias("bias_3", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp3 = BuildConv2dOp("conv_3", reluOp1, filterOp3, biasOp3, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        // create graph
        std::vector<Operator> inputs {inputOp, filterOp0, filterOp1, filterOp2, filterOp3, biasOp0, biasOp1};
        std::vector<Operator> outputs {conv2dOp2, conv2dOp3};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    void BuildGraphWithFmapLinkOtherNode(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildInputOp("input", {1, 32, 32, 32}, FORMAT_NHWC, DT_FLOAT);

        auto filterOp0 = BuildFilter("filter_0", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp0 = BuildBias("bias_0", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp0 = BuildConv2dOp("conv_0", inputOp, filterOp0, biasOp0, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp0 = BuildRelu("relu_0", conv2dOp0);

        auto filterOp1 = BuildFilter("filter_1", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp1 = BuildBias("bias_1", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp1 = BuildConv2dOp("conv_1", inputOp, filterOp1, biasOp1, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp1 = BuildRelu("relu_1", conv2dOp1);

        // other node relu
        auto reluOp2 = BuildRelu("relu_2", inputOp);

        auto filterOp2 = BuildFilter("filter_2", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp2 = BuildBias("bias_2", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp2 = BuildConv2dOp("conv_2", reluOp0, filterOp2, biasOp2, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        auto filterOp3 = BuildFilter("filter_3", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp3 = BuildBias("bias_3", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp3 = BuildConv2dOp("conv_3", reluOp1, filterOp3, biasOp3, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        // create graph
        std::vector<Operator> inputs {inputOp, filterOp0, filterOp1, filterOp2, filterOp3, biasOp0, biasOp1};
        std::vector<Operator> outputs {conv2dOp2, conv2dOp3};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    void BuildGraphWithDiffConv2dFormat(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildInputOp("input", {1, 32, 32, 32}, FORMAT_NHWC, DT_FLOAT);

        auto filterOp0 = BuildFilter("filter_0", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp0 = BuildBias("bias_0", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp0 = BuildConv2dOp("conv_0", inputOp, filterOp0, biasOp0, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp0 = BuildRelu("relu_0", conv2dOp0);

        // different data format for conv2d
        auto filterOp1 = BuildFilter("filter_1", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp1 = BuildBias("bias_1", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp1 = BuildConv2dOp("conv_1", inputOp, filterOp1, biasOp1, false, "NCHW", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp1 = BuildRelu("relu_1", conv2dOp1);

        auto filterOp2 = BuildFilter("filter_2", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp2 = BuildBias("bias_2", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp2 = BuildConv2dOp("conv_2", reluOp0, filterOp2, biasOp2, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        auto filterOp3 = BuildFilter("filter_3", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp3 = BuildBias("bias_3", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp3 = BuildConv2dOp("conv_3", reluOp1, filterOp3, biasOp3, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        // create graph
        std::vector<Operator> inputs {inputOp, filterOp0, filterOp1, filterOp2, filterOp3};
        std::vector<Operator> outputs {conv2dOp2, conv2dOp3};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    void BuildGraphWithDiffConv2dStrides(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildInputOp("input", {1, 32, 32, 32}, FORMAT_NHWC, DT_FLOAT);

        auto filterOp0 = BuildFilter("filter_0", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp0 = BuildBias("bias_0", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp0 = BuildConv2dOp("conv_0", inputOp, filterOp0, biasOp0, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp0 = BuildRelu("relu_0", conv2dOp0);

        // different strides for conv2d
        auto filterOp1 = BuildFilter("filter_1", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp1 = BuildBias("bias_1", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp1 = BuildConv2dOp("conv_1", inputOp, filterOp1, biasOp1, true, "NHWC", {1, 1, 2, 1}, {0, 0, 0, 0});
        auto reluOp1 = BuildRelu("relu_1", conv2dOp1);

        auto filterOp2 = BuildFilter("filter_2", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp2 = BuildBias("bias_2", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp2 = BuildConv2dOp("conv_2", reluOp0, filterOp2, biasOp2, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        auto filterOp3 = BuildFilter("filter_3", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp3 = BuildBias("bias_3", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp3 = BuildConv2dOp("conv_3", reluOp1, filterOp3, biasOp3, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        // create graph
        std::vector<Operator> inputs {inputOp, filterOp0, filterOp1, filterOp2, filterOp3, biasOp0, biasOp1};
        std::vector<Operator> outputs {conv2dOp2, conv2dOp3};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    void BuildGraphWithDiffConv2dPads(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildInputOp("input", {1, 32, 32, 32}, FORMAT_NHWC, DT_FLOAT);

        auto filterOp0 = BuildFilter("filter_0", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp0 = BuildBias("bias_0", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp0 = BuildConv2dOp("conv_0", inputOp, filterOp0, biasOp0, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp0 = BuildRelu("relu_0", conv2dOp0);

        // different pads for conv2d
        auto filterOp1 = BuildFilter("filter_1", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp1 = BuildBias("bias_1", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp1 = BuildConv2dOp("conv_1", inputOp, filterOp1, biasOp1, true, "NHWC", {1, 1, 1, 1}, {0, 1, 0, 0});
        auto reluOp1 = BuildRelu("relu_1", conv2dOp1);

        auto filterOp2 = BuildFilter("filter_2", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp2 = BuildBias("bias_2", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp2 = BuildConv2dOp("conv_2", reluOp0, filterOp2, biasOp2, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        auto filterOp3 = BuildFilter("filter_3", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp3 = BuildBias("bias_3", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp3 = BuildConv2dOp("conv_3", reluOp1, filterOp3, biasOp3, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        // create graph
        std::vector<Operator> inputs {inputOp, filterOp0, filterOp1, filterOp2, filterOp3, biasOp0, biasOp1};
        std::vector<Operator> outputs {conv2dOp2, conv2dOp3};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    // conv format 
    void BuildGraphWithDiffFilterFormat(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildInputOp("input", {1, 32, 32, 32}, FORMAT_NCHW, DT_FLOAT);

        auto filterOp0 = BuildFilter("filter_0", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp0 = BuildBias("bias_0", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp0 = BuildConv2dOp("conv_0", inputOp, filterOp0, biasOp0, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp0 = BuildRelu("relu_0", conv2dOp0);

        auto filterOp1 = BuildFilter("filter_1", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp1 = BuildBias("bias_1", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp1 = BuildConv2dOp("conv_1", inputOp, filterOp1, biasOp1, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp1 = BuildRelu("relu_1", conv2dOp1);

        auto filterOp2 = BuildFilter("filter_2", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp2 = BuildBias("bias_2", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp2 = BuildConv2dOp("conv_2", reluOp0, filterOp2, biasOp2, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        auto filterOp3 = BuildFilter("filter_3", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp3 = BuildBias("bias_3", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp3 = BuildConv2dOp("conv_3", reluOp1, filterOp3, biasOp3, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        // create graph
        std::vector<Operator> inputs {inputOp, filterOp0, filterOp1, filterOp2, filterOp3};
        std::vector<Operator> outputs {conv2dOp2, conv2dOp3};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    void BuildGraphWithDiffConvOffsetW(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildInputOp("input", {1, 32, 32, 32}, FORMAT_NHWC, DT_FLOAT);

        auto filterOp0 = BuildFilter("filter_0", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp0 = BuildBias("bias_0", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp0 = BuildConv2dOp("conv_0", inputOp, filterOp0, biasOp0, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp0 = BuildRelu("relu_0", conv2dOp0);

        // input offset_w is not supported
        // conv2d 0 has offset w
        auto filterOp1 = BuildFilter("filter_1", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp1 = BuildBias("bias_1", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp1 = BuildConv2dOp("conv_1", inputOp, filterOp1, biasOp1, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp1 = BuildRelu("relu_1", conv2dOp1);

        auto filterOp2 = BuildFilter("filter_2", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp2 = BuildBias("bias_2", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp2 = BuildConv2dOp("conv_2", reluOp0, filterOp2, biasOp2, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        auto filterOp3 = BuildFilter("filter_3", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp3 = BuildBias("bias_3", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp3 = BuildConv2dOp("conv_3", reluOp1, filterOp3, biasOp3, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        // create graph
        std::vector<Operator> inputs {inputOp, filterOp0, filterOp1, filterOp2, filterOp3, biasOp0, biasOp1};
        std::vector<Operator> outputs {conv2dOp2, conv2dOp3};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    void BuildGraphInputConst(ComputeGraphPtr &computeGraph)
    {
        auto inputOp0 = BuildInputOp("input_0", {1, 32, 32, 32}, FORMAT_NCHW, DT_FLOAT);
        auto inputOp1 = BuildInputOp("input_1", {1, 32, 32, 32}, FORMAT_NCHW, DT_FLOAT);

        auto filterOp0 = BuildFilter("filter_0", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp0 = BuildBias("bias_0", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp0 = BuildConv2dOp("conv_0", inputOp0, filterOp0, biasOp0, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp0 = BuildRelu("relu_0", conv2dOp0);

        auto filterOp1 = BuildFilter("filter_1", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto conv2dOp1 = BuildConv2dOp("conv_1", inputOp1, filterOp1, biasOp0, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp1 = BuildRelu("relu_1", conv2dOp1);

        auto filterOp2 = BuildFilter("filter_2", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp2 = BuildBias("bias_2", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp2 = BuildConv2dOp("conv_2", reluOp0, filterOp2, biasOp2, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        auto filterOp3 = BuildFilter("filter_3", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp3 = BuildBias("bias_3", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp3 = BuildConv2dOp("conv_3", reluOp1, filterOp3, biasOp3, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        // create graph
        std::vector<Operator> inputs {inputOp0, inputOp1, filterOp0, filterOp1, filterOp2, filterOp3, biasOp0};
        std::vector<Operator> outputs {conv2dOp2, conv2dOp3};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    void BuildGraphWithMultiConv(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildInputOp("input", {1, 32, 32, 32}, FORMAT_NHWC, DT_FLOAT);

        auto filterOp0 = BuildFilter("filter_0", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp0 = BuildBias("bias_0", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp0 = BuildConv2dOp("conv_0", inputOp, filterOp0, biasOp0, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp0 = BuildRelu("relu_0", conv2dOp0);

        auto filterOp1 = BuildFilter("filter_1", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp1 = BuildBias("bias_1", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp1 = BuildConv2dOp("conv_1", inputOp, filterOp1, biasOp1, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp1 = BuildRelu("relu_1", conv2dOp1);

        auto filterOpMulti = BuildFilter("filter_multi", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOpMulti = BuildBias("bias_multi", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOpMulti = BuildConv2dOp("conv_multi", inputOp, filterOpMulti, biasOpMulti, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOpMulti = BuildRelu("relu_multi", conv2dOpMulti);

        auto filterOp2 = BuildFilter("filter_2", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp2 = BuildBias("bias_2", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp2 = BuildConv2dOp("conv_2", reluOp0, filterOp2, biasOp2, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        auto filterOp3 = BuildFilter("filter_3", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp3 = BuildBias("bias_3", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp3 = BuildConv2dOp("conv_3", reluOp1, filterOp3, biasOp3, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        auto filterOp4 = BuildFilter("filter_4", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp4 = BuildBias("bias_4", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp4 = BuildConv2dOp("conv_4", reluOpMulti, filterOp4, biasOp4, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        // create graph
        std::vector<Operator> inputs {inputOp, filterOp0, filterOp1, filterOp2, filterOp3, filterOp4, filterOpMulti, biasOp0, biasOp1, biasOpMulti};
        std::vector<Operator> outputs {conv2dOp2, conv2dOp3, conv2dOp4};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    void BuildGraphWithMultiConvPartSame(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildInputOp("input", {1, 32, 32, 32}, FORMAT_NHWC, DT_FLOAT);

        auto filterOp0 = BuildFilter("filter_0", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp0 = BuildBias("bias_0", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp0 = BuildConv2dOp("conv_0", inputOp, filterOp0, biasOp0, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp0 = BuildRelu("relu_0", conv2dOp0);

        auto filterOp1 = BuildFilter("filter_1", vector<float>(32, 0.1), {32, 2, 2, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp1 = BuildBias("bias_1", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp1 = BuildConv2dOp("conv_1", inputOp, filterOp1, biasOp1, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp1 = BuildRelu("relu_1", conv2dOp1);

        auto filterOpMulti = BuildFilter("filter_multi", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOpMulti = BuildBias("bias_multi", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOpMulti = BuildConv2dOp("conv_multi", inputOp, filterOpMulti, biasOpMulti, true, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOpMulti = BuildRelu("relu_multi", conv2dOpMulti);

        auto filterOp2 = BuildFilter("filter_2", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp2 = BuildBias("bias_2", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp2 = BuildConv2dOp("conv_2", reluOp0, filterOp2, biasOp2, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        auto filterOp3 = BuildFilter("filter_3", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp3 = BuildBias("bias_3", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp3 = BuildConv2dOp("conv_3", reluOp1, filterOp3, biasOp3, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        auto filterOp4 = BuildFilter("filter_4", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp4 = BuildBias("bias_4", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp4 = BuildConv2dOp("conv_4", reluOpMulti, filterOp4, biasOp4, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        // create graph
        std::vector<Operator> inputs {inputOp, filterOp0, filterOp1, filterOp2, filterOp3, filterOp4, filterOpMulti, biasOp0, biasOp1, biasOpMulti};
        std::vector<Operator> outputs {conv2dOp2, conv2dOp3, conv2dOp4};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    void BuildGraphWithDiffKernel(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildInputOp("input", {1, 32, 32, 32}, FORMAT_NHWC, DT_FLOAT);

        auto filterOp0 = BuildFilter("filter_0", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp0 = BuildBias("bias_0", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp0 = BuildConv2dOp("conv_0", inputOp, filterOp0, biasOp0, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp0 = BuildRelu("relu_0", conv2dOp0);

        auto filterOp1 = BuildFilter("filter_1", vector<float>(32, 0.1), {32, 2, 2, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp1 = BuildBias("bias_1", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp1 = BuildConv2dOp("conv_1", inputOp, filterOp1, biasOp1, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto reluOp1 = BuildRelu("relu_1", conv2dOp1);

        auto filterOp2 = BuildFilter("filter_2", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp2 = BuildBias("bias_2", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp2 = BuildConv2dOp("conv_2", reluOp0, filterOp2, biasOp2, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        auto filterOp3 = BuildFilter("filter_3", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp3 = BuildBias("bias_3", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp3 = BuildConv2dOp("conv_3", reluOp1, filterOp3, biasOp3, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        // create graph
        std::vector<Operator> inputs {inputOp, filterOp0, filterOp1, filterOp2, filterOp3};
        std::vector<Operator> outputs {conv2dOp2, conv2dOp3};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    void BuildGraphRequant(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildInputOp("input", {1, 32, 32, 32}, FORMAT_NHWC, DT_FLOAT);

        auto filterOp0 = BuildFilter("filter_0", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp0 = BuildBias("bias_0", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp0 = BuildConv2dOp("conv_0", inputOp, filterOp0, biasOp0, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto requantOp0 = BuildRequant("requant_0", conv2dOp0);

        auto filterOp1 = BuildFilter("filter_1", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp1 = BuildBias("bias_1", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp1 = BuildConv2dOp("conv_1", inputOp, filterOp1, biasOp1, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto requantOp1 = BuildRequant("requant_1", conv2dOp1);

        auto filterOp2 = BuildFilter("filter_2", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp2 = BuildBias("bias_2", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp2 = BuildConv2dOp("conv_2", requantOp0, filterOp2, biasOp2, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        auto filterOp3 = BuildFilter("filter_3", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp3 = BuildBias("bias_3", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp3 = BuildConv2dOp("conv_3", requantOp1, filterOp3, biasOp3, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        // create graph
        std::vector<Operator> inputs {inputOp, filterOp0, filterOp1, filterOp2, filterOp3};
        std::vector<Operator> outputs {conv2dOp2, conv2dOp3};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    void BuildGraphQuant(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildInputOp("input", {1, 32, 32, 32}, FORMAT_NHWC, DT_FLOAT);

        auto filterOp0 = BuildFilter("filter_0", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp0 = BuildBias("bias_0", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp0 = BuildConv2dOp("conv_0", inputOp, filterOp0, biasOp0, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto dequantOp0 = BuildDequant("dequant_0", conv2dOp0);
        auto quantOp0 = BuildQuant("quant_0", dequantOp0);

        auto filterOp1 = BuildFilter("filter_1", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp1 = BuildBias("bias_1", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp1 = BuildConv2dOp("conv_1", inputOp, filterOp1, biasOp1, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto dequantOp1 = BuildDequant("dequant_1", conv2dOp1);
        auto quantOp1 = BuildQuant("quant_1", dequantOp1);

        auto filterOp2 = BuildFilter("filter_2", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp2 = BuildBias("bias_2", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp2 = BuildConv2dOp("conv_2", quantOp0, filterOp2, biasOp2, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        auto filterOp3 = BuildFilter("filter_3", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp3 = BuildBias("bias_3", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp3 = BuildConv2dOp("conv_3", quantOp1, filterOp3, biasOp3, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});

        // create graph
        std::vector<Operator> inputs {inputOp, filterOp0, filterOp1, filterOp2, filterOp3};
        std::vector<Operator> outputs {conv2dOp2, conv2dOp3};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    void BuildGraphQuantNext(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildInputOp("input", {1, 32, 32, 32}, FORMAT_NHWC, DT_FLOAT);

        auto filterOp0 = BuildFilter("filter_0", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp0 = BuildBias("bias_0", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp0 = BuildConv2dOp("conv_0", inputOp, filterOp0, biasOp0, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto dequantOp0 = BuildDequant("dequant_0", conv2dOp0);
        auto quantOp0 = BuildQuant("quant_0", dequantOp0);

        auto filterOp1 = BuildFilter("filter_1", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp1 = BuildBias("bias_1", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp1 = BuildConv2dOp("conv_1", inputOp, filterOp1, biasOp1, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto dequantOp1 = BuildDequant("dequant_1", conv2dOp1);
        auto quantOp1 = BuildQuant("quant_1", dequantOp1);

        auto filterOp2 = BuildFilter("filter_2", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp2 = BuildBias("bias_2", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp2 = BuildConv2dOp("conv_2", quantOp0, filterOp2, biasOp2, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto dequantOp2 = BuildDequant("dequant_2", conv2dOp2);
        auto quantOp2 = BuildQuant("quant_2", dequantOp2);

        auto filterOp3 = BuildFilter("filter_3", vector<float>(32, 0.1), {32, 1, 1, 1}, FORMAT_NHWC, DT_FLOAT);
        auto biasOp3 = BuildBias("bias_3", vector<float>(32, 0.1), {32}, FORMAT_ND, DT_FLOAT);
        auto conv2dOp3 = BuildConv2dOp("conv_3", quantOp1, filterOp3, biasOp3, false, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0});
        auto dequantOp3 = BuildDequant("dequant_3", conv2dOp3);
        auto quantOp3 = BuildQuant("quant_3", dequantOp3);

        // create graph
        std::vector<Operator> inputs {inputOp, filterOp0, filterOp1, filterOp2, filterOp3};
        std::vector<Operator> outputs {quantOp2, quantOp3};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }
};

TEST_F(same_input_conv2d_test, same_input_conv2d_smoke_test_01)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraph(computeGraph);

    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 1;
    opti_compilation_info.soc_version = "Ascend910A";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910A"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SameInputConv2dPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);

    uint32_t conv2dNodeExpect = 3;
    uint32_t splitNodeExpect = 1;
    uint32_t constNodeExpect = 8;
    uint32_t reluNodeExpect = 2;
    uint32_t concatNodeExpect = 2;
    CheckNodesCnt(computeGraph, conv2dNodeExpect, splitNodeExpect, constNodeExpect, reluNodeExpect, concatNodeExpect);

    for (auto node: computeGraph->GetAllNodes()) {
        cout << node->GetName();
        cout << " " << node->GetInDataNodes().size();
        cout << " " << node->GetOutDataNodes().size() << endl;
        if (node->GetType() == "SplitV") {
            auto weights = ge::OpDescUtils::GetWeights(node);
            cout << "data type " << weights[0]->GetTensorDesc().GetDataType() << endl;
            
            int32_t* sizeSplit = (int32_t*)weights[0]->GetData().GetData();
            cout << "size split " << *sizeSplit << " " << *(sizeSplit + 1) << endl;

            int32_t* dimSplit = (int32_t*)weights[1]->GetData().GetData();
            cout << "dim split " << *dimSplit << endl;
        }
        if (node->GetType() == "ConcatV2") {
            auto weights = ge::OpDescUtils::GetWeights(node);
            cout << "concat data type " << weights[2]->GetTensorDesc().GetDataType() << endl;
            
            int32_t* value = (int32_t*)weights[2]->GetData().GetData();
            cout << "concat data size " << weights[2]->GetData().GetSize() << endl;
            cout << "concat  " << *value << endl;
        }
    }
}

TEST_F(same_input_conv2d_test, same_input_conv2d_with_no_bias_test_02)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphWithNoBias(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SameInputConv2dPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);

    uint32_t conv2dNodeExpect = 3;
    uint32_t splitNodeExpect = 1;
    uint32_t constNodeExpect = 5;
    uint32_t reluNodeExpect = 2;
    uint32_t concatNodeExpect = 1;
    CheckNodesCnt(computeGraph, conv2dNodeExpect, splitNodeExpect, constNodeExpect, reluNodeExpect, concatNodeExpect);
}

TEST_F(same_input_conv2d_test, same_input_conv2d_with_one_bias_test_03)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphWithOneBias(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SameInputConv2dPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);

    uint32_t conv2dNodeExpect = 4;
    uint32_t splitNodeExpect = 0;
    uint32_t constNodeExpect = 5;
    uint32_t reluNodeExpect = 2;
    uint32_t concatNodeExpect = 0;
    CheckNodesCnt(computeGraph, conv2dNodeExpect, splitNodeExpect, constNodeExpect, reluNodeExpect, concatNodeExpect);
}

TEST_F(same_input_conv2d_test, same_input_conv2d_with_common_bias_test_04)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphWithCommonBias(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SameInputConv2dPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);

    uint32_t conv2dNodeExpect = 3;
    uint32_t splitNodeExpect = 1;
    uint32_t constNodeExpect = 6;
    uint32_t reluNodeExpect = 2;
    uint32_t concatNodeExpect = 2;
    CheckNodesCnt(computeGraph, conv2dNodeExpect, splitNodeExpect, constNodeExpect, reluNodeExpect, concatNodeExpect);
}

TEST_F(same_input_conv2d_test, same_input_conv2d_with_common_filter_test_05)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphWithCommonFilter(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SameInputConv2dPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);

    uint32_t conv2dNodeExpect = 3;
    uint32_t splitNodeExpect = 1;
    uint32_t constNodeExpect = 6;
    uint32_t reluNodeExpect = 2;
    uint32_t concatNodeExpect = 2;
    CheckNodesCnt(computeGraph, conv2dNodeExpect, splitNodeExpect, constNodeExpect, reluNodeExpect, concatNodeExpect);
}

TEST_F(same_input_conv2d_test, same_input_conv2d_with_common_filter_bias_test_06)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphWithCommonFilterBias(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SameInputConv2dPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);

    uint32_t conv2dNodeExpect = 3;
    uint32_t splitNodeExpect = 1;
    uint32_t constNodeExpect = 6;
    uint32_t reluNodeExpect = 2;
    uint32_t concatNodeExpect = 2;
    CheckNodesCnt(computeGraph, conv2dNodeExpect, splitNodeExpect, constNodeExpect, reluNodeExpect, concatNodeExpect);
}

TEST_F(same_input_conv2d_test, same_input_conv2d_with_bias_link_other_node_test_07)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphWithBiasLinkOtherNode(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SameInputConv2dPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);
}

TEST_F(same_input_conv2d_test, same_input_conv2d_with_fmap_link_other_node_test_08)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphWithFmapLinkOtherNode(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SameInputConv2dPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);

    uint32_t conv2dNodeExpect = 3;
    uint32_t splitNodeExpect = 1;
    uint32_t constNodeExpect = 6;
    uint32_t reluNodeExpect = 3;
    uint32_t concatNodeExpect = 2;
    CheckNodesCnt(computeGraph, conv2dNodeExpect, splitNodeExpect, constNodeExpect, reluNodeExpect, concatNodeExpect);
}

TEST_F(same_input_conv2d_test, same_input_conv2d_with_diff_conv_format_test_09)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphWithDiffConv2dFormat(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SameInputConv2dPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);
}

TEST_F(same_input_conv2d_test, same_input_conv2d_with_diff_conv_strides_test_10)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphWithDiffConv2dStrides(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SameInputConv2dPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);

    uint32_t conv2dNodeExpect = 4;
    uint32_t splitNodeExpect = 0;
    uint32_t constNodeExpect = 6;
    uint32_t reluNodeExpect = 2;
    uint32_t concatNodeExpect = 0;
    CheckNodesCnt(computeGraph, conv2dNodeExpect, splitNodeExpect, constNodeExpect, reluNodeExpect, concatNodeExpect);
}

TEST_F(same_input_conv2d_test, same_input_conv2d_with_diff_conv_pads_test_11)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphWithDiffConv2dPads(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SameInputConv2dPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);

    uint32_t conv2dNodeExpect = 4;
    uint32_t splitNodeExpect = 0;
    uint32_t constNodeExpect = 6;
    uint32_t reluNodeExpect = 2;
    uint32_t concatNodeExpect = 0;
    CheckNodesCnt(computeGraph, conv2dNodeExpect, splitNodeExpect, constNodeExpect, reluNodeExpect, concatNodeExpect);
}

TEST_F(same_input_conv2d_test, same_input_conv2d_with_diff_filter_format_test_12)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphWithDiffFilterFormat(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SameInputConv2dPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);
}

// input offset_w is not supported
TEST_F(same_input_conv2d_test, same_input_conv2d_with_diff_conv_offset_w_test_13)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphWithDiffConvOffsetW(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SameInputConv2dPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);
}

TEST_F(same_input_conv2d_test, same_input_conv2d_with_multi_conv_test_14)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphWithMultiConv(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SameInputConv2dPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);

    uint32_t conv2dNodeExpect = 4;
    uint32_t splitNodeExpect = 1;
    uint32_t constNodeExpect = 7;
    uint32_t reluNodeExpect = 3;
    uint32_t concatNodeExpect = 2;
    CheckNodesCnt(computeGraph, conv2dNodeExpect, splitNodeExpect, constNodeExpect, reluNodeExpect, concatNodeExpect);
}

TEST_F(same_input_conv2d_test, same_input_conv2d_with_multi_conv_part_same_test_15)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphWithMultiConvPartSame(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SameInputConv2dPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);

    uint32_t conv2dNodeExpect = 5;
    uint32_t splitNodeExpect = 1;
    uint32_t constNodeExpect = 9;
    uint32_t reluNodeExpect = 3;
    uint32_t concatNodeExpect = 2;
    CheckNodesCnt(computeGraph, conv2dNodeExpect, splitNodeExpect, constNodeExpect, reluNodeExpect, concatNodeExpect);
}

TEST_F(same_input_conv2d_test, same_input_conv2d_input_const_test_15)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphInputConst(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SameInputConv2dPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);

    uint32_t conv2dNodeExpect = 4;
    uint32_t splitNodeExpect = 0;
    uint32_t constNodeExpect = 5;
    uint32_t reluNodeExpect = 2;
    uint32_t concatNodeExpect = 0;
    CheckNodesCnt(computeGraph, conv2dNodeExpect, splitNodeExpect, constNodeExpect, reluNodeExpect, concatNodeExpect);
}

TEST_F(same_input_conv2d_test, same_input_conv2d_diff_kernel_test_16)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphWithDiffKernel(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SameInputConv2dPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);

    uint32_t conv2dNodeExpect = 4;
    uint32_t splitNodeExpect = 0;
    uint32_t constNodeExpect = 4;
    uint32_t reluNodeExpect = 2;
    uint32_t concatNodeExpect = 0;
    CheckNodesCnt(computeGraph, conv2dNodeExpect, splitNodeExpect, constNodeExpect, reluNodeExpect, concatNodeExpect);
}

TEST_F(same_input_conv2d_test, same_input_conv2d_quant_test_17)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphQuant(computeGraph);
    cout << "build graph quant" << endl;
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SameInputConv2dPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);

    uint32_t conv2dNodeExpect = 3;
    uint32_t splitNodeExpect = 1;
    uint32_t constNodeExpect = 4;
    uint32_t reluNodeExpect = 0;
    uint32_t concatNodeExpect = 2;
    CheckNodesCnt(computeGraph, conv2dNodeExpect, splitNodeExpect, constNodeExpect, reluNodeExpect, concatNodeExpect);
}

TEST_F(same_input_conv2d_test, same_input_conv2d_requant_test_18)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphRequant(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SameInputConv2dPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);

    uint32_t conv2dNodeExpect = 3;
    uint32_t splitNodeExpect = 1;
    uint32_t constNodeExpect = 4;
    uint32_t reluNodeExpect = 0;
    uint32_t concatNodeExpect = 2;
    CheckNodesCnt(computeGraph, conv2dNodeExpect, splitNodeExpect, constNodeExpect, reluNodeExpect, concatNodeExpect);
}

TEST_F(same_input_conv2d_test, same_input_conv2d_quant_next_test_19)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphQuantNext(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SameInputConv2dPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);

    uint32_t conv2dNodeExpect = 3;
    uint32_t splitNodeExpect = 1;
    uint32_t constNodeExpect = 4;
    uint32_t reluNodeExpect = 0;
    uint32_t concatNodeExpect = 2;
    CheckNodesCnt(computeGraph, conv2dNodeExpect, splitNodeExpect, constNodeExpect, reluNodeExpect, concatNodeExpect);
}

TEST_F(same_input_conv2d_test, same_input_conv2d_test_sd3403)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraph(computeGraph);

    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 1;
    opti_compilation_info.soc_version = "SD3403";
    fe::PlatformInfoManager::Instance().platform_info_map_["SD3403"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SameInputConv2dPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);

    uint32_t conv2dNodeExpect = 4;
    uint32_t splitNodeExpect = 0;
    uint32_t constNodeExpect = 10;
    uint32_t reluNodeExpect = 2;
    uint32_t concatNodeExpect = 0;
    CheckNodesCnt(computeGraph, conv2dNodeExpect, splitNodeExpect, constNodeExpect, reluNodeExpect, concatNodeExpect);
}

} // namespace fe

