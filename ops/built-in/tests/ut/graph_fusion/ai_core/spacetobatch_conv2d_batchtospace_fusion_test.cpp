#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_calculation_ops.h"
#include "transformation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "fp16_t.hpp"

using namespace ge;

namespace fe {

class spacetobatch_conv2d_batchtospace_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "spacetobatch_conv2d_batchtospace SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "spacetobatch_conv2d_batchtospace TearDown" << std::endl;
    }

    /************************************
    *
    *      block  input  paddings
    *           \   |   /
    *          spacetobatch
    *               |
    *      filter   |  bias
    *            \  |  /  
    *             conv2d
    *               |
    *       block   |  crops
    *            \  |   /
    *          batchtospace
    *               |
    *            output
    *
    *************************************/
    ge::op::Data BuildDataOp(std::string opName, const std::vector<int64_t>& inputDims, Format format, DataType dataType)
    {
        auto inputOp = op::Data(opName.c_str());
        ge::Shape inputShape(inputDims);
        ge::TensorDesc inputTensorDesc(inputShape, format, dataType);
        inputOp.update_input_desc_x(inputTensorDesc);
        inputOp.update_output_desc_y(inputTensorDesc);

        return inputOp;
    }

    ge::op::Conv2D BuildConv2dOp(std::string opName, ge::Operator& inputOp, ge::op::Const& filterOp, ge::op::Const& biasOp,
        std::string format, const std::vector<int64_t>& strides, const std::vector<int64_t>& pads, const std::vector<int64_t>& dilations)
    {
        auto conv2dOp = op::Conv2D(opName.c_str());
        conv2dOp.set_input_x(inputOp);
        conv2dOp.set_input_filter(filterOp);
        conv2dOp.set_input_bias(biasOp);
        conv2dOp.set_attr_data_format(format.c_str());
        conv2dOp.set_attr_strides(strides);
        conv2dOp.set_attr_pads(pads);
        conv2dOp.set_attr_dilations(dilations);

        return conv2dOp;
    }

    ge::op::Const BuildConstOp(std::string opName, const std::vector<int64_t>& dims, uint8_t* tensorValue, uint32_t tensorLen,
        Format format, DataType dataType)
    {
        auto shape = ge::Shape(dims);
        TensorDesc constTensorDesc(shape, format, dataType);
        Tensor constTensor(constTensorDesc);
        constTensor.SetData(tensorValue, tensorLen);

        auto constOp = op::Const(opName.c_str());
        constOp.set_attr_value(constTensor);
        constOp.update_output_desc_y(constTensorDesc);

        return constOp;
    }

    ge::op::SpaceToBatchND BuildSpaceToBatchOp(std::string opName, ge::Operator& inputOp, ge::Operator& blockOp, ge::Operator& paddingsOp)
    {
        auto spacetobatchOp = op::SpaceToBatchND(opName.c_str());
        spacetobatchOp.set_input_x(inputOp);
        spacetobatchOp.set_input_block_shape(blockOp);
        spacetobatchOp.set_input_paddings(paddingsOp);

        return spacetobatchOp;
    }

    ge::op::BatchToSpaceND BuildBatchToSpaceOp(std::string opName, ge::Operator& conv2dOp, ge::Operator& blockOp, ge::Operator& cropsOp)
    {
        auto batchtospaceOp = op::BatchToSpaceND(opName.c_str());
        batchtospaceOp.set_input_x(conv2dOp);
        batchtospaceOp.set_input_block_shape(blockOp);
        batchtospaceOp.set_input_crops(cropsOp);

        return batchtospaceOp;
    }

    void BuildGraph(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildDataOp("input", {1, 32, 32, 1}, FORMAT_NHWC, DT_FLOAT);

        std::vector<uint32_t> block {2, 2};
        auto blockOp = BuildConstOp("space_block", {2}, reinterpret_cast<uint8_t*>(block.data()), block.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);

        std::vector<uint32_t> paddings {0, 0, 0, 0};
        auto paddingsOp = BuildConstOp("space_paddings", {2, 2}, reinterpret_cast<uint8_t*>(paddings.data()), paddings.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);

        auto spacetobatchOp = BuildSpaceToBatchOp("spacetobatch", inputOp, blockOp, paddingsOp);

        std::vector<float> filter {0.1};
        auto convFilterOp = BuildConstOp("conv_filter", {1, 1, 1, 1}, reinterpret_cast<uint8_t*>(filter.data()), filter.size() * sizeof(float), FORMAT_NHWC, DT_FLOAT);

        std::vector<float> bias {0.1};
        auto convBiasOp = BuildConstOp("conv_bias", {1}, reinterpret_cast<uint8_t*>(bias.data()), bias.size() * sizeof(float), FORMAT_NHWC, DT_FLOAT);
        auto conv2dOp = BuildConv2dOp("conv2d", spacetobatchOp, convFilterOp, convBiasOp, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 1, 1});

        std::vector<uint32_t> batchBlock {2, 2};
        auto batchBlockOp = BuildConstOp("batch_block", {2}, reinterpret_cast<uint8_t*>(batchBlock.data()), batchBlock.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);

        std::vector<uint32_t> crops {0, 0, 0, 0};
        auto batchCropsOp = BuildConstOp("batch_crops", {2, 2}, reinterpret_cast<uint8_t*>(crops.data()), crops.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);
        auto batchtospaceOp = BuildBatchToSpaceOp("batchtospace", conv2dOp, batchBlockOp, batchCropsOp);

        auto outputOp = BuildDataOp("output", {1, 32, 32, 1}, FORMAT_NHWC, DT_FLOAT);
        outputOp.set_input_x(batchtospaceOp);

        // create graph
        std::vector<Operator> inputs {inputOp, blockOp, paddingsOp, convFilterOp, convBiasOp, batchBlockOp, batchCropsOp};
        std::vector<Operator> outputs {outputOp};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    void BuildGraphNoPattern(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildDataOp("input", {1, 32, 32, 1}, FORMAT_NHWC, DT_FLOAT);
        auto outputOp = BuildDataOp("output", {1, 32, 32, 1}, FORMAT_NHWC, DT_FLOAT);
        outputOp.set_input_x(inputOp);

        // create graph
        std::vector<Operator> inputs {inputOp};
        std::vector<Operator> outputs {outputOp};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    void BuildGraphDilation255(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildDataOp("input", {1, 4096, 4096, 1}, FORMAT_NHWC, DT_FLOAT);

        std::vector<uint32_t> block {8, 8};
        auto blockOp = BuildConstOp("space_block", {2}, reinterpret_cast<uint8_t*>(block.data()), block.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);

        std::vector<uint32_t> paddings {0, 0, 0, 0};
        auto paddingsOp = BuildConstOp("space_paddings", {2, 2}, reinterpret_cast<uint8_t*>(paddings.data()), paddings.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);

        auto spacetobatchOp = BuildSpaceToBatchOp("spacetobatch", inputOp, blockOp, paddingsOp);

        std::vector<float> filter {0.1};
        auto convFilterOp = BuildConstOp("conv_filter", {1, 1, 1, 1}, reinterpret_cast<uint8_t*>(filter.data()), filter.size() * sizeof(float), FORMAT_NHWC, DT_FLOAT);

        std::vector<float> bias {0.1};
        auto convBiasOp = BuildConstOp("conv_bias", {1}, reinterpret_cast<uint8_t*>(bias.data()), bias.size() * sizeof(float), FORMAT_NHWC, DT_FLOAT);
        auto conv2dOp = BuildConv2dOp("conv2d", spacetobatchOp, convFilterOp, convBiasOp, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0}, {32, 32, 32, 32});

        std::vector<uint32_t> batchBlock {1, 1};
        auto batchBlockOp = BuildConstOp("batch_block", {2}, reinterpret_cast<uint8_t*>(batchBlock.data()), batchBlock.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);

        std::vector<uint32_t> crops {0, 0, 0, 0};
        auto batchCropsOp = BuildConstOp("batch_crops", {2, 2}, reinterpret_cast<uint8_t*>(crops.data()), crops.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);
        auto batchtospaceOp = BuildBatchToSpaceOp("batchtospace", conv2dOp, batchBlockOp, batchCropsOp);

        auto outputOp = BuildDataOp("output", {1, 4096, 4096, 1}, FORMAT_NHWC, DT_FLOAT);
        outputOp.set_input_x(batchtospaceOp);

        // create graph
        std::vector<Operator> inputs {inputOp};
        std::vector<Operator> outputs {outputOp};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    void BuildGraphStride2(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildDataOp("input", {1, 32, 32, 1}, FORMAT_NHWC, DT_FLOAT);

        std::vector<uint32_t> block {2, 2};
        auto blockOp = BuildConstOp("space_block", {2}, reinterpret_cast<uint8_t*>(block.data()), block.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);

        std::vector<uint32_t> paddings {0, 0, 0, 0};
        auto paddingsOp = BuildConstOp("space_paddings", {2, 2}, reinterpret_cast<uint8_t*>(paddings.data()), paddings.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);

        auto spacetobatchOp = BuildSpaceToBatchOp("spacetobatch", inputOp, blockOp, paddingsOp);

        std::vector<float> filter {0.1};
        auto convFilterOp = BuildConstOp("conv_filter", {1, 1, 1, 1}, reinterpret_cast<uint8_t*>(filter.data()), filter.size() * sizeof(float), FORMAT_NHWC, DT_FLOAT);

        std::vector<float> bias {0.1};
        auto convBiasOp = BuildConstOp("conv_bias", {1}, reinterpret_cast<uint8_t*>(bias.data()), bias.size() * sizeof(float), FORMAT_NHWC, DT_FLOAT);
        auto conv2dOp = BuildConv2dOp("conv2d", spacetobatchOp, convFilterOp, convBiasOp, "NHWC", {2, 2, 2, 2}, {0, 0, 0, 0}, {1, 1, 1, 1});

        std::vector<uint32_t> batchBlock {1, 1};
        auto batchBlockOp = BuildConstOp("batch_block", {2}, reinterpret_cast<uint8_t*>(batchBlock.data()), batchBlock.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);

        std::vector<uint32_t> crops {0, 0, 0, 0};
        auto batchCropsOp = BuildConstOp("batch_crops", {2, 2}, reinterpret_cast<uint8_t*>(crops.data()), crops.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);
        auto batchtospaceOp = BuildBatchToSpaceOp("batchtospace", conv2dOp, batchBlockOp, batchCropsOp);

        auto outputOp = BuildDataOp("output", {1, 32, 32, 1}, FORMAT_NHWC, DT_FLOAT);
        outputOp.set_input_x(batchtospaceOp);

        // create graph
        std::vector<Operator> inputs {inputOp};
        std::vector<Operator> outputs {outputOp};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    void BuildGraphPad(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildDataOp("input", {1, 32, 32, 1}, FORMAT_NHWC, DT_FLOAT);

        std::vector<uint32_t> block {1, 1};
        auto blockOp = BuildConstOp("space_block", {2}, reinterpret_cast<uint8_t*>(block.data()), block.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);

        std::vector<uint32_t> paddings {128, 128, 128, 128};
        auto paddingsOp = BuildConstOp("space_paddings", {2, 2}, reinterpret_cast<uint8_t*>(paddings.data()), paddings.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);

        auto spacetobatchOp = BuildSpaceToBatchOp("spacetobatch", inputOp, blockOp, paddingsOp);

        std::vector<float> filter {0.1};
        auto convFilterOp = BuildConstOp("conv_filter", {1, 1, 1, 1}, reinterpret_cast<uint8_t*>(filter.data()), filter.size() * sizeof(float), FORMAT_NHWC, DT_FLOAT);

        std::vector<float> bias {0.1};
        auto convBiasOp = BuildConstOp("conv_bias", {1}, reinterpret_cast<uint8_t*>(bias.data()), bias.size() * sizeof(float), FORMAT_NHWC, DT_FLOAT);
        auto conv2dOp = BuildConv2dOp("conv2d", spacetobatchOp, convFilterOp, convBiasOp, "NHWC", {1, 1, 1, 1}, {128, 128, 128, 128}, {1, 1, 1, 1});

        std::vector<uint32_t> batchBlock {1, 1};
        auto batchBlockOp = BuildConstOp("batch_block", {2}, reinterpret_cast<uint8_t*>(batchBlock.data()), batchBlock.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);

        std::vector<uint32_t> crops {0, 0, 0, 0};
        auto batchCropsOp = BuildConstOp("batch_crops", {1, 1}, reinterpret_cast<uint8_t*>(crops.data()), crops.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);
        auto batchtospaceOp = BuildBatchToSpaceOp("batchtospace", conv2dOp, batchBlockOp, batchCropsOp);

        auto outputOp = BuildDataOp("output", {1, 288, 288, 1}, FORMAT_NHWC, DT_FLOAT);
        outputOp.set_input_x(batchtospaceOp);

        // create graph
        std::vector<Operator> inputs {inputOp};
        std::vector<Operator> outputs {outputOp};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    void BuildGraphSpaceConvPads(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildDataOp("input", {1, 32, 32, 1}, FORMAT_NHWC, DT_FLOAT);

        std::vector<uint32_t> block {1, 1};
        auto blockOp = BuildConstOp("space_block", {2}, reinterpret_cast<uint8_t*>(block.data()), block.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);

        std::vector<uint32_t> paddings {1, 1, 1, 1};
        auto paddingsOp = BuildConstOp("space_paddings", {2, 2}, reinterpret_cast<uint8_t*>(paddings.data()), paddings.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);

        auto spacetobatchOp = BuildSpaceToBatchOp("spacetobatch", inputOp, blockOp, paddingsOp);

        std::vector<float> filter {0.1};
        auto convFilterOp = BuildConstOp("conv_filter", {1, 1, 1, 1}, reinterpret_cast<uint8_t*>(filter.data()), filter.size() * sizeof(float), FORMAT_NHWC, DT_FLOAT);

        std::vector<float> bias {0.1};
        auto convBiasOp = BuildConstOp("conv_bias", {1}, reinterpret_cast<uint8_t*>(bias.data()), bias.size() * sizeof(float), FORMAT_NHWC, DT_FLOAT);
        auto conv2dOp = BuildConv2dOp("conv2d", spacetobatchOp, convFilterOp, convBiasOp, "NHWC", {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1});

        std::vector<uint32_t> batchBlock {1, 1};
        auto batchBlockOp = BuildConstOp("batch_block", {2}, reinterpret_cast<uint8_t*>(batchBlock.data()), batchBlock.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);

        std::vector<uint32_t> crops {0, 0, 0, 0};
        auto batchCropsOp = BuildConstOp("batch_crops", {1, 1}, reinterpret_cast<uint8_t*>(crops.data()), crops.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);
        auto batchtospaceOp = BuildBatchToSpaceOp("batchtospace", conv2dOp, batchBlockOp, batchCropsOp);

        auto outputOp = BuildDataOp("output", {1, 36, 36, 1}, FORMAT_NHWC, DT_FLOAT);
        outputOp.set_input_x(batchtospaceOp);

        // create graph
        std::vector<Operator> inputs {inputOp};
        std::vector<Operator> outputs {outputOp};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    void BuildGraphBlock(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildDataOp("input", {1, 32, 32, 1}, FORMAT_NHWC, DT_FLOAT);

        std::vector<uint32_t> block {2, 2};
        auto blockOp = BuildConstOp("space_block", {2}, reinterpret_cast<uint8_t*>(block.data()), block.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);

        std::vector<uint32_t> paddings {1, 1, 1, 1};
        auto paddingsOp = BuildConstOp("space_paddings", {2, 2}, reinterpret_cast<uint8_t*>(paddings.data()), paddings.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);

        auto spacetobatchOp = BuildSpaceToBatchOp("spacetobatch", inputOp, blockOp, paddingsOp);

        std::vector<float> filter {0.1};
        auto convFilterOp = BuildConstOp("conv_filter", {1, 1, 1, 1}, reinterpret_cast<uint8_t*>(filter.data()), filter.size() * sizeof(float), FORMAT_NHWC, DT_FLOAT);

        std::vector<float> bias {0.1};
        auto convBiasOp = BuildConstOp("conv_bias", {1}, reinterpret_cast<uint8_t*>(bias.data()), bias.size() * sizeof(float), FORMAT_NHWC, DT_FLOAT);
        auto conv2dOp = BuildConv2dOp("conv2d", spacetobatchOp, convFilterOp, convBiasOp, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 1, 1});

        std::vector<uint32_t> batchBlock {2, 2};
        auto batchBlockOp = BuildConstOp("batch_block", {2}, reinterpret_cast<uint8_t*>(batchBlock.data()), batchBlock.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);

        std::vector<uint32_t> crops {0, 0, 0, 0};
        auto batchCropsOp = BuildConstOp("batch_crops", {1, 1}, reinterpret_cast<uint8_t*>(crops.data()), crops.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);
        auto batchtospaceOp = BuildBatchToSpaceOp("batchtospace", conv2dOp, batchBlockOp, batchCropsOp);

        auto outputOp = BuildDataOp("output", {1, 32, 32, 1}, FORMAT_NHWC, DT_FLOAT);
        outputOp.set_input_x(batchtospaceOp);

        // create graph
        std::vector<Operator> inputs {inputOp};
        std::vector<Operator> outputs {outputOp};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }

    void BuildGraphCropsNot0(ComputeGraphPtr &computeGraph)
    {
        auto inputOp = BuildDataOp("input", {1, 32, 32, 1}, FORMAT_NHWC, DT_FLOAT);

        std::vector<uint32_t> block {2, 2};
        auto blockOp = BuildConstOp("space_block", {2}, reinterpret_cast<uint8_t*>(block.data()), block.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);

        std::vector<uint32_t> paddings {0, 0, 0, 0};
        auto paddingsOp = BuildConstOp("space_paddings", {2, 2}, reinterpret_cast<uint8_t*>(paddings.data()), paddings.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);

        auto spacetobatchOp = BuildSpaceToBatchOp("spacetobatch", inputOp, blockOp, paddingsOp);

        std::vector<float> filter {0.1};
        auto convFilterOp = BuildConstOp("conv_filter", {1, 1, 1, 1}, reinterpret_cast<uint8_t*>(filter.data()), filter.size() * sizeof(float), FORMAT_NHWC, DT_FLOAT);

        std::vector<float> bias {0.1};
        auto convBiasOp = BuildConstOp("conv_bias", {1}, reinterpret_cast<uint8_t*>(bias.data()), bias.size() * sizeof(float), FORMAT_NHWC, DT_FLOAT);
        auto conv2dOp = BuildConv2dOp("conv2d", spacetobatchOp, convFilterOp, convBiasOp, "NHWC", {1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 1, 1});

        std::vector<uint32_t> batchBlock {2, 2};
        auto batchBlockOp = BuildConstOp("batch_block", {2}, reinterpret_cast<uint8_t*>(batchBlock.data()), batchBlock.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);

        std::vector<uint32_t> crops {1, 0, 0, 0};
        auto batchCropsOp = BuildConstOp("batch_crops", {2, 2}, reinterpret_cast<uint8_t*>(crops.data()), crops.size() * sizeof(uint32_t), FORMAT_ND, DT_INT32);
        auto batchtospaceOp = BuildBatchToSpaceOp("batchtospace", conv2dOp, batchBlockOp, batchCropsOp);

        auto outputOp = BuildDataOp("output", {1, 32, 32, 1}, FORMAT_NHWC, DT_FLOAT);
        outputOp.set_input_x(batchtospaceOp);

        // create graph
        std::vector<Operator> inputs {inputOp, blockOp, paddingsOp, convFilterOp, convBiasOp, batchBlockOp, batchCropsOp};
        std::vector<Operator> outputs {outputOp};

        ge::Graph graph("test");
        graph.SetInputs(inputs).SetOutputs(outputs);
        computeGraph = ge::GraphUtils::GetComputeGraph(graph);
    }
};

TEST_F(spacetobatch_conv2d_batchtospace_test, spacetobatch_conv2d_batchtospace_test_smoke)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraph(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SpaceToBatchConv2dBatchToSpacePass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);
    bool findSpacetobatch = false;
    bool findBatchtospace = false;
    for (auto node: computeGraph->GetAllNodes()) {
        if (node->GetType() == "SpaceToBatchND") {
            std::cout << "find SpaceToBatchND, failed" << std::endl;
            findSpacetobatch = true;
        }
        if (node->GetType() == "BatchToSpaceND") {
            std::cout << "find BatchToSpaceND, failed" << std::endl;
            findBatchtospace = true;
        }
    }
    EXPECT_EQ(findSpacetobatch, false);
    EXPECT_EQ(findBatchtospace, false);
}

TEST_F(spacetobatch_conv2d_batchtospace_test, spacetobatch_conv2d_batchtospace_test_no_pattern)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphNoPattern(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SpaceToBatchConv2dBatchToSpacePass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);
    bool findSpacetobatch = false;
    bool findBatchtospace = false;
    for (auto node: computeGraph->GetAllNodes()) {
        if (node->GetType() == "SpaceToBatchND") {
            findSpacetobatch = true;
        }
        if (node->GetType() == "BatchToSpaceND") {
            findBatchtospace = true;
        }
    }
    EXPECT_EQ(findSpacetobatch, false);
    EXPECT_EQ(findBatchtospace, false);
}

TEST_F(spacetobatch_conv2d_batchtospace_test, spacetobatch_conv2d_batchtospace_test_dilation_255)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphDilation255(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SpaceToBatchConv2dBatchToSpacePass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);
    bool findSpacetobatch = false;
    bool findBatchtospace = false;
    for (auto node: computeGraph->GetAllNodes()) {
        if (node->GetType() == "SpaceToBatchND") {
            findSpacetobatch = true;
        }
        if (node->GetType() == "BatchToSpaceND") {
            findBatchtospace = true;
        }
    }
    EXPECT_EQ(findSpacetobatch, true);
    EXPECT_EQ(findBatchtospace, true);
}

TEST_F(spacetobatch_conv2d_batchtospace_test, spacetobatch_conv2d_batchtospace_test_stride_2_2)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphStride2(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SpaceToBatchConv2dBatchToSpacePass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);
    bool findSpacetobatch = false;
    bool findBatchtospace = false;
    for (auto node: computeGraph->GetAllNodes()) {
        if (node->GetType() == "SpaceToBatchND") {
            findSpacetobatch = true;
        }
        if (node->GetType() == "BatchToSpaceND") {
            findBatchtospace = true;
        }
    }
    EXPECT_EQ(findSpacetobatch, true);
    EXPECT_EQ(findBatchtospace, true);
}

TEST_F(spacetobatch_conv2d_batchtospace_test, spacetobatch_conv2d_batchtospace_test_pad)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphPad(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SpaceToBatchConv2dBatchToSpacePass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);
    bool findSpacetobatch = false;
    bool findBatchtospace = false;
    for (auto node: computeGraph->GetAllNodes()) {
        if (node->GetType() == "SpaceToBatchND") {
            findSpacetobatch = true;
        }
        if (node->GetType() == "BatchToSpaceND") {
            findBatchtospace = true;
        }
    }
    EXPECT_EQ(findSpacetobatch, true);
    EXPECT_EQ(findBatchtospace, true);
}

TEST_F(spacetobatch_conv2d_batchtospace_test, spacetobatch_conv2d_batchtospace_test_space_conv_pads)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphSpaceConvPads(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SpaceToBatchConv2dBatchToSpacePass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);
    bool findSpacetobatch = false;
    bool findBatchtospace = false;
    for (auto node: computeGraph->GetAllNodes()) {
        if (node->GetType() == "SpaceToBatchND") {
            findSpacetobatch = true;
        }
        if (node->GetType() == "BatchToSpaceND") {
            findBatchtospace = true;
        }
    }
    EXPECT_EQ(findSpacetobatch, false);
    EXPECT_EQ(findBatchtospace, false);
}

TEST_F(spacetobatch_conv2d_batchtospace_test, spacetobatch_conv2d_batchtospace_test_block)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphSpaceConvPads(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SpaceToBatchConv2dBatchToSpacePass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);
    bool findSpacetobatch = false;
    bool findBatchtospace = false;
    for (auto node: computeGraph->GetAllNodes()) {
        if (node->GetType() == "SpaceToBatchND") {
            findSpacetobatch = true;
        }
        if (node->GetType() == "BatchToSpaceND") {
            findBatchtospace = true;
        }
    }
    EXPECT_EQ(findSpacetobatch, false);
    EXPECT_EQ(findBatchtospace, false);
}

TEST_F(spacetobatch_conv2d_batchtospace_test, spacetobatch_conv2d_batchtospace_test_crops_not_0)
{
    ge::ComputeGraphPtr computeGraph;
    BuildGraphCropsNot0(computeGraph);
    FusionPassTestUtils::InferShapeAndType(computeGraph);
    FusionPassTestUtils::RunGraphFusionPass("SpaceToBatchConv2dBatchToSpacePass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);
    bool findSpacetobatch = false;
    bool findBatchtospace = false;
    for (auto node: computeGraph->GetAllNodes()) {
        if (node->GetType() == "SpaceToBatchND") {
            findSpacetobatch = true;
        }
        if (node->GetType() == "BatchToSpaceND") {
            findBatchtospace = true;
        }
    }
    EXPECT_EQ(findSpacetobatch, true);
    EXPECT_EQ(findBatchtospace, true);
}

} // namespace fe
