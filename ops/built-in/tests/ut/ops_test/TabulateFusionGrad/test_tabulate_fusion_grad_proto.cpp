#include <iostream>
#include <gtest/gtest.h>
#include "array_ops.h"
#include "deep_md.h"
#include "op_proto_test_util.h"

class TabulateFusionGradProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TabulateFusionGrad Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TabulateFusionGrad Proto Test TearDown" << std::endl;
  }
};

TEST_F(TabulateFusionGradProtoTest, TabulateFusionGradVerifyTest_1) {
  int32_t nloc = 12288;
  int32_t nnei = 138;
  int32_t lastLayerSize = 128;
  int32_t split_count = 1;
  int32_t split_index = 0;

  int32_t tableDim0 = 1024;
  int32_t descriptorDim0 = 1024;
  int32_t descriptorDim1 = 1024;

  ge::op::TabulateFusionGrad op;
  op.UpdateInputDesc("table", create_desc({tableDim0, lastLayerSize * 6}, ge::DT_FLOAT));
  op.UpdateInputDesc("table_info", create_desc({6,}, ge::DT_FLOAT));
  op.UpdateInputDesc("em_x", create_desc({nloc, nnei}, ge::DT_FLOAT));
  op.UpdateInputDesc("em", create_desc({nloc, nnei, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("dy", create_desc({nloc, 4, lastLayerSize}, ge::DT_FLOAT));
  op.UpdateInputDesc("descriptor", create_desc({descriptorDim0, descriptorDim1, lastLayerSize}, ge::DT_FLOAT));
  op.SetAttr("split_count", split_count);
  op.SetAttr("split_index", split_index);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(TabulateFusionGradProtoTest, TabulateFusionGradInferShapeTest_1) {
  int32_t nloc = 12288;
  int32_t nnei = 138;
  int32_t lastLayerSize = 128;
  int32_t split_count = 1;
  int32_t split_index = 0;

  ge::op::TabulateFusionGrad op;
  op.UpdateInputDesc("table",
                     create_desc_shape_range({1024, lastLayerSize * 6}, ge::DT_FLOAT, ge::FORMAT_ND,
                                             {1024, lastLayerSize * 6}, ge::FORMAT_ND,
                                             {{1024, 1024}, {lastLayerSize * 6, lastLayerSize * 6}}));
  op.UpdateInputDesc("table_info",
                     create_desc_shape_range({6,}, ge::DT_FLOAT, ge::FORMAT_ND, {6,}, ge::FORMAT_ND, {{6, 6}}));
  op.UpdateInputDesc("em_x", create_desc_shape_range({nloc, nnei}, ge::DT_FLOAT, ge::FORMAT_ND, {nloc, nnei},
                                                     ge::FORMAT_ND, {{nloc, nloc}, {nnei, nnei}}));
  op.UpdateInputDesc("em", create_desc_shape_range({nloc, nnei, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {nloc, nnei, 4},
                                                   ge::FORMAT_ND, {{nloc, nloc}, {nnei, nnei}, {4, 4}}));
  op.UpdateInputDesc("dy",
      create_desc_shape_range({nloc, 4, lastLayerSize}, ge::DT_FLOAT, ge::FORMAT_ND, {nloc, 4, lastLayerSize},
                                    ge::FORMAT_ND, {{nloc, nloc}, {4, 4}, {lastLayerSize, lastLayerSize}}));
  op.UpdateInputDesc("descriptor",
      create_desc_shape_range(
          {nloc, 4, lastLayerSize}, ge::DT_FLOAT, ge::FORMAT_ND,
          {nloc, 4, lastLayerSize}, ge::FORMAT_ND,
          {{nloc, nloc}, {4, 4}, {lastLayerSize, lastLayerSize}}));
  op.SetAttr("split_count", split_count);
  op.SetAttr("split_index", split_index);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  {
    auto output_desc = op.GetOutputDescByName("dy_dem_x");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {nloc, nnei};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  }

  {
    auto output_desc = op.GetOutputDescByName("dy_dem");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {nloc, nnei, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  }
}

TEST_F(TabulateFusionGradProtoTest, TabulateFusionGradInferShapeTest_2) {
  int32_t nloc = -1;
  int32_t nnei = 138;
  int32_t lastLayerSize = 128;
  int32_t split_count = 1;
  int32_t split_index = 0;

  ge::op::TabulateFusionGrad op;
  op.UpdateInputDesc("table",
                     create_desc_shape_range({1024, lastLayerSize * 6}, ge::DT_FLOAT, ge::FORMAT_ND,
                                             {1024, lastLayerSize * 6}, ge::FORMAT_ND,
                                             {{1024, 1024}, {lastLayerSize * 6, lastLayerSize * 6}}));
  op.UpdateInputDesc("table_info", create_desc_shape_range({6,}, ge::DT_FLOAT, ge::FORMAT_ND, {6,}, ge::FORMAT_ND,
                                                           {{6, 6}}));
  op.UpdateInputDesc("em_x", create_desc_shape_range({nloc, nnei}, ge::DT_FLOAT, ge::FORMAT_ND, {nloc, nnei},
                                                     ge::FORMAT_ND, {{nloc, nloc}, {nnei, nnei}}));
  op.UpdateInputDesc("em", create_desc_shape_range({nloc, nnei, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {nloc, nnei, 4},
                                                   ge::FORMAT_ND, {{nloc, nloc}, {nnei, nnei}, {4, 4}}));
  op.UpdateInputDesc("dy",
      create_desc_shape_range({nloc, 4, lastLayerSize}, ge::DT_FLOAT, ge::FORMAT_ND, {nloc, 4, lastLayerSize},
                                    ge::FORMAT_ND, {{nloc, nloc}, {4, 4}, {lastLayerSize, lastLayerSize}}));
  op.UpdateInputDesc("descriptor",
      create_desc_shape_range(
          {nloc, 4, lastLayerSize}, ge::DT_FLOAT, ge::FORMAT_ND,
          {nloc, 4, lastLayerSize}, ge::FORMAT_ND,
          {{nloc, nloc}, {4, 4}, {lastLayerSize, lastLayerSize}}));
  op.SetAttr("split_count", split_count);
  op.SetAttr("split_index", split_index);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  {
    auto output_desc = op.GetOutputDescByName("dy_dem_x");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {nloc, nnei};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

    std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{0, -1}, {nnei, nnei}};
    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    output_desc.GetShapeRange(output_shape_range);
    EXPECT_EQ(output_shape_range, expected_output_shape_range);
  }

  {
    auto output_desc = op.GetOutputDescByName("dy_dem");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {nloc, nnei, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

    std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{0, -1}, {nnei, nnei}, {4, 4}};
    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    output_desc.GetShapeRange(output_shape_range);
    EXPECT_EQ(output_shape_range, expected_output_shape_range);
  }
}
