#include <string>
#include <vector>
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "aipp.h"
#include "graph/debug/ge_attr_define.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include <nlohmann/json.hpp>
#include "securec.h"

class Aipp : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Aipp Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Aipp Proto Test TearDown" << std::endl;
  }
};


TEST_F(Aipp, aipp_data_slice_infer1) {
  ge::op::Aipp op;

  auto tensor_desc = create_desc_with_ori({4,3,224,224}, ge::DT_UINT8, ge::FORMAT_NCHW, {4,3,224,224}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("images", tensor_desc);

  auto output_tensor_desc_temp = create_desc_with_ori({4,1,224,224,32}, ge::DT_UINT8, ge::FORMAT_NC1HWC0, {4,3,224,224}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("features", output_tensor_desc_temp);

  std::vector<std::vector<int64_t>> output_data_slice ={{0,1}, {}, {}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc("features");
  ge::AttrUtils::SetListListInt(output_tensor_desc, ge::ATTR_NAME_DATA_SLICE, output_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_images = op_desc->MutableInputDesc("images");
  std::vector<std::vector<int64_t>> images_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_images, ge::ATTR_NAME_DATA_SLICE, images_data_slice);

  std::vector<std::vector<int64_t>> expected_images_data_slice = {{0,1}, {}, {}, {}};
  EXPECT_EQ(expected_images_data_slice, images_data_slice);
}


TEST_F(Aipp, aipp_infershape_diff_test_1) {
  ge::op::Aipp op;

  auto tensor_desc = create_desc_with_ori({1,3,224,224}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1,3,224,224}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("images", tensor_desc);

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string cfgFile = caseDir + "/aipp_static.cfg";
  std::cout << "cfgFile:" << cfgFile.c_str() << std::endl;
  op.SetAttr("aipp_config_path", cfgFile);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("features");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {1, 3, 224, 224};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(Aipp, aipp_infershape_diff_test_2) {
  ge::op::Aipp op;

  op.UpdateInputDesc("images", create_desc_shape_range({1,3,224,224}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1,3,224,224},
                     ge::FORMAT_NCHW, {{1,1},{3,3},{224,224},{224,224}}));
  op.UpdateInputDesc("params", create_desc_shape_range({3168}, ge::DT_UINT8, ge::FORMAT_ND, {3168},
                     ge::FORMAT_ND, {{160,3168}}));

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string cfgFile = caseDir + "/aipp_dynamic.cfg";
  std::cout << "cfgFile:" << cfgFile.c_str() << std::endl;
  op.SetAttr("aipp_config_path", cfgFile);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("features");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {1, 3, 224, 224};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1,1},{3,3},{224,224},{224,224}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(Aipp, aipp_infershape_diff_test_3) {
  ge::op::Aipp op;

  op.UpdateInputDesc("images", create_desc_shape_range({1,3,64,64}, ge::DT_UINT8, ge::FORMAT_NCHW, {1,3,64,64},
                     ge::FORMAT_NCHW, {{1,1},{3,3},{64,64},{64,64}}));
  op.UpdateOutputDesc("features", create_desc_shape_range({1,1,64,64,16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1,3,64,64},
                      ge::FORMAT_NCHW, {{1,1},{1,1},{64,64},{64,64},{16,16}}));
  nlohmann::json aipp_config_path_json;
  aipp_config_path_json["aipp_mode"] = "dynamic";
  op.SetAttr("aipp_config_path", aipp_config_path_json.dump());

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_UINT8);
  constDesc.SetSize(160 * sizeof(uint8_t));
  constTensor.SetTensorDesc(constDesc);
  uint8_t constData[160] = {0};
  *(constData + 0) = (uint8_t) 1;
  *(constData + 4) = (int8_t) 1;
  *(constData + 8) = (int8_t) 64;
  *(constData + 12) = (int8_t) 64;
  *(constData + 64) = (int8_t) 1;
  *(constData + 64 + 2) = (int8_t) 1;
  *(constData + 64 + 16) = (int8_t) 64;
  *(constData + 64 + 20) = (int8_t) 64;
  constTensor.SetData((uint8_t*)constData, 160 * sizeof(uint8_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);

  op.set_input_params(const0);
  auto desc = op.GetInputDesc("params");
  desc.SetDataType(ge::DT_UINT8);
  op.UpdateInputDesc("params", desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(Aipp, aipp_infershape_diff_test_4) {
  ge::op::Aipp op;

  op.UpdateInputDesc("images", create_desc_shape_range({1,3,64,64}, ge::DT_UINT8, ge::FORMAT_NCHW, {1,3,64,64},
                     ge::FORMAT_NCHW, {{1,1},{3,3},{64,64},{64,64}}));
  op.UpdateOutputDesc("features", create_desc_shape_range({1,1,64,64,16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1,3,64,64},
                      ge::FORMAT_NCHW, {{1,1},{1,1},{64,64},{64,64},{16,16}}));
  nlohmann::json aipp_config_path_json;
  aipp_config_path_json["aipp_mode"] = "dynamic";
  op.SetAttr("aipp_config_path", aipp_config_path_json.dump());

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_UINT8);
  constDesc.SetSize(160 * sizeof(uint8_t));
  constTensor.SetTensorDesc(constDesc);
  uint8_t constData[160] = {0};
  *(constData + 0) = (uint8_t) 5;
  *(constData + 4) = (int8_t) 1;
  *(constData + 8) = (int8_t) 64;
  *(constData + 12) = (int8_t) 64;
  *(constData + 64) = (int8_t) 1;
  *(constData + 64 + 2) = (int8_t) 1;
  *(constData + 64 + 16) = (int8_t) 64;
  *(constData + 64 + 20) = (int8_t) 64;
  constTensor.SetData((uint8_t*)constData, 160 * sizeof(uint8_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);

  op.set_input_params(const0);
  auto desc = op.GetInputDesc("params");
  desc.SetDataType(ge::DT_UINT8);
  op.UpdateInputDesc("params", desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(Aipp, aipp_infershape_diff_test_5) {
  ge::op::Aipp op;

  op.UpdateInputDesc("images", create_desc_shape_range({1,3,64,64}, ge::DT_UINT8, ge::FORMAT_NCHW, {1,3,64,64},
                     ge::FORMAT_NCHW, {{1,1},{3,3},{64,64},{64,64}}));
  op.UpdateOutputDesc("features", create_desc_shape_range({1,1,64,64,16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1,3,64,64},
                      ge::FORMAT_NCHW, {{1,1},{1,1},{64,64},{64,64},{16,16}}));
  nlohmann::json aipp_config_path_json;
  aipp_config_path_json["aipp_mode"] = "dynamic";
  op.SetAttr("aipp_config_path", aipp_config_path_json.dump());

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_UINT8);
  constDesc.SetSize(160 * sizeof(uint8_t));
  constTensor.SetTensorDesc(constDesc);
  uint8_t constData[160] = {0};
  *(constData + 0) = (uint8_t) 10;
  *(constData + 4) = (int8_t) 1;
  *(constData + 8) = (int8_t) 64;
  *(constData + 12) = (int8_t) 64;
  *(constData + 64) = (int8_t) 1;
  *(constData + 64 + 2) = (int8_t) 1;
  *(constData + 64 + 16) = (int8_t) 64;
  *(constData + 64 + 20) = (int8_t) 64;
  constTensor.SetData((uint8_t*)constData, 160 * sizeof(uint8_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);

  op.set_input_params(const0);
  auto desc = op.GetInputDesc("params");
  desc.SetDataType(ge::DT_UINT8);
  op.UpdateInputDesc("params", desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(Aipp, aipp_infershape_diff_test_6) {
  ge::op::Aipp op;

  op.UpdateInputDesc("images", create_desc_shape_range({1,3,224,224}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1,3,224,224},
                     ge::FORMAT_NCHW, {{1,1},{3,3},{224,224},{224,224}}));
  op.UpdateInputDesc("params", create_desc_shape_range({3168}, ge::DT_UINT8, ge::FORMAT_ND, {3168},
                     ge::FORMAT_ND, {{160,3168}}));

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string cfgFile = caseDir + "/aipp_dynamic_error.cfg";
  std::cout << "cfgFile:" << cfgFile.c_str() << std::endl;
  op.SetAttr("aipp_config_path", cfgFile);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
