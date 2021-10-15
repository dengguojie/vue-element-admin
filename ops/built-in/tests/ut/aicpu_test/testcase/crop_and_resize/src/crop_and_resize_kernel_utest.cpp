#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "aicpu_read_file.h"
#undef private
#undef protected
#include "Eigen/Core"
#include <fstream>

using namespace std;
using namespace aicpu;

class TEST_CropAndResize_UTest : public testing::Test {};

constexpr int BATCH_SIZE = 1;
constexpr int NUM_BOXES = 5;
constexpr int IMAGE_HEIGHT = 256;
constexpr int IMAGE_WIDTH = 256;
constexpr int CHANNELS = 3;
float inputs[BATCH_SIZE][IMAGE_HEIGHT][IMAGE_WIDTH][CHANNELS] = {0};
float boxes[NUM_BOXES][4] = {0};
float y[5][24][24][3] = {0};
float expected_y[5][24][24][3] = {0};

TEST_F(TEST_CropAndResize_UTest, CropAndResizeKernel_Success) {
  // raw data
  int32_t crop_size[2] = {24, 24};
  int32_t box_index[5] = {0};

  std::ifstream in_image(ktestcaseFilePath + "crop_and_resize/data/image_data");
  if (!in_image.is_open()) {
    std::cout << "open image_data error" << std::endl;
    exit(1);
  }

  for (int i = 0; i < BATCH_SIZE; i++) {
    for (int j = 0; j < IMAGE_HEIGHT; j++) {
      for (int t = 0; t < IMAGE_WIDTH; t++) {
        for (int s = 0; s < CHANNELS; s++) {
          float value;
          in_image >> value;
          inputs[i][j][t][s] = value;
        }
      }
    }
  }
  in_image.close();

  std::ifstream in_boxes(ktestcaseFilePath + "crop_and_resize/data/boxes_data");
  if (!in_boxes.is_open()) {
    std::cout << "open boxes_data error" << std::endl;
    exit(1);
  }
  for (int i = 0; i < NUM_BOXES; i++) {
    for (int j = 0; j < 4; j++) {
      float value;
      in_boxes >> value;
      boxes[i][j] = value;
    }
  }
  in_boxes.close();

  std::ifstream in_expected(ktestcaseFilePath + "crop_and_resize/data/expected_output_data");
  if (!in_expected.is_open()) {
    std::cout << "open expected_output_data error" << std::endl;
    exit(1);
  }
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 24; j++) {
      for (int t = 0; t < 24; t++) {
        for (int s = 0; s < 3; s++) {
          float value;
          in_expected >> value;
          expected_y[i][j][t][s] = value;
        }
      }
    }
  }
  in_expected.close();

  auto nodeDef = CpuKernelUtils::CreateNodeDef();
  nodeDef->SetOpType("CropAndResize");

  // set attr
  auto method = CpuKernelUtils::CreateAttrValue();
  method->SetString("bilinear");
  nodeDef->AddAttrs("method", method.get());

  auto extrapolation_value = CpuKernelUtils::CreateAttrValue();
  extrapolation_value->SetFloat(0);
  nodeDef->AddAttrs("extrapolation_value", extrapolation_value.get());

  // set input
  // image
  auto inputTensor0 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor0, nullptr);
  auto aicpuShape0 = inputTensor0->GetTensorShape();
  std::vector<int64_t> shapes0 = {BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS};
  aicpuShape0->SetDimSizes(shapes0);
  inputTensor0->SetDataType(DT_FLOAT);
  inputTensor0->SetData(inputs);
  inputTensor0->SetDataSize(sizeof(inputs));
  // boxes
  auto inputTensor1 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor1, nullptr);
  auto aicpuShape1 = inputTensor1->GetTensorShape();
  std::vector<int64_t> shapes1 = {NUM_BOXES, 4};
  aicpuShape1->SetDimSizes(shapes1);
  inputTensor1->SetDataType(DT_FLOAT);
  inputTensor1->SetData(boxes);
  inputTensor1->SetDataSize(sizeof(boxes));
  // box_index
  auto inputTensor2 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor2, nullptr);
  auto aicpuShape2 = inputTensor2->GetTensorShape();
  std::vector<int64_t> shapes2 = {NUM_BOXES};
  aicpuShape2->SetDimSizes(shapes2);
  inputTensor2->SetDataType(DT_INT32);
  inputTensor2->SetData(box_index);
  inputTensor2->SetDataSize(sizeof(box_index));
  // crop_size
  auto inputTensor3 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor3, nullptr);
  auto aicpuShape3 = inputTensor3->GetTensorShape();
  std::vector<int64_t> shapes3 = {2};
  aicpuShape3->SetDimSizes(shapes3);
  inputTensor3->SetDataType(DT_INT32);
  inputTensor3->SetData(crop_size);
  inputTensor3->SetDataSize(sizeof(crop_size));

  // set output
  auto outputTensor1 = nodeDef->AddOutputs();
  EXPECT_NE(outputTensor1, nullptr);
  auto aicpuShape4 = outputTensor1->GetTensorShape();
  std::vector<int64_t> shapes4 = {5, 24, 24, 3};
  aicpuShape4->SetDimSizes(shapes4);
  outputTensor1->SetDataType(DT_FLOAT);
  outputTensor1->SetData(y);
  outputTensor1->SetDataSize(sizeof(y));

  CpuKernelContext ctx(DEVICE);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
  EXPECT_EQ(ret, KERNEL_STATUS_OK);
  if (ret != 0) {
    exit(1);
  }

  float eps = 0.0001;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 24; j++) {
      for (int t = 0; t < 24; t++) {
        for (int s = 0; s < 3; s++) {
          EXPECT_LT(std::abs(y[i][j][t][s] - expected_y[i][j][t][s]), eps);
        }
      }
    }
  }
}

TEST_F(TEST_CropAndResize_UTest, CropAndResizeKernel_V2_Success) {
  // raw data
  int32_t crop_size[2] = {24, 24};
  int32_t box_index[5] = {0};

  std::ifstream in_image(ktestcaseFilePath + "crop_and_resize/data/image_data");
  if (!in_image.is_open()) {
    std::cout << "open image_data error" << std::endl;
    exit(1);
  }

  for (int i = 0; i < BATCH_SIZE; i++) {
    for (int j = 0; j < IMAGE_HEIGHT; j++) {
      for (int t = 0; t < IMAGE_WIDTH; t++) {
        for (int s = 0; s < CHANNELS; s++) {
          float value;
          in_image >> value;
          inputs[i][j][t][s] = value;
        }
      }
    }
  }
  in_image.close();

  std::ifstream in_boxes(ktestcaseFilePath + "crop_and_resize/data/boxes_data");
  if (!in_boxes.is_open()) {
    std::cout << "open boxes_data error" << std::endl;
    exit(1);
  }
  for (int i = 0; i < NUM_BOXES; i++) {
    for (int j = 0; j < 4; j++) {
      float value;
      in_boxes >> value;
      boxes[i][j] = value;
    }
  }
  in_boxes.close();

  std::ifstream in_expected(ktestcaseFilePath + "crop_and_resize/data/expected_output_data_v2");
  if (!in_expected.is_open()) {
    std::cout << "open expected_output_data error" << std::endl;
    exit(1);
  }
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 24; j++) {
      for (int t = 0; t < 24; t++) {
        for (int s = 0; s < 3; s++) {
          float value;
          in_expected >> value;
          expected_y[i][j][t][s] = value;
        }
      }
    }
  }
  in_expected.close();

  auto nodeDef = CpuKernelUtils::CreateNodeDef();
  nodeDef->SetOpType("CropAndResize");

  // set attr
  auto method = CpuKernelUtils::CreateAttrValue();
  method->SetString("bilinear_v2");
  nodeDef->AddAttrs("method", method.get());

  auto extrapolation_value = CpuKernelUtils::CreateAttrValue();
  extrapolation_value->SetFloat(0);
  nodeDef->AddAttrs("extrapolation_value", extrapolation_value.get());

  // set input
  // image
  auto inputTensor0 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor0, nullptr);
  auto aicpuShape0 = inputTensor0->GetTensorShape();
  std::vector<int64_t> shapes0 = {BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS};
  aicpuShape0->SetDimSizes(shapes0);
  inputTensor0->SetDataType(DT_FLOAT);
  inputTensor0->SetData(inputs);
  inputTensor0->SetDataSize(sizeof(inputs));
  // boxes
  auto inputTensor1 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor1, nullptr);
  auto aicpuShape1 = inputTensor1->GetTensorShape();
  std::vector<int64_t> shapes1 = {NUM_BOXES, 4};
  aicpuShape1->SetDimSizes(shapes1);
  inputTensor1->SetDataType(DT_FLOAT);
  inputTensor1->SetData(boxes);
  inputTensor1->SetDataSize(sizeof(boxes));
  // box_index
  auto inputTensor2 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor2, nullptr);
  auto aicpuShape2 = inputTensor2->GetTensorShape();
  std::vector<int64_t> shapes2 = {NUM_BOXES};
  aicpuShape2->SetDimSizes(shapes2);
  inputTensor2->SetDataType(DT_INT32);
  inputTensor2->SetData(box_index);
  inputTensor2->SetDataSize(sizeof(box_index));
  // crop_size
  auto inputTensor3 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor3, nullptr);
  auto aicpuShape3 = inputTensor3->GetTensorShape();
  std::vector<int64_t> shapes3 = {2};
  aicpuShape3->SetDimSizes(shapes3);
  inputTensor3->SetDataType(DT_INT32);
  inputTensor3->SetData(crop_size);
  inputTensor3->SetDataSize(sizeof(crop_size));

  // set output
  auto outputTensor1 = nodeDef->AddOutputs();
  EXPECT_NE(outputTensor1, nullptr);
  auto aicpuShape4 = outputTensor1->GetTensorShape();
  std::vector<int64_t> shapes4 = {5, 24, 24, 3};
  aicpuShape4->SetDimSizes(shapes4);
  outputTensor1->SetDataType(DT_FLOAT);
  outputTensor1->SetData(y);
  outputTensor1->SetDataSize(sizeof(y));

  CpuKernelContext ctx(DEVICE);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
  EXPECT_EQ(ret, KERNEL_STATUS_OK);
  if (ret != 0) {
    exit(1);
  }

  float eps = 0.0001;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 24; j++) {
      for (int t = 0; t < 24; t++) {
        for (int s = 0; s < 3; s++) {
          EXPECT_LT(std::abs(y[i][j][t][s] - expected_y[i][j][t][s]), eps);
        }
      }
    }
  }
}
