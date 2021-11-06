/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "trans_data_rnn.h"
#include <cmath>
#include <vector>
#include <string>
#include "Eigen/Core"
#include "log.h"
#include "securec.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/kernel_util.h"

namespace {
const char *TRANS_DATA_RNN = "TransDataRNN";
constexpr int32_t ALIGN_16 = 16;
}

namespace aicpu {

TransDataRNNCpuKernel::TransDataRNNCpuKernel() {}

template <typename T>
static uint32_t DealBiasData(T *outputData, T *inputData, int32_t dstLen, 
                             int32_t hiddenSize, int32_t hiddenCnt, int32_t hiddenSizeAlign) {
  auto retMem = memset_s(outputData, dstLen * sizeof(T), 0, dstLen * sizeof(T));
  if (retMem != 0) {
    KERNEL_LOG_ERROR("TransDataRNN, GenDataNdRnnBias, memset_s failed");
    return KERNEL_STATUS_INNER_ERROR;
  }

  int32_t dstIndex = 0;
  int32_t srcIndex = 0;
  for (int32_t i = 0; i < hiddenCnt; i++) {
    for (int32_t j = 0; j < hiddenSize; j++) {
      srcIndex = i * hiddenSize + j;
      dstIndex = i * hiddenSizeAlign + j;
      outputData[dstIndex] = inputData[srcIndex];
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t TransDataRNNCpuKernel::GenDataNdRnnBias(std::vector<int64_t> &dims, int32_t hiddenSize,
                                                 const Tensor *srcTensor, Tensor *dstTensor) {
  if (dims.size() != 1) {
    KERNEL_LOG_ERROR("TransDataRNN, dst_format is ND_RNN_BIAS, rank of src shape must be 1!");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  int32_t shape0 = (int32_t)dims[0];
  if (shape0 <= 0 || shape0 % hiddenSize != 0) {
    KERNEL_LOG_ERROR("TransDataRNN, GenDataNdRnnBias, params is invalid, dim 0 is %d, hiddenSize is %d",
                     shape0, hiddenSize);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int32_t hiddenCnt = shape0 / hiddenSize;
  int32_t hiddenSizeAlign = (hiddenSize + (ALIGN_16 - 1)) / ALIGN_16 * ALIGN_16;
  int32_t dstLen = hiddenSizeAlign * hiddenCnt;

  auto srcData = srcTensor->GetData();
  KERNEL_CHECK_NULLPTR(srcData, KERNEL_STATUS_PARAM_INVALID, "TransDataRNN, get src data failed");
  auto dstData = dstTensor->GetData();
  KERNEL_CHECK_NULLPTR(dstData, KERNEL_STATUS_PARAM_INVALID, "TransDataRNN, get output data failed");

  DataType dt = static_cast<DataType>(srcTensor->GetDataType());
  uint32_t ret = KERNEL_STATUS_INNER_ERROR;
  if (dt == DT_FLOAT16) {
    Eigen::half *inputData = const_cast<Eigen::half *>(reinterpret_cast<const Eigen::half *>(srcData));
    Eigen::half *outputData = const_cast<Eigen::half *>(reinterpret_cast<const Eigen::half *>(dstData));
    ret = DealBiasData<Eigen::half>(outputData, inputData, dstLen, hiddenSize, hiddenCnt, hiddenSizeAlign);
  } else if (dt == DT_FLOAT) {
    float *inputData = const_cast<float *>(reinterpret_cast<const float *>(srcData));
    float *outputData = const_cast<float *>(reinterpret_cast<const float *>(dstData));
    ret = DealBiasData<float>(outputData, inputData, dstLen, hiddenSize, hiddenCnt, hiddenSizeAlign);
  } else {
    KERNEL_LOG_ERROR("TransDataRNN, GenDataNdRnnBias, src dtype must be float16 or float32");
    ret = KERNEL_STATUS_PARAM_INVALID;
  }
  return ret;
}

uint32_t TransDataRNNCpuKernel::GenDataFractalZnCase1(std::vector<int64_t> &dims, int32_t hiddenSize, int32_t inputSize,
                                                      const Tensor *srcTensor, Tensor *dstTensor) {
  int32_t shape1 = (int32_t)dims[1];
  int32_t hiddenCnt = shape1 / hiddenSize;
  int32_t hiddenSizeAlign = (hiddenSize + (ALIGN_16 - 1)) / ALIGN_16 * ALIGN_16;
  int32_t inputSizeAlign = (inputSize + (ALIGN_16 - 1)) / ALIGN_16 * ALIGN_16;
  int32_t dstLen = (hiddenSizeAlign * hiddenCnt) * (hiddenSizeAlign + inputSizeAlign);

  int32_t shape0Align = hiddenSizeAlign + inputSizeAlign;
  int32_t shape1Align = hiddenSizeAlign * hiddenCnt;
  int32_t newShape0 = shape0Align / 16;
  int32_t newShape1 = shape1Align / 16;

  auto srcData = srcTensor->GetData();
  KERNEL_CHECK_NULLPTR(srcData, KERNEL_STATUS_PARAM_INVALID, "GenDataFractalZnCase1, get src data failed");
  Eigen::half *inputData = const_cast<Eigen::half *>(reinterpret_cast<const Eigen::half *>(srcData));

  std::vector<std::vector<Eigen::half>> outputData(shape0Align, std::vector<Eigen::half>(shape1Align));
  for (int32_t i = 0; i < inputSize; i++) {
    for (int32_t j = 0; j < shape1; j++) {
      outputData[i][j / hiddenSize * hiddenSizeAlign + j % hiddenSize] = inputData[i * shape1 + j];
    }
  }
  for (int32_t i = 0; i < hiddenSize; i++) {
    for (int32_t j = 0; j < shape1; j++) {
      outputData[i + inputSizeAlign][j / hiddenSize * hiddenSizeAlign + j % hiddenSize] = inputData[inputSize * shape1 + i * shape1 + j];
    }
  }

  Eigen::half *dstData = const_cast<Eigen::half *>(reinterpret_cast<const Eigen::half *>(dstTensor->GetData()));
  auto ret_mem = memset_s(dstData, sizeof(Eigen::half) * dstLen, 0, sizeof(Eigen::half) * dstLen);
  if (ret_mem != 0) {
    KERNEL_LOG_ERROR("memset dstData failed, ret is [%d]", ret_mem);
    return KERNEL_STATUS_INNER_ERROR;
  }

  int32_t idx = 0;
  for (int32_t i = 0; i < newShape0; i++) {
    for (int32_t j = 0; j < newShape1; j++) {
      for (int32_t jj = 0; jj < 16; jj++) {
        for (int32_t ii = 0; ii < 16; ii++) {
          dstData[idx] = outputData[i * 16 + ii][j * 16 + jj];
          idx = idx + 1;
        }
      }
    }
  }

  return KERNEL_STATUS_OK;
}

uint32_t TransDataRNNCpuKernel::GenDataFractalZnCase2(std::vector<int64_t> &dims, int32_t hiddenSize,
                                                      const Tensor *srcTensor, Tensor *dstTensor) {
  int32_t shape0 = (int32_t)dims[0];
  int32_t shape1 = (int32_t)dims[1];
  int32_t hiddenCnt = shape1 / hiddenSize;
  int32_t hiddenSizeAlign = (hiddenSize + (ALIGN_16 - 1)) / ALIGN_16 * ALIGN_16;
  int32_t shape0Align = (shape0 + (ALIGN_16 - 1)) / ALIGN_16 * ALIGN_16;
  int32_t shape1Align = hiddenSizeAlign * hiddenCnt;
  int32_t newShape0 = shape0Align / 16;
  int32_t newShape1 = shape1Align / 16;

  int32_t dstLen = shape0Align * shape1Align;

  auto srcData = srcTensor->GetData();
  KERNEL_CHECK_NULLPTR(srcData, KERNEL_STATUS_PARAM_INVALID, "GenDataFractalZnCase2, get src data failed");
  Eigen::half *inputData = const_cast<Eigen::half *>(reinterpret_cast<const Eigen::half *>(srcData));

  std::vector<std::vector<Eigen::half>> outputData(shape0Align, std::vector<Eigen::half>(shape1Align));
  for (int32_t i = 0; i < shape0; i++) {
    for (int32_t j = 0; j < shape1; j++) {
      outputData[i][j / hiddenSize * hiddenSizeAlign + j % hiddenSize] = inputData[i * shape1 + j];
    }
  }

  Eigen::half *dstData = const_cast<Eigen::half *>(reinterpret_cast<const Eigen::half *>(dstTensor->GetData()));
  auto ret_mem = memset_s(dstData, sizeof(Eigen::half) * dstLen, 0, sizeof(Eigen::half) * dstLen);
  if (ret_mem != 0) {
    KERNEL_LOG_ERROR("memset dstData failed, ret is [%d]", ret_mem);
    return KERNEL_STATUS_INNER_ERROR;
  }

  int32_t idx = 0;
  for (int32_t i = 0; i < newShape0; i++) {
    for (int32_t j = 0; j < newShape1; j++) {
      for (int32_t jj = 0; jj < 16; jj++) {
        for (int32_t ii = 0; ii < 16; ii++) {
          dstData[idx] = outputData[i * 16 + ii][j * 16 + jj];
          idx = idx + 1;
        }
      }
    }
  }

  return KERNEL_STATUS_OK;
}

uint32_t TransDataRNNCpuKernel::GenDataFractalZn(std::vector<int64_t> &dims, int32_t hiddenSize, int32_t inputSize,
                                                 const Tensor *srcTensor, Tensor *dstTensor) {
  if (dims.size() != 2) {
    KERNEL_LOG_ERROR("TransdataRNN, dst_format is FRACTAL_ZN_RNN, rank of src shape must be 2!");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  int32_t shape0 = (int32_t)dims[0];
  int32_t shape1 = (int32_t)dims[1];
  if (shape0 <= 0 || shape1 <= 0 || shape1 % hiddenSize != 0) {
    KERNEL_LOG_ERROR("TransDataRNN, GenDataFractalZn, params is invalid, src shape is [%d,%d], hiddenSize is %d",
                     shape0, shape1, hiddenSize);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (inputSize > 0 && (shape0 == (hiddenSize + inputSize))) {
    return GenDataFractalZnCase1(dims, hiddenSize, inputSize, srcTensor, dstTensor);
  } else if ((hiddenSize > 0 && (shape0 == hiddenSize)) || (inputSize > 0 && (shape0 == inputSize))){
    return GenDataFractalZnCase2(dims, hiddenSize, srcTensor, dstTensor);
  } else {
    KERNEL_LOG_ERROR("TransDataRNN, GenDataFractalZn, params is invalid, dim0 is %d, hiddenSize is %d, inputSize is %d",
                     shape0, hiddenSize, inputSize);
  }
  KERNEL_LOG_ERROR("TransDataRNN GenDataFractalZn failed.");
  return KERNEL_STATUS_PARAM_INVALID;
}

uint32_t TransDataRNNCpuKernel::GetInputAttrs(CpuKernelContext &ctx, int32_t &inputSize, int32_t &hiddenSize,
                                              std::string &srcFormat, std::string &dstFormat) {
  AttrValue *input_size = ctx.GetAttr("input_size");
  KERNEL_CHECK_NULLPTR(input_size, KERNEL_STATUS_PARAM_INVALID, "get input_size failed.");
  inputSize = input_size->GetInt();

  AttrValue *hidden_size = ctx.GetAttr("hidden_size");
  KERNEL_CHECK_NULLPTR(hidden_size, KERNEL_STATUS_PARAM_INVALID, "get hidden_size failed.");
  hiddenSize = hidden_size->GetInt();

  AttrValue *src_format = ctx.GetAttr("src_format");
  KERNEL_CHECK_NULLPTR(src_format, KERNEL_STATUS_PARAM_INVALID, "get src_format failed.");
  srcFormat = src_format->GetString();

  AttrValue *dst_format = ctx.GetAttr("dst_format");
  KERNEL_CHECK_NULLPTR(dst_format, KERNEL_STATUS_PARAM_INVALID, "get dst_format failed.");
  dstFormat = dst_format->GetString();

  if (srcFormat != "ND") {
    KERNEL_LOG_ERROR("TransDataRNN, src_format is invalid.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (dstFormat != "ND_RNN_BIAS" && dstFormat != "FRACTAL_ZN_RNN") {
    KERNEL_LOG_ERROR("TransDataRNN, dst_format is invalid.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (hiddenSize <= 0) {
    KERNEL_LOG_ERROR("TransDataRNN, hidden_size is invalid.must be greater than 0");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t TransDataRNNCpuKernel::Compute(CpuKernelContext &ctx) {
  // get attr
  int32_t inputSize = 0;
  int32_t hiddenSize = 0;
  std::string srcFormat = "";
  std::string dstFormat = "";
  uint32_t ret = KERNEL_STATUS_PARAM_INVALID;
  if ((ret = GetInputAttrs(ctx, inputSize, hiddenSize, srcFormat, dstFormat)) != KERNEL_STATUS_OK) {
    return ret;
  }
  KERNEL_LOG_INFO("AICPU TransDataRNN is begin. hiddenSize = %d, inputSize = %d", hiddenSize, inputSize);
  Tensor *srcTensor = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(srcTensor, KERNEL_STATUS_PARAM_INVALID, "%s get src tensor failed", TRANS_DATA_RNN);
  Tensor *dstTensor = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(dstTensor, KERNEL_STATUS_PARAM_INVALID, "%s get src tensor failed", TRANS_DATA_RNN);

  auto srcShape = srcTensor->GetTensorShape();
  KERNEL_CHECK_NULLPTR(srcShape, KERNEL_STATUS_PARAM_INVALID, "%s get src shape failed", TRANS_DATA_RNN);
  std::vector<int64_t> inputDims = srcShape->GetDimSizes();

  DataType srcDtype = static_cast<DataType>(srcTensor->GetDataType());
  DataType dstDtype = static_cast<DataType>(dstTensor->GetDataType());
  if (dstFormat == "FRACTAL_ZN_RNN" && (srcDtype != DT_FLOAT16 || dstDtype != DT_FLOAT16)) {
    KERNEL_LOG_ERROR("TransdataRNN, dst format FRACTAL_ZN_RNN src and dst dtype must be float16");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (dstFormat == "ND_RNN_BIAS") {
    ret = GenDataNdRnnBias(inputDims, hiddenSize, srcTensor, dstTensor);
  } else if (dstFormat == "FRACTAL_ZN_RNN") {
    ret = GenDataFractalZn(inputDims, hiddenSize, inputSize, srcTensor, dstTensor);
  } else {
    KERNEL_LOG_ERROR("TransDataRNN does not support this dst_format");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (ret != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("TransDataRNN, process failed");
    return ret;
  }
  KERNEL_LOG_INFO("AICPU TransDataRNN run success");
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(TRANS_DATA_RNN, TransDataRNNCpuKernel);

}  // namespace aicpu