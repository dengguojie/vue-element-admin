/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

/*!
 * \file audio_ops.cpp
 * \brief
 */
#include "inc/audio_ops.h"
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "util/error_util.h"

namespace ge {
IMPLEMT_INFERFUNC(Mfcc, MfccInfer) {
  Shape unused;
  if (WithRank(op.GetInputDesc(0), 3, unused, op.GetName().c_str())
      != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(op.GetInputDesc(0).GetShape().GetDims()), "3D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 0, unused, op.GetName().c_str())
      != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
        DebugString(op.GetInputDesc(1).GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  DataType sample_rate_dtype = op.GetInputDesc(1).GetDataType();
  if (sample_rate_dtype != DT_INT32) {
    std::string err_msg = ConcatString("invalid data type", "[" , DTypeStr(sample_rate_dtype) ,"]", " of 1st input[sample_rate], it must be equal to int32");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t channel_count;
  op.GetAttr("filterbank_channel_count", channel_count);
  int64_t output_channels;
  op.GetAttr("dct_coefficient_count", output_channels);
  if (output_channels <= 0) {
    std::string err_msg = ConcatString("invalid value", "[" , output_channels ,"]", " of attr[dct_coefficient_count], it should be greater than 0");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (channel_count < output_channels) {
    std::string err_msg = ConcatString("attr[dct_coefficient_count] must be greater than attr[filterbank_channel_count] , ", "[" , channel_count ,"] and ", "[" , output_channels ,"]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  auto spectrogram_shape = op.GetInputDesc(0).GetShape().GetDims();
  if (spectrogram_shape.size() < 2) {
    return GRAPH_FAILED;
  }
  int64_t spectrogram_channels = spectrogram_shape[0];
  int64_t spectrogram_length = spectrogram_shape[1];

  vector<int64_t> y_shape({spectrogram_channels, spectrogram_length, output_channels});
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(Shape(y_shape));
  y_desc.SetDataType(DT_FLOAT);
  op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Mfcc, MfccInfer);

static int32_t GetLog2Floor(uint32_t n) {
  if (n == 0) {
    return -1;
  }
  int32_t log = 0;
  uint32_t value = n;
  // 4: shift for calc Log2Floor
  for (int32_t i = 4; i >= 0; --i) {
    int32_t shift = (1 << static_cast<uint64_t>(i));
    uint32_t x = value >> static_cast<uint32_t>(shift);
    if (x != 0) {
      value = x;
      log += shift;
    }
  }
  return log;
}

static int32_t GetLog2Ceiling(uint32_t n) {
  int32_t floor = GetLog2Floor(n);
  if (n == (n & ~(n - 1))) {  // zero or a power of two
    return floor;
  } else {
    return floor + 1;
  }
}

static uint32_t CalcNextPowerOfTwo(uint32_t value) {
  int32_t exponent = GetLog2Ceiling(value);
  return 1 << static_cast<uint32_t>(exponent);
}

IMPLEMT_INFERFUNC(AudioSpectrogram, AudioSpectrogramInfer) {
  Shape input;
  auto tensor = op.GetInputDesc(0);
  if (WithRank(tensor, 2, input, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(tensor.GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t window_size = 0;
  if (op.GetAttr("window_size", window_size) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
                                      string("get attr[window_size] failed"));
    return GRAPH_FAILED;
  }
  int64_t stride = 0;
  if (op.GetAttr("stride", stride) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
                                      string("get attr[stride] failed"));
    return GRAPH_FAILED;
  }
  if (stride == 0) {
    std::string err_msg = ConcatString("invalid value", "[" , stride ,"]", " of attr[stride], it should be not equal to 0");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  const int64_t input_length = input.GetDim(0);
  const int64_t input_channels = input.GetDim(1);

  int64_t output_length = UNKNOWN_DIM;
  if (input_length != UNKNOWN_DIM) {
    const int64_t length_minus_window = input_length - window_size;
    if (length_minus_window < 0) {
      output_length = 0;
    } else {
      output_length = 1 + (length_minus_window / stride);
      if (output_length < 0) {
        output_length = UNKNOWN_DIM;
      }
    }
  }
  int64_t output_channels = 1 + CalcNextPowerOfTwo(window_size) / 2;

  Shape out_shape({input_channels, output_length, output_channels});
  TensorDesc out_desc = op.GetOutputDesc("spectrogram");
  out_desc.SetShape(out_shape);
  out_desc.SetDataType(DT_FLOAT);
  return op.UpdateOutputDesc("spectrogram", out_desc);
}

INFER_FUNC_REG(AudioSpectrogram, AudioSpectrogramInfer);

IMPLEMT_INFERFUNC(DecodeWav, DecodeWavInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc(0), 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(op.GetInputDesc(0).GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t channels_dim = 0;
  int32_t desired_channels = 0;
  if (op.GetAttr("desired_channels", desired_channels) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), string("get attr[desired_channels] failed."));
    return GRAPH_FAILED;
  }
  if (desired_channels == -1) {
    channels_dim = ge::UNKNOWN_DIM;
  } else {
    if (desired_channels < 0) {
      std::string err_msg = ConcatString(
          "attr[desired_channels] must be non-negative, current desired_channels is [", desired_channels, "]");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }

    channels_dim = static_cast<int64_t>(desired_channels);
  }
  int64_t samples_dim;
  int32_t desired_samples;
  if (op.GetAttr("desired_samples", desired_samples) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
                                      string("get attr[desired_samples] failed."));
    return GRAPH_FAILED;
  }
  if (desired_samples == -1) {
    samples_dim = ge::UNKNOWN_DIM;
  } else {
    if (desired_samples < 0) {
      std::string err_msg = ConcatString(
          "attr[desired_samples] must be non-negative, current desired_samples is [", desired_channels, "]");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    samples_dim = static_cast<int64_t>(desired_samples);
  }

  Shape audio_shape({samples_dim, channels_dim});
  Shape sample_rate_shape;
  (void)Scalar(sample_rate_shape);
  TensorDesc audio_tensor = op.GetOutputDesc("audio");
  audio_tensor.SetDataType(DT_FLOAT);
  audio_tensor.SetShape(audio_shape);
  (void)op.UpdateOutputDesc("audio", audio_tensor);
  TensorDesc sample_rate_tensor = op.GetOutputDesc("sample_rate");
  sample_rate_tensor.SetDataType(DT_INT32);
  sample_rate_tensor.SetShape(sample_rate_shape);
  return op.UpdateOutputDesc("sample_rate", sample_rate_tensor);
}

INFER_FUNC_REG(DecodeWav, DecodeWavInfer);

IMPLEMT_INFERFUNC(EncodeWav, EncodeWavInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc(0), 2, unused_shape, op.GetName().c_str())
      != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(op.GetInputDesc(0).GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 0, unused_shape, op.GetName().c_str())
      != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
        DebugString(op.GetInputDesc(1).GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  Shape output_shape;
  (void)Scalar(output_shape);
  TensorDesc contents_tensor = op.GetOutputDesc("contents");
  contents_tensor.SetDataType(DT_STRING);
  contents_tensor.SetShape(output_shape);
  return op.UpdateOutputDesc("contents", contents_tensor);
}

INFER_FUNC_REG(EncodeWav, EncodeWavInfer);
}  // namespace ge
