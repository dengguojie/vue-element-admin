/* *
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file audio_ops.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include "inc/audio_ops.h"
#include "op_log.h"
#include "util/common_shape_fns.h"

namespace ge {
IMPLEMT_INFERFUNC(Mfcc, MfccInfer)
{
    Shape unused;
    if (WithRank(op.GetInputDesc(0), 3, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input spectrogram must be 3-D");
        return GRAPH_FAILED;
    }
    if (WithRank(op.GetInputDesc(1), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input sample_rate must be scalar");
        return GRAPH_FAILED;
    }
    DataType sample_rate_dtype = op.GetInputDesc(1).GetDataType();
    if (sample_rate_dtype != DT_INT32) {
        OP_LOGE(op.GetName().c_str(), "date type of sample_rate must be int32");
        return GRAPH_FAILED;
    }

    int64_t channel_count;
    op.GetAttr("filterbank_channel_count", channel_count);
    int64_t output_channels;
    op.GetAttr("dct_coefficient_count", output_channels);
    if (output_channels <= 0) {
        OP_LOGE(op.GetName().c_str(), "attr dct_coefficient_count must be greater than 0");
        return GRAPH_FAILED;
    }
    if (channel_count < output_channels) {
        OP_LOGE(op.GetName().c_str(), "attr filterbank_channel_count >= dct_coefficient_count is required");
        return GRAPH_FAILED;
    }

    auto spectrogram_shape = op.GetInputDesc(0).GetShape().GetDims();
    int64_t spectrogram_channels = spectrogram_shape[0];
    int64_t spectrogram_length = spectrogram_shape[1];

    vector<int64_t> y_shape({ spectrogram_channels, spectrogram_length, output_channels });
    TensorDesc y_desc = op.GetOutputDesc("y");
    y_desc.SetShape(Shape(y_shape));
    y_desc.SetDataType(DT_FLOAT);
    op.UpdateOutputDesc("y", y_desc);
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Mfcc, MfccInfer);

static int32_t GetLog2Floor(uint32_t n)
{
    if (n == 0) {
        return -1;
    }
    int32_t log = 0;
    uint32_t value = n;
    // 4: shift for calc Log2Floor
    for (int32_t i = 4; i >= 0; --i) {
        int32_t shift = (1 << i);
        uint32_t x = value >> shift;
        if (x != 0) {
            value = x;
            log += shift;
        }
    }
    return log;
}

static int32_t GetLog2Ceiling(uint32_t n)
{
    int32_t floor = GetLog2Floor(n);
    if (n == (n & ~(n - 1))) { // zero or a power of two
        return floor;
    } else {
        return floor + 1;
    }
}

static uint32_t CalcNextPowerOfTwo(uint32_t value)
{
    int32_t exponent = GetLog2Ceiling(value);
    return 1 << exponent;
}

IMPLEMT_INFERFUNC(AudioSpectrogram, AudioSpectrogramInfer)
{
    Shape input;
    auto tensor = op.GetInputDesc(0);
    if (WithRank(tensor, 2, input, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input value must be 2-D.");
        return GRAPH_FAILED;
    }

    int64_t window_size = 0;
    if (op.GetAttr("window_size", window_size) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Op AudioSpectrogram GetAttr window_size failed.");
        return GRAPH_FAILED;
    }
    int64_t stride = 0;
    if (op.GetAttr("stride", stride) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Op AudioSpectrogram GetAttr stride failed.");
        return GRAPH_FAILED;
    }
    if (stride == 0) {
        OP_LOGE(op.GetName().c_str(), "Op AudioSpectrogram attr stride value invalid, stride=%d.", stride);
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

    Shape out_shape({ input_channels, output_length, output_channels });
    TensorDesc out_desc = op.GetOutputDesc("spectrogram");
    out_desc.SetShape(out_shape);
    out_desc.SetDataType(DT_FLOAT);
    return op.UpdateOutputDesc("spectrogram", out_desc);
}

INFER_FUNC_REG(AudioSpectrogram, AudioSpectrogramInfer);

IMPLEMT_INFERFUNC(DecodeWav, DecodeWavInfer)
{
    return DecodeWavShapeFn(op);
}

INFER_FUNC_REG(DecodeWav, DecodeWavInfer);

IMPLEMT_INFERFUNC(EncodeWav, EncodeWavInfer)
{
    return EncodeWavShapeFn(op);
}

INFER_FUNC_REG(EncodeWav, EncodeWavInfer);
}   // namespace ge
