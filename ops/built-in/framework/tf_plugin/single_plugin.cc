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
 * \file single_plugin.cpp
 * \brief
 */
#include "register/register.h"

namespace domi {

// register AccumulatorApplyGradient op to GE
REGISTER_CUSTOM_OP("AccumulatorApplyGradient")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AccumulatorApplyGradient")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register AccumulatorNumAccumulated op to GE
REGISTER_CUSTOM_OP("AccumulatorNumAccumulated")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AccumulatorNumAccumulated")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register AccumulatorSetGlobalStep op to GE
REGISTER_CUSTOM_OP("AccumulatorSetGlobalStep")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AccumulatorSetGlobalStep")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register AccumulatorTakeGradient op to GE
REGISTER_CUSTOM_OP("AccumulatorTakeGradient")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AccumulatorTakeGradient")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register AddManySparseToTensorsMap op to GE
REGISTER_CUSTOM_OP("AddManySparseToTensorsMap")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AddManySparseToTensorsMap")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register AddSparseToTensorsMap op to GE
REGISTER_CUSTOM_OP("AddSparseToTensorsMap")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AddSparseToTensorsMap")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register AllCandidateSampler op to GE
REGISTER_CUSTOM_OP("AllCandidateSampler")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AllCandidateSampler")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register DecodeGif op to GE
REGISTER_CUSTOM_OP("DecodeGif")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DecodeGif")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register UnicodeEncode op to GE
REGISTER_CUSTOM_OP("UnicodeEncode")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("UnicodeEncode")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register UnicodeTranscode op to GE
REGISTER_CUSTOM_OP("UnicodeTranscode")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("UnicodeTranscode")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register UnicodeDecode op to GE
REGISTER_CUSTOM_OP("UnicodeDecode")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("UnicodeDecode")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register UnicodeDecodeWithOffsets op to GE
REGISTER_CUSTOM_OP("UnicodeDecodeWithOffsets")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("UnicodeDecodeWithOffsets")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register StringNGrams op to GE
REGISTER_CUSTOM_OP("StringNGrams")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StringNGrams")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register AsString op to GE
REGISTER_CUSTOM_OP("AsString")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AsString")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register AudioSpectrogram op to GE
REGISTER_CUSTOM_OP("AudioSpectrogram")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AudioSpectrogram")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register Barrier op to GE
REGISTER_CUSTOM_OP("Barrier")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Barrier")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register IFFT op to GE
REGISTER_CUSTOM_OP("IFFT")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("IFFT")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register BarrierClose op to GE
REGISTER_CUSTOM_OP("BarrierClose")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BarrierClose")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register BarrierIncompleteSize op to GE
REGISTER_CUSTOM_OP("BarrierIncompleteSize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BarrierIncompleteSize")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register BarrierInsertMany op to GE
REGISTER_CUSTOM_OP("BarrierInsertMany")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BarrierInsertMany")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register BarrierReadySize op to GE
REGISTER_CUSTOM_OP("BarrierReadySize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BarrierReadySize")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register Betainc op to GE
REGISTER_CUSTOM_OP("Betainc")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Betainc")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register Bucketize op to GE
REGISTER_CUSTOM_OP("Bucketize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Bucketize")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register CheckNumerics op to GE
REGISTER_CUSTOM_OP("CheckNumerics")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("CheckNumerics")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register Cholesky op to GE
REGISTER_CUSTOM_OP("Cholesky")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Cholesky")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register CholeskyGrad op to GE
REGISTER_CUSTOM_OP("CholeskyGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("CholeskyGrad")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register CompareAndBitpack op to GE
REGISTER_CUSTOM_OP("CompareAndBitpack")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("CompareAndBitpack")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register ComputeAccidentalHits op to GE
REGISTER_CUSTOM_OP("ComputeAccidentalHits")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ComputeAccidentalHits")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register ConditionalAccumulator op to GE
REGISTER_CUSTOM_OP("ConditionalAccumulator")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ConditionalAccumulator")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register CountUpTo op to GE
REGISTER_CUSTOM_OP("CountUpTo")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("CountUpTo")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register DataFormatVecPermute op to GE
REGISTER_CUSTOM_OP("DataFormatVecPermute")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DataFormatVecPermute")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register DecodeBase64 op to GE
REGISTER_CUSTOM_OP("DecodeBase64")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DecodeBase64")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register DecodeWav op to GE
REGISTER_CUSTOM_OP("DecodeWav")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DecodeWav")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register DenseToDenseSetOperation op to GE
REGISTER_CUSTOM_OP("DenseToDenseSetOperation")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DenseToDenseSetOperation")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register DenseToSparseSetOperation op to GE
REGISTER_CUSTOM_OP("DenseToSparseSetOperation")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DenseToSparseSetOperation")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register DeserializeManySparse op to GE
REGISTER_CUSTOM_OP("DeserializeManySparse")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DeserializeManySparse")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register DeserializeSparse op to GE
REGISTER_CUSTOM_OP("DeserializeSparse")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DeserializeSparse")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register DropOutGenMask op to GE
REGISTER_CUSTOM_OP("DropOutGenMask")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DropOutGenMask")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register EncodeBase64 op to GE
REGISTER_CUSTOM_OP("EncodeBase64")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("EncodeBase64")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register Angle op to GE
REGISTER_CUSTOM_OP("DecodeRaw")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DecodeRaw")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register Angle op to GE
REGISTER_CUSTOM_OP("DecodePng")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DecodePng")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register DecodeBmp op to GE
REGISTER_CUSTOM_OP("DecodeBmp")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DecodeBmp")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register DecodeAndCropJpeg op to GE
REGISTER_CUSTOM_OP("DecodeAndCropJpeg")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DecodeAndCropJpeg")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register ParseTensor op to GE
REGISTER_CUSTOM_OP("ParseTensor")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ParseTensor")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register FixedUnigramCandidateSampler op to GE
REGISTER_CUSTOM_OP("FixedUnigramCandidateSampler")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("FixedUnigramCandidateSampler")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register FractionalAvgPool op to GE
REGISTER_CUSTOM_OP("FractionalAvgPool")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("FractionalAvgPool")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register FractionalAvgPoolGrad op to GE
REGISTER_CUSTOM_OP("FractionalAvgPoolGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("FractionalAvgPoolGrad")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register FractionalMaxPool op to GE
REGISTER_CUSTOM_OP("FractionalMaxPool")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("FractionalMaxPool")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register FractionalMaxPoolGrad op to GE
REGISTER_CUSTOM_OP("FractionalMaxPoolGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("FractionalMaxPoolGrad")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register HSVToRGB op to GE
REGISTER_CUSTOM_OP("HSVToRGB")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("HSVToRGB")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register Igamma op to GE
REGISTER_CUSTOM_OP("Igamma")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Igamma")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register Igammac op to GE
REGISTER_CUSTOM_OP("Igammac")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Igammac")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register InitializeTable op to GE
REGISTER_CUSTOM_OP("InitializeTable")
    .FrameworkType(TENSORFLOW)
    .OriginOpType({"InitializeTable", "InitializeTableV2"})
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register LearnedUnigramCandidateSampler op to GE
REGISTER_CUSTOM_OP("LearnedUnigramCandidateSampler")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("LearnedUnigramCandidateSampler")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register ListDiff op to GE
REGISTER_CUSTOM_OP("ListDiff")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ListDiff")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register LogMatrixDeterminant op to GE
REGISTER_CUSTOM_OP("LogMatrixDeterminant")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("LogMatrixDeterminant")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register LogUniformCandidateSampler op to GE
REGISTER_CUSTOM_OP("LogUniformCandidateSampler")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("LogUniformCandidateSampler")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register LookupTableExport op to GE
REGISTER_CUSTOM_OP("LookupTableExport")
    .FrameworkType(TENSORFLOW)
    .OriginOpType({"LookupTableExport", "LookupTableExportV2"})
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register LookupTableImport op to GE
REGISTER_CUSTOM_OP("LookupTableImport")
    .FrameworkType(TENSORFLOW)
    .OriginOpType({"LookupTableImport", "LookupTableImportV2"})
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register LookupTableInsert op to GE
REGISTER_CUSTOM_OP("LookupTableInsert")
    .FrameworkType(TENSORFLOW)
    .OriginOpType({"LookupTableInsert", "LookupTableInsertV2"})
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register LookupTableSize op to GE
REGISTER_CUSTOM_OP("LookupTableSize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType({"LookupTableSize", "LookupTableSizeV2"})
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register LowerBound op to GE
REGISTER_CUSTOM_OP("LowerBound")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("LowerBound")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register MapClear op to GE
REGISTER_CUSTOM_OP("MapClear")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MapClear")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register MapIncompleteSize op to GE
REGISTER_CUSTOM_OP("MapIncompleteSize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MapIncompleteSize")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register MapSize op to GE
REGISTER_CUSTOM_OP("MapSize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MapSize")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register MatrixBandPart op to GE
REGISTER_CUSTOM_OP("MatrixBandPart")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MatrixBandPart")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register MatrixDeterminant op to GE
REGISTER_CUSTOM_OP("MatrixDeterminant")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MatrixDeterminant")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register MatrixInverse op to GE
REGISTER_CUSTOM_OP("MatrixInverse")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MatrixInverse")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register MatrixSolve op to GE
REGISTER_CUSTOM_OP("MatrixSolve")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MatrixSolve")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register MatrixSolveLs op to GE
REGISTER_CUSTOM_OP("MatrixSolveLs")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MatrixSolveLs")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register MatrixTriangularSolve op to GE
REGISTER_CUSTOM_OP("MatrixTriangularSolve")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MatrixTriangularSolve")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register Mfcc op to GE
REGISTER_CUSTOM_OP("Mfcc")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Mfcc")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register MirrorPad op to GE
REGISTER_CUSTOM_OP("MirrorPad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MirrorPad")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register MirrorPadGrad op to GE
REGISTER_CUSTOM_OP("MirrorPadGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MirrorPadGrad")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register MutableDenseHashTable op to GE
REGISTER_CUSTOM_OP("MutableDenseHashTable")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MutableDenseHashTableV2")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register MutableHashTableOfTensors op to GE
REGISTER_CUSTOM_OP("MutableHashTableOfTensors")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MutableHashTableOfTensorsV2")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register MutableHashTable op to GE
REGISTER_CUSTOM_OP("MutableHashTable")
    .FrameworkType(TENSORFLOW)
    .OriginOpType({"MutableHashTable", "MutableHashTableV2"})
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register NthElement op to GE
REGISTER_CUSTOM_OP("NthElement")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("NthElement")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register OrderedMapClear op to GE
REGISTER_CUSTOM_OP("OrderedMapClear")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("OrderedMapClear")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register OrderedMapIncompleteSize op to GE
REGISTER_CUSTOM_OP("OrderedMapIncompleteSize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("OrderedMapIncompleteSize")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register OrderedMapSize op to GE
REGISTER_CUSTOM_OP("OrderedMapSize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("OrderedMapSize")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register PaddingFIFOQueue op to GE
REGISTER_CUSTOM_OP("PaddingFIFOQueue")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("PaddingFIFOQueueV2")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register ParameterizedTruncatedNormal op to GE
REGISTER_CUSTOM_OP("ParameterizedTruncatedNormal")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ParameterizedTruncatedNormal")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register PrintV2 op to GE
REGISTER_CUSTOM_OP("PrintV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("PrintV2")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register PriorityQueue op to GE
REGISTER_CUSTOM_OP("PriorityQueue")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("PriorityQueueV2")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register Qr op to GE
REGISTER_CUSTOM_OP("Qr")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Qr")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register RandomPoisson op to GE
REGISTER_CUSTOM_OP("RandomPoisson")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RandomPoissonV2")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register RandomShuffle op to GE
REGISTER_CUSTOM_OP("RandomShuffle")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RandomShuffle")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register RandomStandardNormal op to GE
REGISTER_CUSTOM_OP("RandomStandardNormal")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RandomStandardNormal")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register RandomUniformInt op to GE
REGISTER_CUSTOM_OP("RandomUniformInt")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RandomUniformInt")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register RecordInput op to GE
REGISTER_CUSTOM_OP("RecordInput")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RecordInput")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register RegexFullMatch op to GE
REGISTER_CUSTOM_OP("RegexFullMatch")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RegexFullMatch")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register RegexReplace op to GE
REGISTER_CUSTOM_OP("RegexReplace")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RegexReplace")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register RGBToHSV op to GE
REGISTER_CUSTOM_OP("RGBToHSV")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RGBToHSV")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register RightShift op to GE
REGISTER_CUSTOM_OP("RightShift")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RightShift")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SampleDistortedBoundingBox op to GE
REGISTER_CUSTOM_OP("SampleDistortedBoundingBox")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SampleDistortedBoundingBox")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SampleDistortedBoundingBoxExt2 op to GE
REGISTER_CUSTOM_OP("SampleDistortedBoundingBoxExt2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SampleDistortedBoundingBoxV2")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SerializeManySparse op to GE
REGISTER_CUSTOM_OP("SerializeManySparse")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SerializeManySparse")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SerializeSparse op to GE
REGISTER_CUSTOM_OP("SerializeSparse")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SerializeSparse")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SetSize op to GE
REGISTER_CUSTOM_OP("SetSize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SetSize")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseAccumulatorApplyGradient op to GE
REGISTER_CUSTOM_OP("SparseAccumulatorApplyGradient")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseAccumulatorApplyGradient")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseAccumulatorTakeGradient op to GE
REGISTER_CUSTOM_OP("SparseAccumulatorTakeGradient")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseAccumulatorTakeGradient")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseAdd op to GE
REGISTER_CUSTOM_OP("SparseAdd")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseAdd")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseAddGrad op to GE
REGISTER_CUSTOM_OP("SparseAddGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseAddGrad")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseConditionalAccumulator op to GE
REGISTER_CUSTOM_OP("SparseConditionalAccumulator")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseConditionalAccumulator")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseDenseCwiseAdd op to GE
REGISTER_CUSTOM_OP("SparseDenseCwiseAdd")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseDenseCwiseAdd")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseDenseCwiseDiv op to GE
REGISTER_CUSTOM_OP("SparseDenseCwiseDiv")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseDenseCwiseDiv")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseDenseCwiseMul op to GE
REGISTER_CUSTOM_OP("SparseDenseCwiseMul")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseDenseCwiseMul")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseFillEmptyRows op to GE
REGISTER_CUSTOM_OP("SparseFillEmptyRows")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseFillEmptyRows")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseFillEmptyRowsGrad op to GE
REGISTER_CUSTOM_OP("SparseFillEmptyRowsGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseFillEmptyRowsGrad")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseReduceMax op to GE
REGISTER_CUSTOM_OP("SparseReduceMax")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseReduceMax")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseReduceMaxSparse op to GE
REGISTER_CUSTOM_OP("SparseReduceMaxSparse")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseReduceMaxSparse")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseReduceSum op to GE
REGISTER_CUSTOM_OP("SparseReduceSum")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseReduceSum")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseReduceSumSparse op to GE
REGISTER_CUSTOM_OP("SparseReduceSumSparse")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseReduceSumSparse")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseReorder op to GE
REGISTER_CUSTOM_OP("SparseReorder")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseReorder")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseReshape op to GE
REGISTER_CUSTOM_OP("SparseReshape")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseReshape")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseSlice op to GE
REGISTER_CUSTOM_OP("SparseSlice")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseSlice")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseSliceGrad op to GE
REGISTER_CUSTOM_OP("SparseSliceGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseSliceGrad")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register FakeQueue op to GE
REGISTER_CUSTOM_OP("FakeQueue")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("FakeQueue")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseSoftmax op to GE
REGISTER_CUSTOM_OP("SparseSoftmax")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseSoftmax")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseSparseMaximum op to GE
REGISTER_CUSTOM_OP("SparseSparseMaximum")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseSparseMaximum")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseSparseMinimum op to GE
REGISTER_CUSTOM_OP("SparseSparseMinimum")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseSparseMinimum")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseTensorDenseMatMul op to GE
REGISTER_CUSTOM_OP("SparseTensorDenseMatMul")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseTensorDenseMatMul")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseToDense op to GE
REGISTER_CUSTOM_OP("SparseToDense")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseToDense")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseToSparseSetOperation op to GE
REGISTER_CUSTOM_OP("SparseToSparseSetOperation")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseToSparseSetOperation")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register StackClose op to GE
REGISTER_CUSTOM_OP("StackClose")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StackCloseV2")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register StageClear op to GE
REGISTER_CUSTOM_OP("StageClear")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StageClear")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register StageSize op to GE
REGISTER_CUSTOM_OP("StageSize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StageSize")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register StatelessMultinomial op to GE
REGISTER_CUSTOM_OP("StatelessMultinomial")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StatelessMultinomial")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register StringLength op to GE
REGISTER_CUSTOM_OP("StringLength")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StringLength")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register StringSplit op to GE
REGISTER_CUSTOM_OP("StringSplit")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StringSplit")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register StaticRegexReplace op to GE
REGISTER_CUSTOM_OP("StaticRegexReplace")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StaticRegexReplace")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register StaticRegexFullMatch op to GE
REGISTER_CUSTOM_OP("StaticRegexFullMatch")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StaticRegexFullMatch")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register ReduceJoin op to GE
REGISTER_CUSTOM_OP("ReduceJoin")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ReduceJoin")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register UnsortedSegmentJoin op to GE
REGISTER_CUSTOM_OP("UnsortedSegmentJoin")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StaticRegexReplace")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register StringLower op to GE
REGISTER_CUSTOM_OP("StringLower")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StringLower")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register StringUpper op to GE
REGISTER_CUSTOM_OP("StringUpper")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StringUpper")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register StringSplitV2 op to GE
REGISTER_CUSTOM_OP("StringSplitV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StringSplitV2")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register StringStrip op to GE
REGISTER_CUSTOM_OP("StringStrip")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StringStrip")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register StringToHashBucket op to GE
REGISTER_CUSTOM_OP("StringToHashBucket")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StringToHashBucket")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register StringToHashBucketFast op to GE
REGISTER_CUSTOM_OP("StringToHashBucketFast")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StringToHashBucketFast")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register StringToHashBucketStrong op to GE
REGISTER_CUSTOM_OP("StringToHashBucketStrong")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StringToHashBucketStrong")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register Substr op to GE
REGISTER_CUSTOM_OP("Substr")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Substr")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register Svd op to GE
REGISTER_CUSTOM_OP("Svd")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Svd")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register TakeManySparseFromTensorsMap op to GE
REGISTER_CUSTOM_OP("TakeManySparseFromTensorsMap")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TakeManySparseFromTensorsMap")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register TensorArrayClose op to GE
REGISTER_CUSTOM_OP("TensorArrayClose")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorArrayCloseV3")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register TensorArrayConcat op to GE
REGISTER_CUSTOM_OP("TensorArrayConcat")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorArrayConcatV3")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register TensorArrayGradWithShape op to GE
REGISTER_CUSTOM_OP("TensorArrayGradWithShape")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorArrayGradWithShape")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register TensorArraySplit op to GE
REGISTER_CUSTOM_OP("TensorArraySplit")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorArraySplitV3")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register ThreadUnsafeUnigramCandidateSampler op to GE
REGISTER_CUSTOM_OP("ThreadUnsafeUnigramCandidateSampler")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ThreadUnsafeUnigramCandidateSampler")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register Timestamp op to GE
REGISTER_CUSTOM_OP("Timestamp")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Timestamp")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register TruncatedNormal op to GE
REGISTER_CUSTOM_OP("TruncatedNormal")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TruncatedNormal")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register Unbatch op to GE
REGISTER_CUSTOM_OP("Unbatch")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Unbatch")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register UnbatchGrad op to GE
REGISTER_CUSTOM_OP("UnbatchGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("UnbatchGrad")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register FusedBatchNormV2 op to GE
REGISTER_CUSTOM_OP("FusedBatchNormV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("FusedBatchNormV2")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register UnicodeScript op to GE
REGISTER_CUSTOM_OP("UnicodeScript")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("UnicodeScript")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register UniformCandidateSampler op to GE
REGISTER_CUSTOM_OP("UniformCandidateSampler")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("UniformCandidateSampler")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register Unique op to GE
REGISTER_CUSTOM_OP("Unique")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Unique")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register UniqueExt2 op to GE
REGISTER_CUSTOM_OP("UniqueExt2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("UniqueV2")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register UniqueWithCounts op to GE
REGISTER_CUSTOM_OP("UniqueWithCounts")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("UniqueWithCounts")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register UniqueWithCountsExt2 op to GE
REGISTER_CUSTOM_OP("UniqueWithCountsExt2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("UniqueWithCountsV2")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register UnravelIndex op to GE
REGISTER_CUSTOM_OP("UnravelIndex")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("UnravelIndex")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register UpperBound op to GE
REGISTER_CUSTOM_OP("UpperBound")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("UpperBound")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register Zeta op to GE
REGISTER_CUSTOM_OP("Zeta")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Zeta")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register DrawBoundingBoxesV2 op to GE
REGISTER_CUSTOM_OP("DrawBoundingBoxesV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DrawBoundingBoxesV2")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register NonMaxSuppressionV5 op to GE
REGISTER_CUSTOM_OP("NonMaxSuppressionV5")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("NonMaxSuppressionV5")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register NonDeterministicInts op to GE
REGISTER_CUSTOM_OP("NonDeterministicInts")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("NonDeterministicInts")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register StatelessRandomUniformInt op to GE
REGISTER_CUSTOM_OP("StatelessRandomUniformInt")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StatelessRandomUniformInt")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register CombinedNonMaxSuppression op to GE
REGISTER_CUSTOM_OP("CombinedNonMaxSuppression")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("CombinedNonMaxSuppression")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::TVM);

// register Fingerprint op to GE
REGISTER_CUSTOM_OP("Fingerprint")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Fingerprint")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register Lu op to GE
REGISTER_CUSTOM_OP("Lu")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Lu")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register MatrixSquareRoot op to GE
REGISTER_CUSTOM_OP("MatrixSquareRoot")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MatrixSquareRoot")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register TridiagonalSolve op to GE
REGISTER_CUSTOM_OP("TridiagonalSolve")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TridiagonalSolve")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register CTCLoss op to GE
REGISTER_CUSTOM_OP("CTCLoss")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("CTCLoss")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register RngSkip op to GE
REGISTER_CUSTOM_OP("RngSkip")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RngSkip")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register StatefulRandomBinomial op to GE
REGISTER_CUSTOM_OP("StatefulRandomBinomial")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StatefulRandomBinomial")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register StatefulStandardNormalV2 op to GE
REGISTER_CUSTOM_OP("StatefulStandardNormalV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StatefulStandardNormalV2")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register StatefulTruncatedNormal op to GE
REGISTER_CUSTOM_OP("StatefulTruncatedNormal")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StatefulTruncatedNormal")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register StatefulUniform op to GE
REGISTER_CUSTOM_OP("StatefulUniform")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StatefulUniform")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register StatefulUniformFullInt op to GE
REGISTER_CUSTOM_OP("StatefulUniformFullInt")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StatefulUniformFullInt")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register StatefulUniformInt op to GE
REGISTER_CUSTOM_OP("StatefulUniformInt")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StatefulUniformInt")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register RaggedRange op to GE
REGISTER_CUSTOM_OP("RaggedRange")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RaggedRange")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register NextAfter op to GE
REGISTER_CUSTOM_OP("NextAfter")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("NextAfter")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register ResourceConditionalAccumulator op to GE
REGISTER_CUSTOM_OP("ResourceConditionalAccumulator")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ResourceConditionalAccumulator")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register ResourceAccumulatorApplyGradient op to GE
REGISTER_CUSTOM_OP("ResourceAccumulatorApplyGradient")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ResourceAccumulatorApplyGradient")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register ResourceAccumulatorNumAccumulated op to GE
REGISTER_CUSTOM_OP("ResourceAccumulatorNumAccumulated")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ResourceAccumulatorNumAccumulated")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register ResourceAccumulatorSetGlobalStep op to GE
REGISTER_CUSTOM_OP("ResourceAccumulatorSetGlobalStep")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ResourceAccumulatorSetGlobalStep")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register ResourceAccumulatorTakeGradient op to GE
REGISTER_CUSTOM_OP("ResourceAccumulatorTakeGradient")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ResourceAccumulatorTakeGradient")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register IsFinite op to GE
REGISTER_CUSTOM_OP("IsFinite")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("IsFinite")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register IsInf op to GE
REGISTER_CUSTOM_OP("IsInf")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("IsInf")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register RFFT op to GE
REGISTER_CUSTOM_OP("RFFT")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RFFT")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register ComplexAbs op to GE
REGISTER_CUSTOM_OP("ComplexAbs")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ComplexAbs")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register IsNan op to GE
REGISTER_CUSTOM_OP("IsNan")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("IsNan")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register MatrixDiagPartV2 op to GE
REGISTER_CUSTOM_OP("MatrixDiagPartV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MatrixDiagPartV2")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register MatrixSetDiagV2 op to GE
REGISTER_CUSTOM_OP("MatrixSetDiagV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MatrixSetDiagV2")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register MatrixDiagV2 op to GE
REGISTER_CUSTOM_OP("MatrixDiagV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MatrixDiagV2")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register BroadcastGradientArgs op to GE
REGISTER_CUSTOM_OP("BroadcastGradientArgs")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BroadcastGradientArgs")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register Real op to GE
REGISTER_CUSTOM_OP("Real")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Real")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register Conj op to GE
REGISTER_CUSTOM_OP("Conj")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Conj")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register CTCGreedyDecoder op to GE
REGISTER_CUSTOM_OP("CTCGreedyDecoder")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("CTCGreedyDecoder")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register EditDistance op to GE
REGISTER_CUSTOM_OP("EditDistance")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("EditDistance")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register AscendPadding op to GE
REGISTER_CUSTOM_OP("AscendPadding")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AscendPadding")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

REGISTER_CUSTOM_OP("EmbeddingRankId")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("EmbeddingRankId")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseSegmentMeanGrad op to GE
REGISTER_CUSTOM_OP("SparseSegmentMeanGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseSegmentMeanGrad")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseSegmentMean op to GE
REGISTER_CUSTOM_OP("SparseSegmentMean")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseSegmentMean")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register SparseSegmentSum op to GE
REGISTER_CUSTOM_OP("SparseSegmentSum")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseSegmentSum")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register DeformableOffsets op to GE
REGISTER_CUSTOM_OP("DeformableOffsets")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DeformableOffsets")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register DeformableOffsetsGrad op to GE
REGISTER_CUSTOM_OP("DeformableOffsetsGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DeformableOffsetsGrad")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register DenseImageWarpGrad op to GE
REGISTER_CUSTOM_OP("DenseImageWarpGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DenseImageWarpGrad")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register Complex op to GE
REGISTER_CUSTOM_OP("Complex")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Complex")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register Imag op to GE
REGISTER_CUSTOM_OP("Imag")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Imag")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register Angle op to GE
REGISTER_CUSTOM_OP("Angle")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Angle")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register FFT op to GE
REGISTER_CUSTOM_OP("FFT")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("FFT")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register IFFT2D op to GE
REGISTER_CUSTOM_OP("IFFT2D")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("IFFT2D")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register IRFFT op to GE
REGISTER_CUSTOM_OP("IRFFT")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("IRFFT")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

// register FFT2D op to GE
REGISTER_CUSTOM_OP("FFT2D")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("FFT2D")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);

}  // namespace domi
