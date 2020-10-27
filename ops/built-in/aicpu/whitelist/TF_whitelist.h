/**
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file TF_whitelist.h
 *
 * @brief
 *
 * @version 1.0
 *
 */


#ifndef TF_WHITELIST_H
#define TF_WHITELIST_H

REGISTER_FRAMEWORK_OP("AddSparseToTensorsMap")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AddSparseToTensorsMap")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("AdjustContrast")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AdjustContrastv2")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("AdjustHue")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AdjustHue")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("AdjustSaturation")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AdjustSaturation")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("AudioSpectrogram")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AudioSpectrogram")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("CholeskyGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("CholeskyGrad")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("CropAndResize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("CropAndResize")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("CropAndResizeGradBoxes")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("CropAndResizeGradBoxes")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("CropAndResizeGradImage")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("CropAndResizeGradImage")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("DrawBoundingBoxes")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DrawBoundingBoxes")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("DropOutGenMask")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DropOutGenMask")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ExtractGlimpse")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ExtractGlimpse")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("FixedUnigramCandidateSampler")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("FixedUnigramCandidateSampler")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("FractionalAvgPool")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("FractionalAvgPool")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("FractionalAvgPoolGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("FractionalAvgPoolGrad")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("FractionalMaxPool")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("FractionalMaxPool")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("FractionalMaxPoolGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("FractionalMaxPoolGrad")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("HSVToRGB")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("HSVToRGB")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("Igamma")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Igamma")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("Igammac")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Igammac")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("LearnedUnigramCandidateSampler")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("LearnedUnigramCandidateSampler")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("LogMatrixDeterminant")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("LogMatrixDeterminant")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("LogUniformCandidateSampler")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("LogUniformCandidateSampler")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("LowerBound")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("LowerBound")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("MatrixDeterminant")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MatrixDeterminant")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("MatrixInverse")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MatrixInverse")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("MatrixSolve")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MatrixSolve")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("MatrixSolveLs")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MatrixSolveLs")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("MatrixTriangularSolve")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MatrixTriangularSolve")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("Mfcc")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Mfcc")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("Multinomial")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Multinomial")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("NthElement")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("NthElement")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("ParameterizedTruncatedNormal")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ParameterizedTruncatedNormal")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("Qr")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Qr")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("QuantizedResizeBilinear")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("QuantizedResizeBilinear")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("RandomGamma")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RandomGamma")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("RandomGammaGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RandomGammaGrad")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("RandomPoisson")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RandomPoissonV2")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("RandomShuffle")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RandomShuffle")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("RandomStandardNormal")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RandomStandardNormal")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("RandomUniform")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RandomUniform")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("RandomUniformInt")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RandomUniformInt")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ResizeArea")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ResizeArea")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ResizeBicubic")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ResizeBicubic")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ResizeBicubicGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ResizeBicubicGrad")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ResizeNearestNeighborGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ResizeNearestNeighborGrad")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("RGBToHSV")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RGBToHSV")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("SampleDistortedBoundingBoxExt2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SampleDistortedBoundingBoxV2")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("SelfAdjointEig")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SelfAdjointEigV2")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("SparseAddGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseAddGrad")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("SparseConcat")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseConcat")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("SparseDenseCwiseAdd")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseDenseCwiseAdd")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("SparseDenseCwiseDiv")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseDenseCwiseDiv")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("SparseDenseCwiseMul")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseDenseCwiseMul")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("SparseFillEmptyRowsGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseFillEmptyRowsGrad")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("SparseReorder")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseReorder")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("SparseReshape")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseReshape")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("SparseSlice")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseSlice")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("SparseSliceGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseSliceGrad")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("SparseSoftmax")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseSoftmax")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("SparseTensorDenseAdd")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseTensorDenseAdd")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("SparseTensorDenseMatMul")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseTensorDenseMatMul")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("SparseToDense")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseToDense")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("Svd")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Svd")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ThreadUnsafeUnigramCandidateSampler")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ThreadUnsafeUnigramCandidateSampler")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("TruncatedNormal")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TruncatedNormal")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("UniformCandidateSampler")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("UniformCandidateSampler")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("StackPop")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StackPopV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("StackPush")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StackPushV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("Stack")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StackV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("StackClose")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StackCloseV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("Stage")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Stage")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("StackClear")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StackClear")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("StackPeek")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StackPeek")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("StageSize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StageSize")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("OrderedMapClear")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("OrderedMapClear")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("OrderedMapIncompleteSize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("OrderedMapIncompleteSize")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("OrderedMapPeek")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("OrderedMapPeek")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("OrderedMapSize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("OrderedMapSize")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("OrderedMapStage")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("OrderedMapStage")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("OrderedMapUnstageNoKey")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("OrderedMapUnstageNoKey")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("TensorArrayClose")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorArrayCloseV3")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("TensorArrayConcat")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorArrayConcatV3")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("TensorArrayGather")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorArrayGatherV3")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("TensorArrayGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorArrayGradV3")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("TensorArray")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorArrayV3")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("TensorArrayWrite")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorArrayWriteV3")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("DynamicPartition")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DynamicPartition")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("DynamicStitch")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DynamicStitch")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("ParallelDynamicStitch")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ParallelDynamicStitch")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("HashTable")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("HashTableV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("InitializeTable")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("InitializeTableV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("RandomShuffleQueue")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RandomShuffleQueueV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("QueueDequeueMany")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("QueueDequeueManyV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("QueueDequeueUpTo")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("QueueDequeueUpToV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("QueueDequeue")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("QueueDequeueV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("QueueEnqueueMany")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("QueueEnqueueManyV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("QueueEnqueue")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("QueueEnqueueV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("QueueIsClosed")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("QueueIsClosedV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("QueueSize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("QueueSizeV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("FIFOQueue")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("FIFOQueueV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("UniqueWithCounts")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("UniqueWithCounts")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("UniqueWithCountsExt2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("UniqueWithCountsV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("Unstage")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Unstage")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("MatrixBandPart")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MatrixBandPart")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("MirrorPad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MirrorPad")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("UnravelIndex")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("UnravelIndex")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("UpperBound")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("UpperBound")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("MirrorPadGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MirrorPadGrad")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("AllCandidateSampler")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AllCandidateSampler")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ComputeAccidentalHits")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ComputeAccidentalHits")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("CompareAndBitpack")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("CompareAndBitpack")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("Bincount")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Bincount")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("DataFormatVecPermute")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DataFormatVecPermute")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("Unique")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Unique")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("UniqueExt2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("UniqueV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("RightShift")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RightShift")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("CheckNumerics")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("CheckNumerics")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("IsFinite")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("IsFinite")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("IsInf")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("IsInf")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("RFFT")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RFFT")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("ComplexAbs")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ComplexAbs")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("IsNan")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("IsNan")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("BoostedTreesBucketize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BoostedTreesBucketize")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("InvertPermutation")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("InvertPermutation")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("MutableDenseHashTable")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MutableDenseHashTableV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("MutableHashTableOfTensors")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MutableHashTableOfTensorsV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("MutableHashTable")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MutableHashTableV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("PaddingFIFOQueue")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("PaddingFIFOQueueV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("PriorityQueue")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("PriorityQueueV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("QueueClose")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("QueueCloseV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("OrderedMapUnstage")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("OrderedMapUnstage")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("ReverseSequence")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ReverseSequence")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("NonMaxSuppressionWithOverlaps")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("NonMaxSuppressionWithOverlaps")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("MapPeek")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MapPeek")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("MapSize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MapSize")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("MapStage")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MapStage")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("MapUnstage")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MapUnstage")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("MapUnstageNoKey")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MapUnstageNoKey")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("LookupTableExport")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("LookupTableExportV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("LookupTableFind")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("LookupTableFindV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("LookupTableImport")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("LookupTableImportV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("LookupTableInsert")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("LookupTableInsertV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("LookupTableSize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("LookupTableSizeV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("MapClear")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MapClear")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("MapIncompleteSize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MapIncompleteSize")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("SparseAdd")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseAdd")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("SparseFillEmptyRows")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseFillEmptyRows")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("SparseReduceMax")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseReduceMax")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("SparseReduceMaxSparse")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseReduceMaxSparse")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("SparseReduceSum")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseReduceSum")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("SparseReduceSumSparse")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseReduceSumSparse")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("SparseSparseMaximum")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseSparseMaximum")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("SparseSparseMinimum")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseSparseMinimum")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("TensorArrayGradWithShape")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorArrayGradWithShape")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("TensorArrayRead")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorArrayReadV3")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("TensorArrayScatter")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorArrayScatterV3")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("TensorArraySplit")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorArraySplitV3")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("TensorArraySize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TensorArraySizeV3")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("DenseToSparseSetOperation")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DenseToSparseSetOperation")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("ListDiff")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ListDiff")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("Batch")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Batch")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("NonMaxSuppression")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("NonMaxSuppression")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("NonMaxSuppressionV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("NonMaxSuppressionV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("NonMaxSuppressionV3")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("NonMaxSuppressionV3")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("NonMaxSuppressionV4")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("NonMaxSuppressionV4")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("AddManySparseToTensorsMap")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AddManySparseToTensorsMap")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("SetSize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SetSize")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("Cholesky")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Cholesky")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("SparseToSparseSetOperation")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseToSparseSetOperation")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("TakeManySparseFromTensorsMap")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TakeManySparseFromTensorsMap")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("StringFormat")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StringFormat")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("StringJoin")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StringJoin")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("StringLength")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StringLength")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("StringStrip")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StringStrip")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("StringToHashBucket")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StringToHashBucket")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("StringToHashBucketStrong")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StringToHashBucketStrong")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("StringToHashBucketFast")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StringToHashBucketFast")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("UnicodeScript")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("UnicodeScript")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("Substr")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Substr")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("SparseCross")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseCross")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("DecodeWav")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DecodeWav")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("DeserializeManySparse")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DeserializeManySparse")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("DeserializeSparse")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DeserializeSparse")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("SerializeManySparse")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SerializeManySparse")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("SerializeSparse")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SerializeSparse")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("EncodeJpeg")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("EncodeJpeg")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("EncodePng")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("EncodePng")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("Betainc")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Betainc")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("Bucketize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Bucketize")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("Zeta")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Zeta")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("RegexFullMatch")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RegexFullMatch")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("RegexReplace")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RegexReplace")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("AsString")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AsString")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("EncodeBase64")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("EncodeBase64")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("DecodeBase64")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DecodeBase64")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("RecordInput")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RecordInput")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("Barrier")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Barrier")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("BarrierClose")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BarrierClose")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("BarrierIncompleteSize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BarrierIncompleteSize")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("BarrierInsertMany")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BarrierInsertMany")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("BarrierTakeMany")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BarrierTakeMany")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("BarrierReadySize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BarrierReadySize")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("StringSplit")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StringSplit")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("StringSplitV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StringSplitV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("EncodeWav")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("EncodeWav")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("Timestamp")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Timestamp")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("StatelessMultinomial")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StatelessMultinomial")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("DenseToDenseSetOperation")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DenseToDenseSetOperation")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("AccumulatorApplyGradient")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AccumulatorApplyGradient")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("AccumulatorNumAccumulated")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AccumulatorNumAccumulated")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("AccumulatorSetGlobalStep")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AccumulatorSetGlobalStep")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("AccumulatorTakeGradient")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AccumulatorTakeGradient")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("SparseAccumulatorApplyGradient")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseAccumulatorApplyGradient")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("SparseAccumulatorTakeGradient")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseAccumulatorTakeGradient")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("SparseConditionalAccumulator")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseConditionalAccumulator")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ConditionalAccumulator")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ConditionalAccumulator")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ResourceConditionalAccumulator")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ResourceConditionalAccumulator")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ResourceAccumulatorApplyGradient")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ResourceAccumulatorApplyGradient")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ResourceAccumulatorNumAccumulated")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ResourceAccumulatorNumAccumulated")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ResourceAccumulatorSetGlobalStep")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ResourceAccumulatorSetGlobalStep")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ResourceAccumulatorTakeGradient")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ResourceAccumulatorTakeGradient")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("ExtractJpegShape")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ExtractJpegShape")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("SparseSplit")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseSplit")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("Unbatch")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Unbatch")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("UnbatchGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("UnbatchGrad")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("StringToNumber")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StringToNumber")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("Print")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Print")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("PrintV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("PrintV2")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("Assert")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Assert")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("OutfeedEnqueueOp")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("OutfeedEnqueueOp")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("RandomChoiceWithMask")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RandomChoiceWithMask")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("SparseSegmentSum")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseSegmentSum")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("SparseSegmentMean")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseSegmentMean")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("SparseSegmentMeanGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseSegmentMeanGrad")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("IgammaGradA")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("IgammaGradA")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("Where")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Where")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("Assign")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Assign")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("AssignAdd")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AssignAdd")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ScatterUpdate")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ScatterUpdate")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ScatterAdd")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ScatterAdd")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ScatterSub")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ScatterSub")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ScatterMul")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ScatterMul")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ScatterDiv")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ScatterDiv")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ScatterMin")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ScatterMin")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ScatterMax")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ScatterMax")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ScatterNdUpdate")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ScatterNdUpdate")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ScatterNdAdd")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ScatterNdAdd")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ScatterNdSub")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ScatterNdSub")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("DrawBoundingBoxesV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DrawBoundingBoxesV2")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("NonMaxSuppressionV5")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("NonMaxSuppressionV5")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("NonDeterministicInts")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("NonDeterministicInts")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("StatelessRandomUniformInt")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StatelessRandomUniformInt")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ScaleAndTranslate")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ScaleAndTranslate")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ScaleAndTranslateGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ScaleAndTranslateGrad")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("CombinedNonMaxSuppression")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("CombinedNonMaxSuppression")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("RaggedGather")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RaggedGather")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("RaggedRange")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RaggedRange")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("Fingerprint")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Fingerprint")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("Lu")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Lu")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("MatrixSquareRoot")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MatrixSquareRoot")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("TridiagonalSolve")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("TridiagonalSolve")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("CTCLoss")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("CTCLoss")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("RngSkip")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RngSkip")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("StatefulRandomBinomial")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StatefulRandomBinomial")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("StatefulStandardNormalV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StatefulStandardNormalV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("StatefulTruncatedNormal")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StatefulTruncatedNormal")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("StatefulUniform")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StatefulUniform")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("StatefulUniformFullInt")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StatefulUniformFullInt")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("StatefulUniformInt")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StatefulUniformInt")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("SdcaOptimizerV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SdcaOptimizerV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("RaggedTensorToTensor")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RaggedTensorToTensor")
    .IsGray(true);
    
REGISTER_FRAMEWORK_OP("MatrixDiagPartV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MatrixDiagPartV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("MatrixSetDiagV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MatrixSetDiagV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("MatrixDiagV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MatrixDiagV2")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("Real")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Real")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("Conj")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Conj")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("AscendPadding")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AscendPadding")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("ReduceMean")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ReduceMean")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("MulNoNan")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MulNoNan")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("CTCGreedyDecoder")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("CTCGreedyDecoder")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("CTCBeamSearchDecoder")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("CTCBeamSearchDecoder")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("EditDistance")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("EditDistance")
    .IsGray(true);

REGISTER_FRAMEWORK_OP("BroadcastGradientArgs")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BroadcastGradientArgs")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("Div")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Div")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("EmbeddingRankId")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("EmbeddingRankId")
    .IsGray(false);

REGISTER_FRAMEWORK_OP("DeformableOffsets")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DeformableOffsets")
    .IsGray(false);
#endif // TF_WHITELIST_H
