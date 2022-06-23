/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file test_transpose_tiling.cpp
 * \brief
 */
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <gtest/gtest.h>
#define private public
#include <register/op_tiling.h>
#include "runtime/transpose.h"
#include "op_tiling/op_tiling_util.h"
#include "transformation_ops.h"
#include "array_ops.h"
#include "test_common.h"
#include "common/utils/ut_op_util.h"
#include "common_unittest.h"

using namespace std;
using namespace optiling;

string opTypeV2 = "transpose";

class TransposeTilingTestV2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    // std::cout << "TransposeTilingTestV2 setup\n";
  }
  static void TearDownTestCase() {
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "2", 1);
    // std::cout << "TransposeTilingTestV2 teardown";
  }
};

TEST_F(TransposeTilingTestV2, reduce_axis_merge) {
  /*
   *  inShape         perm                   reducedInShape       reducedOutShape       reducedPerm
   *  ---------------------------------------------------------------------------------------------------
   *  4,5,6,7         1,0,2,3                4,5,42               5,4,42                1,0,2
   *
   */
  ShapeInfo shapeInfo1;
  shapeInfo1.origDim = 4;
  shapeInfo1.dim = 4;
  shapeInfo1.reducedInShape[0] = 4;
  shapeInfo1.reducedInShape[1] = 5;
  shapeInfo1.reducedInShape[2] = 6;
  shapeInfo1.reducedInShape[3] = 7;
  shapeInfo1.reducedPerm[0] = 1;
  shapeInfo1.reducedPerm[1] = 0;
  shapeInfo1.reducedPerm[2] = 2;
  shapeInfo1.reducedPerm[3] = 3;
  MergeAxisV2(shapeInfo1);
  EXPECT_EQ(shapeInfo1.dim, 3);
  EXPECT_EQ(shapeInfo1.reducedInShape[0], 4);
  EXPECT_EQ(shapeInfo1.reducedInShape[1], 5);
  EXPECT_EQ(shapeInfo1.reducedInShape[2], 42);
  EXPECT_EQ(shapeInfo1.reducedOutShape[0], 5);
  EXPECT_EQ(shapeInfo1.reducedOutShape[1], 4);
  EXPECT_EQ(shapeInfo1.reducedOutShape[2], 42);
  EXPECT_EQ(shapeInfo1.reducedPerm[0], 1);
  EXPECT_EQ(shapeInfo1.reducedPerm[1], 0);
  EXPECT_EQ(shapeInfo1.reducedPerm[2], 2);

  /*
   *  inShape         perm                   reducedInShape       reducedOutShape       reducedPerm
   *  ---------------------------------------------------------------------------------------------------
   *  2,3,4,5,6       0,4,1,2,3              2,60,6               2,6,60                0,2,1
   *
   */
  shapeInfo1.Reset();
  shapeInfo1.origDim = 5;
  shapeInfo1.dim = 5;
  shapeInfo1.reducedInShape[0] = 2;
  shapeInfo1.reducedInShape[1] = 3;
  shapeInfo1.reducedInShape[2] = 4;
  shapeInfo1.reducedInShape[3] = 5;
  shapeInfo1.reducedInShape[4] = 6;
  shapeInfo1.reducedPerm[0] = 0;
  shapeInfo1.reducedPerm[1] = 4;
  shapeInfo1.reducedPerm[2] = 1;
  shapeInfo1.reducedPerm[3] = 2;
  shapeInfo1.reducedPerm[4] = 3;
  MergeAxisV2(shapeInfo1);
  EXPECT_EQ(shapeInfo1.dim, 3);
  EXPECT_EQ(shapeInfo1.reducedInShape[0], 2);
  EXPECT_EQ(shapeInfo1.reducedInShape[1], 60);
  EXPECT_EQ(shapeInfo1.reducedInShape[2], 6);
  EXPECT_EQ(shapeInfo1.reducedOutShape[0], 2);
  EXPECT_EQ(shapeInfo1.reducedOutShape[1], 6);
  EXPECT_EQ(shapeInfo1.reducedOutShape[2], 60);
  EXPECT_EQ(shapeInfo1.reducedPerm[0], 0);
  EXPECT_EQ(shapeInfo1.reducedPerm[1], 2);
  EXPECT_EQ(shapeInfo1.reducedPerm[2], 1);

  /*
   *  inShape         perm                   reducedInShape       reducedOutShape       reducedPerm
   *  ---------------------------------------------------------------------------------------------------
   *  2,3,4,5,6       2,3,4,0,1              6,120                120,6                 1,0
   *
   */
  shapeInfo1.Reset();
  shapeInfo1.origDim = 5;
  shapeInfo1.dim = 5;
  shapeInfo1.reducedInShape[0] = 2;
  shapeInfo1.reducedInShape[1] = 3;
  shapeInfo1.reducedInShape[2] = 4;
  shapeInfo1.reducedInShape[3] = 5;
  shapeInfo1.reducedInShape[4] = 6;
  shapeInfo1.reducedPerm[0] = 2;
  shapeInfo1.reducedPerm[1] = 3;
  shapeInfo1.reducedPerm[2] = 4;
  shapeInfo1.reducedPerm[3] = 0;
  shapeInfo1.reducedPerm[4] = 1;
  MergeAxisV2(shapeInfo1);
  EXPECT_EQ(shapeInfo1.dim, 2);
  EXPECT_EQ(shapeInfo1.reducedInShape[0], 6);
  EXPECT_EQ(shapeInfo1.reducedInShape[1], 120);
  EXPECT_EQ(shapeInfo1.reducedOutShape[0], 120);
  EXPECT_EQ(shapeInfo1.reducedOutShape[1], 6);
  EXPECT_EQ(shapeInfo1.reducedPerm[0], 1);
  EXPECT_EQ(shapeInfo1.reducedPerm[1], 0);
}

TEST_F(TransposeTilingTestV2, reduce_axis_remove) {
  /*
   *  inShape         perm                   reducedInShape       reducedOutShape       reducedPerm
   *  ---------------------------------------------------------------------------------------------------
   *  4,1,6,1         0,1,2,3                4,6                  4,6                   0,1
   *
   */
  ShapeInfo shapeInfo1;
  shapeInfo1.origDim = 4;
  shapeInfo1.dim = 4;
  shapeInfo1.inShape[0] = 4;
  shapeInfo1.inShape[1] = 1;
  shapeInfo1.inShape[2] = 6;
  shapeInfo1.inShape[3] = 1;
  shapeInfo1.perm[0] = 0;
  shapeInfo1.perm[1] = 1;
  shapeInfo1.perm[2] = 2;
  shapeInfo1.perm[3] = 3;
  RemoveAxisV2(shapeInfo1);
  EXPECT_EQ(shapeInfo1.dim, 2);
  EXPECT_EQ(shapeInfo1.reducedInShape[0], 4);
  EXPECT_EQ(shapeInfo1.reducedInShape[1], 6);
  EXPECT_EQ(shapeInfo1.reducedOutShape[0], 4);
  EXPECT_EQ(shapeInfo1.reducedOutShape[1], 6);
  EXPECT_EQ(shapeInfo1.reducedPerm[0], 0);
  EXPECT_EQ(shapeInfo1.reducedPerm[1], 1);

  /*
   *  inShape         perm                   reducedInShape       reducedOutShape       reducedPerm
   *  ---------------------------------------------------------------------------------------------------
   *  4,1,6,1         1,2,0,3                4,6                  6,4                   1,0
   *
   */
  shapeInfo1.Reset();
  shapeInfo1.origDim = 4;
  shapeInfo1.dim = 4;
  shapeInfo1.inShape[0] = 4;
  shapeInfo1.inShape[1] = 1;
  shapeInfo1.inShape[2] = 6;
  shapeInfo1.inShape[3] = 1;
  shapeInfo1.perm[0] = 1;
  shapeInfo1.perm[1] = 2;
  shapeInfo1.perm[2] = 0;
  shapeInfo1.perm[3] = 3;
  RemoveAxisV2(shapeInfo1);
  EXPECT_EQ(shapeInfo1.dim, 2);
  EXPECT_EQ(shapeInfo1.reducedInShape[0], 4);
  EXPECT_EQ(shapeInfo1.reducedInShape[1], 6);
  EXPECT_EQ(shapeInfo1.reducedOutShape[0], 6);
  EXPECT_EQ(shapeInfo1.reducedOutShape[1], 4);
  EXPECT_EQ(shapeInfo1.reducedPerm[0], 1);
  EXPECT_EQ(shapeInfo1.reducedPerm[1], 0);

  /*
   *  inShape         perm                   reducedInShape       reducedOutShape       reducedPerm
   *  ---------------------------------------------------------------------------------------------------
   *  1,1,4,6         3,2,0,1                4,6                  6,4                   1,0
   *
   */
  shapeInfo1.Reset();
  shapeInfo1.origDim = 4;
  shapeInfo1.dim = 4;
  shapeInfo1.inShape[0] = 1;
  shapeInfo1.inShape[1] = 1;
  shapeInfo1.inShape[2] = 4;
  shapeInfo1.inShape[3] = 6;
  shapeInfo1.perm[0] = 3;
  shapeInfo1.perm[1] = 2;
  shapeInfo1.perm[2] = 0;
  shapeInfo1.perm[3] = 1;
  RemoveAxisV2(shapeInfo1);
  EXPECT_EQ(shapeInfo1.dim, 2);
  EXPECT_EQ(shapeInfo1.reducedInShape[0], 4);
  EXPECT_EQ(shapeInfo1.reducedInShape[1], 6);
  EXPECT_EQ(shapeInfo1.reducedOutShape[0], 6);
  EXPECT_EQ(shapeInfo1.reducedOutShape[1], 4);
  EXPECT_EQ(shapeInfo1.reducedPerm[0], 1);
  EXPECT_EQ(shapeInfo1.reducedPerm[1], 0);

  /*
   *  inShape         perm                   reducedInShape       reducedOutShape       reducedPerm
   *  ---------------------------------------------------------------------------------------------------
   *  4,6,1,1         3,2,0,1                4,6                  4,6                   0,1
   *
   */
  shapeInfo1.Reset();
  shapeInfo1.origDim = 4;
  shapeInfo1.dim = 4;
  shapeInfo1.inShape[0] = 4;
  shapeInfo1.inShape[1] = 6;
  shapeInfo1.inShape[2] = 1;
  shapeInfo1.inShape[3] = 1;
  shapeInfo1.perm[0] = 3;
  shapeInfo1.perm[1] = 2;
  shapeInfo1.perm[2] = 0;
  shapeInfo1.perm[3] = 1;
  RemoveAxisV2(shapeInfo1);
  EXPECT_EQ(shapeInfo1.dim, 2);
  EXPECT_EQ(shapeInfo1.reducedInShape[0], 4);
  EXPECT_EQ(shapeInfo1.reducedInShape[1], 6);
  EXPECT_EQ(shapeInfo1.reducedOutShape[0], 4);
  EXPECT_EQ(shapeInfo1.reducedOutShape[1], 6);
  EXPECT_EQ(shapeInfo1.reducedPerm[0], 0);
  EXPECT_EQ(shapeInfo1.reducedPerm[1], 1);
}

TEST_F(TransposeTilingTestV2, stride_lt_65535) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 3;
  shapeInfo.dim = 3;
  shapeInfo.inShape[0] = 7000;
  shapeInfo.inShape[1] = 32768;
  shapeInfo.inShape[2] = 8;
  shapeInfo.outShape[0] = 32768;
  shapeInfo.outShape[1] = 7000;
  shapeInfo.outShape[2] = 8;
  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 0;
  shapeInfo.perm[2] = 2;

  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_TRUE(TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo));
}

TEST_F(TransposeTilingTestV2, identical_shape) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 2;
  shapeInfo.dim = 2;
  shapeInfo.inShape[0] = 1000;
  shapeInfo.inShape[1] = 2000;
  shapeInfo.outShape[0] = 1000;
  shapeInfo.outShape[1] = 2000;
  shapeInfo.perm[0] = 0;
  shapeInfo.perm[1] = 1;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_TRUE(TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo));
  EXPECT_EQ(shapeInfo.scenario, 0);
}

TEST_F(TransposeTilingTestV2, small_shape) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 2;
  shapeInfo.dim = 2;
  shapeInfo.inShape[0] = 10;
  shapeInfo.inShape[1] = 20;
  shapeInfo.outShape[0] = 20;
  shapeInfo.outShape[1] = 10;
  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 0;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_TRUE(TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo));
  EXPECT_EQ(shapeInfo.scenario, 6);
}

TEST_F(TransposeTilingTestV2, stride_gt_65535) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 3;
  shapeInfo.dim = 3;
  shapeInfo.inShape[0] = 7000;
  shapeInfo.inShape[1] = 32768;
  shapeInfo.inShape[2] = 16;
  shapeInfo.outShape[0] = 32768;
  shapeInfo.outShape[1] = 7000;
  shapeInfo.outShape[2] = 16;
  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 0;
  shapeInfo.perm[2] = 2;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_EQ(shapeInfo.scenario, 5);
  EXPECT_TRUE(TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo));
}

TEST_F(TransposeTilingTestV2, huge_last_axis) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 3;
  shapeInfo.dim = 3;
  shapeInfo.inShape[0] = 7000;
  shapeInfo.inShape[1] = 32768;
  shapeInfo.inShape[2] = 160000;
  shapeInfo.outShape[0] = 32768;
  shapeInfo.outShape[1] = 7000;
  shapeInfo.outShape[2] = 160000;
  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 0;
  shapeInfo.perm[2] = 2;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_EQ(shapeInfo.scenario, 3);
  EXPECT_TRUE(TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo));
}

TEST_F(TransposeTilingTestV2, last_axis_join_transpose_2d) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 2;
  shapeInfo.dim = 2;
  shapeInfo.inShape[0] = 7000;
  shapeInfo.inShape[1] = 6000;
  shapeInfo.outShape[0] = 6000;
  shapeInfo.outShape[1] = 7000;
  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 0;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_EQ(shapeInfo.scenario, 11);
}

TEST_F(TransposeTilingTestV2, last_axis_join_transpose_with_multi_src_axis) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 4;
  shapeInfo.dim = 4;
  shapeInfo.inShape[0]=4;
  shapeInfo.inShape[1]=128;
  shapeInfo.inShape[2]=8;
  shapeInfo.inShape[3]=224;

  shapeInfo.outShape[0] =224;
  shapeInfo.outShape[1] =8;
  shapeInfo.outShape[2] =128;
  shapeInfo.outShape[3] =4;

  shapeInfo.perm[0] = 3;
  shapeInfo.perm[1] = 2;
  shapeInfo.perm[2] = 1;
  shapeInfo.perm[3] = 0;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);

  EXPECT_EQ(runtimeInfo.dstJumpStride[0], 4096);
  EXPECT_EQ(runtimeInfo.srcJumpAxisNum, 3);

  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoCol.colPerMC, 128);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoCol.loopOnMC, 1);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoCol.colTC, 96);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoCol.colOffset, 0);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoCol.backStepLeft, 0);

  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoRow.rowPerMR, 128);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoRow.loopOnMR, 1);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoRow.rowTR, 0);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoRow.rowOffset, 0);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoRow.backStepUp, 0);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoRow.initSrcTuple[1], 0);

  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.colPerMC, 128);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.loopOnMC, 1);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.colTC, 96);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.colOffset, 0);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.backStepLeft, 0);

  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.rowPerMR, 128);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.loopOnMR, 1);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.rowTR, 0);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.rowOffset, 128);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.backStepUp, 0);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.initSrcTuple[1], 32);
}

TEST_F(TransposeTilingTestV2, last_axis_join_transpose_with_tail_row) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 4;
  shapeInfo.dim = 4;
  shapeInfo.inShape[0] = 5;
  shapeInfo.inShape[1] = 128;
  shapeInfo.inShape[2] = 8;
  shapeInfo.inShape[3] = 224;

  shapeInfo.outShape[0] =224;
  shapeInfo.outShape[1] =8;
  shapeInfo.outShape[2] =128;
  shapeInfo.outShape[3] =5;

  shapeInfo.perm[0] = 3;
  shapeInfo.perm[1] = 2;
  shapeInfo.perm[2] = 1;
  shapeInfo.perm[3] = 0;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);

  EXPECT_EQ(runtimeInfo.dstJumpStride[0], 5120);
  EXPECT_EQ(runtimeInfo.srcJumpAxisNum, 3);

  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoCol.colPerMC, 128);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoCol.loopOnMC, 1);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoCol.colTC, 96);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoCol.colOffset, 0);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoCol.backStepLeft, 0);

  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoRow.rowPerMR, 128);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoRow.loopOnMR, 1);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoRow.rowTR, 32);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoRow.rowOffset, 0);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoRow.backStepUp, 0);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoRow.initSrcTuple[1], 0);

  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.colPerMC, 128);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.loopOnMC, 1);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.colTC, 96);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.colOffset, 0);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.backStepLeft, 0);

  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.rowPerMR, 128);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.loopOnMR, 1);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.rowTR, 32);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.rowOffset, 160);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.backStepUp, 0);

  EXPECT_EQ(runtimeInfo.infoPerCore[31].infoRow.initSrcTuple[0], 0);
  EXPECT_EQ(runtimeInfo.infoPerCore[31].infoRow.initSrcTuple[1], 96);
  EXPECT_EQ(runtimeInfo.infoPerCore[31].infoRow.initSrcTuple[2], 7);
  EXPECT_EQ(runtimeInfo.infoPerCore[31].infoRow.initSrcTuple[3], 0);
  EXPECT_EQ(runtimeInfo.infoPerCore[31].infoRow.initSrcTuple[4], 0);
  EXPECT_EQ(runtimeInfo.infoPerCore[31].infoRow.initSrcTuple[5], 0);
}

TEST_F(TransposeTilingTestV2, last_axis_join_transpose_col_and_row_with_multi_axis) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 6;
  shapeInfo.dim = 6;
  shapeInfo.inShape[0] =2;
  shapeInfo.inShape[1] =3;
  shapeInfo.inShape[2] =4;
  shapeInfo.inShape[3] =128;
  shapeInfo.inShape[4] =8;
  shapeInfo.inShape[5] =224;

  shapeInfo.outShape[0] = 2;
  shapeInfo.outShape[1] = 3;
  shapeInfo.outShape[2] = 224;
  shapeInfo.outShape[3] = 8;
  shapeInfo.outShape[4] = 128;
  shapeInfo.outShape[5] = 4;

  shapeInfo.perm[0] = 0;
  shapeInfo.perm[1] = 1;
  shapeInfo.perm[2] = 5;
  shapeInfo.perm[3] = 4;
  shapeInfo.perm[4] = 3;
  shapeInfo.perm[5] = 2;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);

  EXPECT_EQ(runtimeInfo.dstJumpStride[0], 4096);
  EXPECT_EQ(runtimeInfo.srcJumpAxisNum, 3);

  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoCol.colPerMC, 128);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoCol.loopOnMC, 1);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoCol.colTC, 96);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoCol.colOffset, 0);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoCol.backStepLeft, 0);

  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoRow.rowPerMR, 128);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoRow.loopOnMR, 1);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoRow.rowTR, 0);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoRow.rowOffset, 0);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoRow.backStepUp, 0);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoRow.initSrcTuple[1], 0);
  EXPECT_EQ(runtimeInfo.infoPerCore[0].infoRow.initSrcTuple[1], 0);

  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.colPerMC, 128);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.loopOnMC, 1);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.colTC, 96);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.colOffset, 0);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.backStepLeft, 0);

  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.rowPerMR, 128);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.loopOnMR, 1);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.rowTR, 0);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.rowOffset, 128);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.backStepUp, 0);
  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.initSrcTuple[1], 32);
}

TEST_F(TransposeTilingTestV2, last_axis_join_transpose) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 5;
  shapeInfo.dim = 5;
  shapeInfo.inShape[0] = 5;
  shapeInfo.inShape[1] = 6;
  shapeInfo.inShape[2] = 7;
  shapeInfo.inShape[3] = 8;
  shapeInfo.inShape[4] = 90;

  shapeInfo.outShape[0] = 5;
  shapeInfo.outShape[1] = 90;
  shapeInfo.outShape[2] = 8;
  shapeInfo.outShape[3] = 7;
  shapeInfo.outShape[4] = 6;

  shapeInfo.perm[0] = 0;
  shapeInfo.perm[1] = 4;
  shapeInfo.perm[2] = 3;
  shapeInfo.perm[3] = 2;
  shapeInfo.perm[4] = 1;

  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);

  EXPECT_EQ(runtimeInfo.nJumpAxisNum, 1);
  EXPECT_EQ(runtimeInfo.srcJumpAxisNum, 2);
  EXPECT_EQ(runtimeInfo.dstJumpAxisNum, 2);
  EXPECT_EQ(runtimeInfo.srcJumpStride[0], 5040);
  EXPECT_EQ(runtimeInfo.srcJumpStride[1], 720);
  EXPECT_EQ(runtimeInfo.dstJumpStride[0], 336);
  EXPECT_EQ(runtimeInfo.dstJumpStride[1], 42);

  EXPECT_EQ(runtimeInfo.infoPerCore[6].infoN.loopOnN, 1);

  EXPECT_EQ(runtimeInfo.infoPerCore[6].infoCol.colPerMC, 120);
  EXPECT_EQ(runtimeInfo.infoPerCore[6].infoCol.loopOnMC, 1);

  EXPECT_EQ(runtimeInfo.infoPerCore[6].infoRow.rowPerMR, 40);
  EXPECT_EQ(runtimeInfo.infoPerCore[6].infoRow.loopOnMR, 1);
}

TEST_F(TransposeTilingTestV2, last_axis_join_transpose_t2f) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 2;
  shapeInfo.dim = 2;
  shapeInfo.inShape[0] = 20000;
  shapeInfo.inShape[1] = 7;
  shapeInfo.outShape[0] = 7;
  shapeInfo.outShape[1] = 20000;
  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 0;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);

  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.rowPerMR, 288);
}

TEST_F(TransposeTilingTestV2, last_axis_join_transpose_t2f_fp16) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT16;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 2;
  shapeInfo.dim = 2;
  shapeInfo.inShape[0] = 20000;
  shapeInfo.inShape[1] = 7;
  shapeInfo.outShape[0] = 7;
  shapeInfo.outShape[1] = 20000;
  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 0;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);

  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.rowPerMR, 576);
}

TEST_F(TransposeTilingTestV2, last_axis_join_transpose_f2t) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 2;
  shapeInfo.dim = 2;
  shapeInfo.inShape[0] = 2;
  shapeInfo.inShape[1] = 30000;
  shapeInfo.outShape[0] = 30000;
  shapeInfo.outShape[1] = 2;
  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 0;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);

  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.colPerMC, 944);
}

TEST_F(TransposeTilingTestV2, last_axis_join_transpose_f2t_fp16) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT16;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 2;
  shapeInfo.dim = 2;
  shapeInfo.inShape[0] = 2;
  shapeInfo.inShape[1] = 60000;
  shapeInfo.outShape[0] = 60000;
  shapeInfo.outShape[1] = 2;
  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 0;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);

  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.colPerMC, 944 * 2);
}

TEST_F(TransposeTilingTestV2, small_shape_check_split_with_unit_ele_per_block) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 4;
  shapeInfo.dim = 4;
  shapeInfo.inShape[0] = 8;
  shapeInfo.inShape[1] = 6;
  shapeInfo.inShape[2] = 1;
  shapeInfo.inShape[3] = 3;
  shapeInfo.outShape[4] = 1;
  shapeInfo.outShape[4] = 8;
  shapeInfo.outShape[4] = 6;
  shapeInfo.outShape[4] = 3;
  shapeInfo.perm[0] = 2;
  shapeInfo.perm[1] = 0;
  shapeInfo.perm[2] = 1;
  shapeInfo.perm[3] = 3;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);

  EXPECT_EQ(runtimeInfo.infoPerCoreIdentical[0].base, 0);
  EXPECT_EQ(runtimeInfo.infoPerCoreIdentical[0].eleNum, 8);
  EXPECT_EQ(runtimeInfo.infoPerCoreIdentical[1].base, 8);
  EXPECT_EQ(runtimeInfo.infoPerCoreIdentical[1].eleNum, 8);
  EXPECT_EQ(runtimeInfo.infoPerCoreIdentical[19].base, 0);
  EXPECT_EQ(runtimeInfo.infoPerCoreIdentical[19].eleNum, 0);
}

TEST_F(TransposeTilingTestV2, small_shape_check_split_with_no_minus) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 1;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 4;
  shapeInfo.dim = 4;
  shapeInfo.inShape[0] = 6;
  shapeInfo.inShape[1] = 4;
  shapeInfo.inShape[2] = 9;
  shapeInfo.inShape[3] = 1;
  shapeInfo.outShape[0] = 9;
  shapeInfo.outShape[1] = 6;
  shapeInfo.outShape[2] = 4;
  shapeInfo.outShape[3] = 1;
  shapeInfo.perm[0] = 2;
  shapeInfo.perm[1] = 0;
  shapeInfo.perm[2] = 1;
  shapeInfo.perm[3] = 3;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);

  EXPECT_EQ(runtimeInfo.infoPerCoreLastAxisNT[0].num, 216);
}

TEST_F(TransposeTilingTestV2, less_than_one_block) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 4;
  shapeInfo.dim = 4;
  shapeInfo.inShape[0] = 1;
  shapeInfo.inShape[1] = 1;
  shapeInfo.inShape[2] = 7;
  shapeInfo.inShape[3] = 1;
  shapeInfo.outShape[0] = 7;
  shapeInfo.outShape[1] = 1;
  shapeInfo.outShape[2] = 1;
  shapeInfo.outShape[3] = 1;
  shapeInfo.perm[0] =2;
  shapeInfo.perm[1] =0;
  shapeInfo.perm[2] =1;
  shapeInfo.perm[3] =3;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);

  EXPECT_EQ(runtimeInfo.infoPerCoreIdentical[0].eleNum, 7);
  EXPECT_EQ(runtimeInfo.infoPerCoreIdentical[1].eleNum, 0);
}

TEST_F(TransposeTilingTestV2, borrow_src_1_dst_1_axis) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 5;
  shapeInfo.dim = 5;
  shapeInfo.inShape[0] = 100;
  shapeInfo.inShape[1] = 20;
  shapeInfo.inShape[2] = 3;
  shapeInfo.inShape[3] = 400;
  shapeInfo.inShape[4] = 8;

  shapeInfo.outShape[0] = 400;
  shapeInfo.outShape[1] = 3;
  shapeInfo.outShape[2] = 20;
  shapeInfo.outShape[3] = 100;
  shapeInfo.outShape[4] = 8;

  shapeInfo.perm[0] = 3;
  shapeInfo.perm[1] = 2;
  shapeInfo.perm[2] = 1;
  shapeInfo.perm[3] = 0;
  shapeInfo.perm[4] = 4;

  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  // EXPECT_EQ(shapeInfo.reducedInShape.size(), 1);
}

TEST_F(TransposeTilingTestV2, borrow_src_2_dst_2_dup_axis) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 4;
  shapeInfo.dim = 4;

  shapeInfo.inShape[0] = 10;
  shapeInfo.inShape[1] = 3000;
  shapeInfo.inShape[2] = 20;
  shapeInfo.inShape[3] = 8;

  shapeInfo.outShape[0] = 20;
  shapeInfo.outShape[1] = 3000;
  shapeInfo.outShape[2] = 10;
  shapeInfo.outShape[3] = 8;

  shapeInfo.perm[0] = 2;
  shapeInfo.perm[1] = 1;
  shapeInfo.perm[2] = 0;
  shapeInfo.perm[3] = 3;

  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  // EXPECT_EQ(shapeInfo.reducedInShape.size(), 1);
}

TEST_F(TransposeTilingTestV2, borrow_src_2_dst_2_dup_axis_with_tail) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 4;
  shapeInfo.dim = 4;
  shapeInfo.inShape[0] = 10;
  shapeInfo.inShape[1] = 3003;
  shapeInfo.inShape[2] = 20;
  shapeInfo.inShape[3] = 8;

  shapeInfo.outShape[0] = 20;
  shapeInfo.outShape[1] = 3003;
  shapeInfo.outShape[2] = 10;
  shapeInfo.outShape[3] = 8;

  shapeInfo.perm[0] = 2;
  shapeInfo.perm[1] = 1;
  shapeInfo.perm[2] = 0;
  shapeInfo.perm[3] = 3;

  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  // EXPECT_EQ(shapeInfo.reducedInShape.size(), 1);
}

TEST_F(TransposeTilingTestV2, borrow_src_2_dst_2_dup_axis_vcopy) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 48;
  compilerInfo.ubSize = 6144;
  runtimeInfo.dType = ge::DT_INT32;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 5;
  shapeInfo.dim = 5;
  shapeInfo.inShape[0] = 6;
  shapeInfo.inShape[1] = 4;
  shapeInfo.inShape[2] = 1280;
  shapeInfo.inShape[3] = 8;
  shapeInfo.inShape[4] = 3;

  shapeInfo.outShape[0] = 6;
  shapeInfo.outShape[1] = 8;
  shapeInfo.outShape[2] = 1280;
  shapeInfo.outShape[3] = 4;
  shapeInfo.outShape[4] = 3;

  shapeInfo.perm[0] = 0;
  shapeInfo.perm[1] = 3;
  shapeInfo.perm[2] = 2;
  shapeInfo.perm[3] = 1;
  shapeInfo.perm[4] = 4;

  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  // EXPECT_EQ(shapeInfo.reducedInShape.size(), 1);
}

TEST_F(TransposeTilingTestV2, borrow_src_2_dst_2_dup_axis_data_move) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_INT32;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 5;
  shapeInfo.dim = 5;
  shapeInfo.inShape[0] = 6;
  shapeInfo.inShape[1] = 8;
  shapeInfo.inShape[2] = 1800;
  shapeInfo.inShape[3] = 8;
  shapeInfo.inShape[4] = 3;

  shapeInfo.outShape[0] = 6;
  shapeInfo.outShape[1] = 8;
  shapeInfo.outShape[2] = 1800;
  shapeInfo.outShape[3] = 8;
  shapeInfo.outShape[4] = 3;

  shapeInfo.perm[0] = 0;
  shapeInfo.perm[1] = 3;
  shapeInfo.perm[2] = 2;
  shapeInfo.perm[3] = 1;
  shapeInfo.perm[4] = 4;

  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  // EXPECT_EQ(shapeInfo.reducedInShape.size(), 1);
}

TEST_F(TransposeTilingTestV2, borrow_dup_one_axis_data_move) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_INT32;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 3;
  shapeInfo.dim = 3;
  shapeInfo.inShape[0] = 7000;
  shapeInfo.inShape[1] = 9;
  shapeInfo.inShape[2] = 8;

  shapeInfo.outShape[0] = 9;
  shapeInfo.outShape[1] = 7000;
  shapeInfo.outShape[2] = 8;

  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 0;
  shapeInfo.perm[2] = 2;

  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  // EXPECT_EQ(shapeInfo.reducedInShape.size(), 1);
}

TEST_F(TransposeTilingTestV2, borrow_src_2_dst_2_dup_not_aligned) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_INT32;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 5;
  shapeInfo.dim = 5;
  shapeInfo.inShape[0] = 6;
  shapeInfo.inShape[1] = 4;
  shapeInfo.inShape[2] = 1800;
  shapeInfo.inShape[3] = 8;
  shapeInfo.inShape[4] = 3;

  shapeInfo.outShape[0] = 6;
  shapeInfo.outShape[1] = 8;
  shapeInfo.outShape[2] = 1800;
  shapeInfo.outShape[3] = 4;
  shapeInfo.outShape[4] = 3;

  shapeInfo.perm[0] = 0;
  shapeInfo.perm[1] = 3;
  shapeInfo.perm[2] = 2;
  shapeInfo.perm[3] = 1;
  shapeInfo.perm[4] = 4;

  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  // EXPECT_EQ(shapeInfo.reducedInShape.size(), 1);
}

TEST_F(TransposeTilingTestV2, borrow_src_2_dst_2_axis) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 5;
  shapeInfo.dim = 5;
  shapeInfo.inShape[0] = 2;
  shapeInfo.inShape[1] = 30;
  shapeInfo.inShape[2] = 40;
  shapeInfo.inShape[3] = 5;
  shapeInfo.inShape[4] = 8;

  shapeInfo.outShape[0] = 5;
  shapeInfo.outShape[1] = 40;
  shapeInfo.outShape[2] = 30;
  shapeInfo.outShape[3] = 2;
  shapeInfo.outShape[4] = 8;

  shapeInfo.perm[0] = 3;
  shapeInfo.perm[1] = 2;
  shapeInfo.perm[2] = 1;
  shapeInfo.perm[3] = 0;
  shapeInfo.perm[4] = 4;

  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  // EXPECT_EQ(shapeInfo.reducedInShape.size(), 1);
}

TEST_F(TransposeTilingTestV2, borrow_src_2_dst_2_axis_with_6_axis) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 7;
  shapeInfo.dim = 7;
  shapeInfo.inShape[0] = 2;
  shapeInfo.inShape[1] = 50;
  shapeInfo.inShape[2] = 9;
  shapeInfo.inShape[3] = 23;
  shapeInfo.inShape[4] = 40;
  shapeInfo.inShape[5] = 5;
  shapeInfo.inShape[6] = 8;

  shapeInfo.outShape[0] = 5;
  shapeInfo.outShape[1] = 40;
  shapeInfo.outShape[2] = 23;
  shapeInfo.outShape[3] = 9;
  shapeInfo.outShape[4] = 50;
  shapeInfo.outShape[5] = 2;
  shapeInfo.outShape[6] = 8;

  shapeInfo.perm[0] = 5;
  shapeInfo.perm[1] = 4;
  shapeInfo.perm[2] = 3;
  shapeInfo.perm[3] = 2;
  shapeInfo.perm[4] = 1;
  shapeInfo.perm[5] = 0;
  shapeInfo.perm[6] = 6;

  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  // EXPECT_EQ(shapeInfo.reducedInShape.size(), 1);
}

TEST_F(TransposeTilingTestV2, borrow_2dup_axis) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 4;
  shapeInfo.dim = 4;
  shapeInfo.inShape[0] = 100;
  shapeInfo.inShape[1] = 70;
  shapeInfo.inShape[2] = 9;
  shapeInfo.inShape[3] = 8;

  shapeInfo.outShape[0] = 100;
  shapeInfo.outShape[1] = 9;
  shapeInfo.outShape[2] = 70;
  shapeInfo.outShape[3] = 8;

  shapeInfo.perm[0] = 0;
  shapeInfo.perm[1] = 2;
  shapeInfo.perm[2] = 1;
  shapeInfo.perm[3] = 3;

  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  // EXPECT_EQ(shapeInfo.reducedInShape.size(), 1);
}

TEST_F(TransposeTilingTestV2, temp) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 2;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT16;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 6;
  shapeInfo.dim = 6;
  shapeInfo.inShape[0] = 5;
  shapeInfo.inShape[1] = 4;
  shapeInfo.inShape[2] = 500;
  shapeInfo.inShape[3] = 1;
  shapeInfo.inShape[4] = 81;
  shapeInfo.inShape[5] = 1;

  shapeInfo.outShape[0] = 4;
  shapeInfo.outShape[1] = 500;
  shapeInfo.outShape[2] = 1;
  shapeInfo.outShape[3] = 81;
  shapeInfo.outShape[4] = 1;
  shapeInfo.outShape[5] = 5;

  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 2;
  shapeInfo.perm[2] = 3;
  shapeInfo.perm[3] = 4;
  shapeInfo.perm[4] = 5;
  shapeInfo.perm[5] = 0;

  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
}

TEST_F(TransposeTilingTestV2, small_shape_all_one) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 4;
  shapeInfo.dim = 4;
  shapeInfo.inShape[0] = 1;
  shapeInfo.inShape[1] = 1;
  shapeInfo.inShape[2] = 1;
  shapeInfo.inShape[3] = 1;
  shapeInfo.outShape[0] = 1;
  shapeInfo.outShape[1] = 1;
  shapeInfo.outShape[2] = 1;
  shapeInfo.outShape[3] = 1;
  shapeInfo.perm[0] = 0;
  shapeInfo.perm[1] = 1;
  shapeInfo.perm[2] = 3;
  shapeInfo.perm[3] = 2;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_EQ(shapeInfo.dim, 1);
}

TEST_F(TransposeTilingTestV2, nchw) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 4;
  shapeInfo.dim = 4;
  shapeInfo.inShape[0] = 6;
  shapeInfo.inShape[1] = 7;
  shapeInfo.inShape[2] = 4;
  shapeInfo.inShape[3] = 7;
  shapeInfo.outShape[0] = 4;
  shapeInfo.outShape[1] = 6;
  shapeInfo.outShape[2] = 7;
  shapeInfo.outShape[3] = 7;
  shapeInfo.perm[0] = 2;
  shapeInfo.perm[1] = 0;
  shapeInfo.perm[2] = 1;
  shapeInfo.perm[3] = 3;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);

  // EXPECT_EQ(runtimeInfo.infoPerCoreLastAxisNT[1].base, 6);
  // EXPECT_EQ(runtimeInfo.infoPerCoreLastAxisNT[1].num, 6);
}

TEST_F(TransposeTilingTestV2, ub310) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 7936;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 2;
  shapeInfo.dim = 2;
  shapeInfo.inShape[0] = 4096;
  shapeInfo.inShape[1] = 8192;
  shapeInfo.outShape[0] = 8192;
  shapeInfo.outShape[1] = 4096;
  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 0;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_TRUE(TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo));
}

TEST_F(TransposeTilingTestV2, ublihisi) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 6144;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 2;
  shapeInfo.dim = 2;
  shapeInfo.inShape[0] = 4096;
  shapeInfo.inShape[1] = 8192;
  shapeInfo.outShape[0] = 8192;
  shapeInfo.outShape[1] = 4096;
  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 0;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_TRUE(TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo));
}

TEST_F(TransposeTilingTestV2, int8) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_INT8;
  runtimeInfo.fp16Times = 1;

  shapeInfo.inShape.push_back(60);
  shapeInfo.inShape.push_back(70);
  shapeInfo.outShape.push_back(70);
  shapeInfo.outShape.push_back(60);
  shapeInfo.perm.push_back(1);
  shapeInfo.perm.push_back(0);
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_TRUE(TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo));
}

TEST_F(TransposeTilingTestV2, int8_scenario_5_001) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_INT8;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 3;
  shapeInfo.dim = 3;
  shapeInfo.inShape[0] = 60;
  shapeInfo.inShape[1] = 70;
  shapeInfo.inShape[2] = 32;
  shapeInfo.outShape[0] = 70;
  shapeInfo.outShape[1] = 60;
  shapeInfo.outShape[2] = 32;
  shapeInfo.perm[0]= 1;
  shapeInfo.perm[1]= 0;
  shapeInfo.perm[2]= 2;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_TRUE(TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo));
  EXPECT_EQ(shapeInfo.scenario, 5);
}

TEST_F(TransposeTilingTestV2, int8_scenario_5_002) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_INT8;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 3;
  shapeInfo.dim = 3;
  shapeInfo.inShape[0] = 60;
  shapeInfo.inShape[1] = 70;
  shapeInfo.inShape[2] = 33;
  shapeInfo.outShape[0] = 70;
  shapeInfo.outShape[1] = 60;
  shapeInfo.outShape[2] = 33;
  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 0;
  shapeInfo.perm[2] = 2;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_TRUE(TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo));
  EXPECT_EQ(shapeInfo.scenario, 5);
}

//TEST_F(TransposeTilingTestV2, int8_scenario_1_001) {
//  TransposeCompilerInfo compilerInfo;
//  ShapeInfo shapeInfo;
//  RuntimeInfoV2 runtimeInfo;
//  compilerInfo.coreNum = 32;
//  compilerInfo.ubSize = 8192;
//  runtimeInfo.dType = ge::DT_INT8;
//  runtimeInfo.fp16Times = 1;
//
//  shapeInfo.inShape.push_back(60);
//  shapeInfo.inShape.push_back(70);
//  shapeInfo.inShape.push_back(32000);
//  shapeInfo.outShape.push_back(70);
//  shapeInfo.outShape.push_back(60);
//  shapeInfo.outShape.push_back(32000);
//  shapeInfo.perm.push_back(1);
//  shapeInfo.perm.push_back(0);
//  shapeInfo.perm.push_back(2);
//  ReduceAxisV2("Transpose", compilerInfo, shapeInfo);
//  EXPECT_TRUE(TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo));
//  EXPECT_EQ(shapeInfo.scenario, 1);
//}

TEST_F(TransposeTilingTestV2, int8_scenario_3_001) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_INT8;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 3;
  shapeInfo.dim = 3;
  shapeInfo.inShape[0] = 60;
  shapeInfo.inShape[1] = 70;
  shapeInfo.inShape[2] = 192 * 1024;
  shapeInfo.outShape[0] = 70;
  shapeInfo.outShape[1] = 60;
  shapeInfo.outShape[2] = 192 * 1024;
  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 0;
  shapeInfo.perm[2] = 2;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_TRUE(TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo));
  EXPECT_EQ(shapeInfo.scenario, 3);
}

TEST_F(TransposeTilingTestV2, int16_scenario_9_001) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_INT16;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 3;
  shapeInfo.dim = 3;
  shapeInfo.inShape[0] = 60;
  shapeInfo.inShape[1] = 16;
  shapeInfo.inShape[2] = 32;
  shapeInfo.outShape[0] = 60;
  shapeInfo.outShape[1] = 32;
  shapeInfo.outShape[2] = 16;
  shapeInfo.perm[0] = 0;
  shapeInfo.perm[1] = 2;
  shapeInfo.perm[2] = 1;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_TRUE(TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo));
  EXPECT_EQ(shapeInfo.scenario, 10);
}

TEST_F(TransposeTilingTestV2, specific_shape_scenario_4) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.ubSizeCouldUse = 8064;
  runtimeInfo.dType = ge::DT_FLOAT16;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 4;
  shapeInfo.dim = 4;
  shapeInfo.inShape[0] = 1000;
  shapeInfo.inShape[1] = 5;
  shapeInfo.inShape[2] = 64;
  shapeInfo.inShape[3] = 64;

  shapeInfo.outShape[0] = 1000;
  shapeInfo.outShape[1] = 64;
  shapeInfo.outShape[2] = 5;
  shapeInfo.outShape[3] = 64;

  shapeInfo.perm[0] = 0;
  shapeInfo.perm[1] = 2;
  shapeInfo.perm[2] = 1;
  shapeInfo.perm[3] = 3;

  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_TRUE(TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo));
  EXPECT_EQ(shapeInfo.scenario, 5);
}

TEST_F(TransposeTilingTestV2, specific_shape_scenario_5) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.ubSizeCouldUse = 8064;
  runtimeInfo.dType = ge::DT_FLOAT16;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 6;
  shapeInfo.dim = 6;
  shapeInfo.inShape[0] = 2;
  shapeInfo.inShape[1] = 3;
  shapeInfo.inShape[2] = 4;
  shapeInfo.inShape[3] = 2;
  shapeInfo.inShape[4] = 3;
  shapeInfo.inShape[5] = 4;

  shapeInfo.outShape[0] = 4;
  shapeInfo.outShape[1] = 3;
  shapeInfo.outShape[2] = 2;
  shapeInfo.outShape[3] = 4;
  shapeInfo.outShape[4] = 3;
  shapeInfo.outShape[5] = 2;

  shapeInfo.perm[0] = 5;
  shapeInfo.perm[1] = 4;
  shapeInfo.perm[2] = 3;
  shapeInfo.perm[3] = 2;
  shapeInfo.perm[4] = 1;
  shapeInfo.perm[5] = 0;

  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_TRUE(TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo));
  EXPECT_EQ(shapeInfo.scenario, 5);
}

TEST_F(TransposeTilingTestV2, specific_shape_scenario_9) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.ubSizeCouldUse = 8064;
  runtimeInfo.dType = ge::DT_FLOAT16;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 3;
  shapeInfo.dim = 3;
  shapeInfo.inShape[0] = 100;
  shapeInfo.inShape[1] = 200;
  shapeInfo.inShape[2] = 128;

  shapeInfo.outShape[0] = 200;
  shapeInfo.outShape[1] = 100;
  shapeInfo.outShape[2] = 128;

  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 0;
  shapeInfo.perm[2] = 2;

  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_TRUE(TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo));
  EXPECT_EQ(shapeInfo.scenario, 9);
}

TEST_F(TransposeTilingTestV2, fp16_310) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 2;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT16;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 3;
  shapeInfo.dim = 3;
  shapeInfo.inShape[0] = 3;
  shapeInfo.inShape[1] = 256000;
  shapeInfo.inShape[2] = 8;
  shapeInfo.outShape[0] = 8;
  shapeInfo.outShape[1] = 256000;
  shapeInfo.outShape[2] = 3;
  shapeInfo.perm[0] = 2;
  shapeInfo.perm[1] = 1;
  shapeInfo.perm[2] = 0;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
}

TEST_F(TransposeTilingTestV2, split_n_with_small_shape_1) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 4;
  shapeInfo.dim = 4;
  shapeInfo.inShape[0] = 6;
  shapeInfo.inShape[1] = 7;
  shapeInfo.inShape[2] = 7;
  shapeInfo.inShape[3] = 1;
  shapeInfo.outShape[0] = 7;
  shapeInfo.outShape[1] = 6;
  shapeInfo.outShape[2] = 7;
  shapeInfo.outShape[3] = 1;
  shapeInfo.perm[0] = 2;
  shapeInfo.perm[1] = 0;
  shapeInfo.perm[2] = 1;
  shapeInfo.perm[3] = 3;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_EQ(shapeInfo.scenario, 5);
}

/*
  scenario  in                  out                 perm        reducedIn           reducedOut          reducedPerm  dim
  lastAxisLen  lastAxisBurstLen  alignElement
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  6         1,2,2,1             2,1,2,1             2,0,1,3     2,2,1               2,2,1               1,0,2        3
  1            1                 7

  coreNum    usedCoreNum    ubSize    ubSizeCouldUse
  --------------------------------------------------
  32         1              8192      0

  backNum  skipEle  srcStrideLogic  srcJumpStride                 dstJumpStride                 dstJumpFactor
  dstJumpFactorMod
  -------------------------------------------------------------------------------------------------------------------------------------------------------
  8        0        1               2,1                           1,2                           2,2 1,2

  base        num        initTuple                     headMajorLoop  headMajorNum  headTailNum  bodyLoopNum
  bodymajorLoop  bodyMajorNum  bodyTailNum  tailMajorLoop  tailMajorNum  tailTailNum
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  0           4          0,0                           0              0             0            0            0 0 0 0 0
  0 0           0          0,0                           0              0             0            0            0 0 0 0
  0             0 0           0          0,0                           0              0             0            0 0 0
  0            0              0             0 0           0          0,0                           0              0 0 0
  0              0             0            0              0             0 0           0          0,0 0              0
  0            0            0              0             0            0              0             0 0           0 0,0
  0              0             0            0            0              0             0            0              0 0 0
  0          0,0                           0              0             0            0            0              0 0 0
  0             0
 */

TEST_F(TransposeTilingTestV2, split_n_with_small_shape_2) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 4;
  shapeInfo.dim = 4;
  shapeInfo.inShape[0] = 1;
  shapeInfo.inShape[1] = 2;
  shapeInfo.inShape[2] = 2;
  shapeInfo.inShape[3] = 1;
  shapeInfo.outShape[0] = 2;
  shapeInfo.outShape[1] = 1;
  shapeInfo.outShape[2] = 2;
  shapeInfo.outShape[3] = 1;
  shapeInfo.perm[0] = 2;
  shapeInfo.perm[1] = 0;
  shapeInfo.perm[2] = 1;
  shapeInfo.perm[3] = 3;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_EQ(runtimeInfo.infoPerCoreLastAxisNT[0].base, 0);
  EXPECT_EQ(runtimeInfo.infoPerCoreLastAxisNT[0].num, 4);
  EXPECT_EQ(runtimeInfo.infoPerCoreLastAxisNT[1].base, 0);
  EXPECT_EQ(runtimeInfo.infoPerCoreLastAxisNT[1].num, 0);
}

TEST_F(TransposeTilingTestV2, scenario_11_small_tail) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 3;
  shapeInfo.dim = 3;
  shapeInfo.inShape[0] = 4099;
  shapeInfo.inShape[1] = 16;
  shapeInfo.inShape[2] = 32;
  shapeInfo.outShape[0] = 4099;
  shapeInfo.outShape[1] = 32;
  shapeInfo.outShape[2] = 16;
  shapeInfo.perm[0] = 0;
  shapeInfo.perm[1] = 2;
  shapeInfo.perm[2] = 1;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
}

TEST_F(TransposeTilingTestV2, scenario_11_big_tail) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 3;
  shapeInfo.dim = 3;
  shapeInfo.inShape[0] = 3;
  shapeInfo.inShape[1] = 2000;
  shapeInfo.inShape[2] = 1000;
  shapeInfo.outShape[0] = 3;
  shapeInfo.outShape[1] = 1000;
  shapeInfo.outShape[2] = 2000;
  shapeInfo.perm[0] = 0;
  shapeInfo.perm[1] = 2;
  shapeInfo.perm[2] = 1;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
}

TEST_F(TransposeTilingTestV2, scenario_2) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT16;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 3;
  shapeInfo.dim = 3;
  shapeInfo.inShape[0] = 2560;
  shapeInfo.inShape[1] = 26;
  shapeInfo.inShape[2] = 512;
  shapeInfo.outShape[0] = 26;
  shapeInfo.outShape[1] = 2560;
  shapeInfo.outShape[2] = 512;
  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 0;
  shapeInfo.perm[2] = 2;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_EQ(shapeInfo.scenario, 2);
}

TEST_F(TransposeTilingTestV2, scenario_2_002) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT16;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 5;
  shapeInfo.dim = 5;
  shapeInfo.inShape[0] = 2;
  shapeInfo.inShape[1] = 3000;
  shapeInfo.inShape[2] = 4000;
  shapeInfo.inShape[3] = 26;
  shapeInfo.inShape[4] = 512;
  shapeInfo.outShape[0] = 2;
  shapeInfo.outShape[1] = 26;
  shapeInfo.outShape[2] = 4000;
  shapeInfo.outShape[3] = 3000;
  shapeInfo.outShape[4] = 512;
  shapeInfo.perm[0] = 0;
  shapeInfo.perm[1] = 3;
  shapeInfo.perm[2] = 2;
  shapeInfo.perm[3] = 1;
  shapeInfo.perm[4] = 4;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_EQ(shapeInfo.scenario, 2);
}

TEST_F(TransposeTilingTestV2, scenario_9_stride_huge_001) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.ubSizeCouldUse = 8064;
  runtimeInfo.dType = ge::DT_FLOAT16;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 3;
  shapeInfo.dim = 3;
  shapeInfo.inShape[0] = 2560;
  shapeInfo.inShape[1] = 26;
  shapeInfo.inShape[2] = 512;
  shapeInfo.outShape[0] = 26;
  shapeInfo.outShape[1] = 2560;
  shapeInfo.outShape[2] = 512;
  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 0;
  shapeInfo.perm[2] = 2;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_EQ(shapeInfo.scenario, 2);
}


TEST_F(TransposeTilingTestV2, scenario_11_stride_huge_goto_scenario_7_001) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT16;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 2;
  shapeInfo.dim = 2;
  shapeInfo.inShape[0] = 32;
  shapeInfo.inShape[1] = 65536*16;
  shapeInfo.outShape[0] = 65536*16;
  shapeInfo.outShape[1] = 32;
  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 0;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_TRUE(TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo));
  EXPECT_EQ(shapeInfo.scenario, 7);
}

TEST_F(TransposeTilingTestV2, scenario_11_stride_huge_goto_scenario_7_002) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT16;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 2;
  shapeInfo.dim = 2;
  shapeInfo.inShape[0] = 65536*16;
  shapeInfo.inShape[1] = 32;
  shapeInfo.outShape[0] = 32;
  shapeInfo.outShape[1] = 65536*16;
  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 0;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_EQ(shapeInfo.scenario, 7);
}

TEST_F(TransposeTilingTestV2, scenario_11_001) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT16;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 3;
  shapeInfo.dim = 3;
  shapeInfo.inShape[0] = 32;
  shapeInfo.inShape[1] = 2048;
  shapeInfo.inShape[2] = 128;
  shapeInfo.outShape[0] = 2048;
  shapeInfo.outShape[1] = 128;
  shapeInfo.outShape[2] = 32;
  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 2;
  shapeInfo.perm[2] = 0;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_EQ(shapeInfo.scenario, 11);
}

TEST_F(TransposeTilingTestV2, scenario_11_002) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT16;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 2;
  shapeInfo.dim = 2;
  shapeInfo.inShape[0] = 1024;
  shapeInfo.inShape[1] = 2880;
  shapeInfo.outShape[0] = 2880;
  shapeInfo.outShape[1] = 1024;
  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 0;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_EQ(shapeInfo.scenario, 11);
}

TEST_F(TransposeTilingTestV2, scenario_11_003) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT16;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 3;
  shapeInfo.dim = 3;
  shapeInfo.inShape[0] = 2048;
  shapeInfo.inShape[1] = 128;
  shapeInfo.inShape[2] = 32;
  shapeInfo.outShape[0] = 32;
  shapeInfo.outShape[1] = 2048;
  shapeInfo.outShape[2] = 128;
  shapeInfo.perm[0] = 2;
  shapeInfo.perm[1] = 0;
  shapeInfo.perm[2] = 1;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_EQ(shapeInfo.scenario, 11);
}


TEST_F(TransposeTilingTestV2, scenario_11_big_tail_2d) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 2;
  shapeInfo.dim = 2;
  shapeInfo.inShape[0] = 2000;
  shapeInfo.inShape[1] = 1000;
  shapeInfo.outShape[0] = 1000;
  shapeInfo.outShape[1] = 2000;
  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 0;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  utils::OpRunInfo runInfo;
  //SerializeScenario11(runInfo, compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_EQ(shapeInfo.scenario, 11);
}

TEST_F(TransposeTilingTestV2, scenario_4) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.ubSizeCouldUse = 8064;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 3;
  shapeInfo.dim = 3;
  shapeInfo.inShape[0] = 32;
  shapeInfo.inShape[1] = 256;
  shapeInfo.inShape[2] = 128;
  shapeInfo.outShape[0] = 256;
  shapeInfo.outShape[1] = 32;
  shapeInfo.outShape[2] = 128;
  shapeInfo.perm[0] =1;
  shapeInfo.perm[1] =0;
  shapeInfo.perm[2] =2;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  utils::OpRunInfo runInfo;
  EXPECT_EQ(shapeInfo.scenario, 9);
}

TEST_F(TransposeTilingTestV2, scenario_5_int_8) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_INT8;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 6;
  shapeInfo.dim = 6;
  shapeInfo.inShape[0] = 3;
  shapeInfo.inShape[1] = 60;
  shapeInfo.inShape[2] = 90;
  shapeInfo.inShape[3] = 6;
  shapeInfo.inShape[4] = 6;
  shapeInfo.inShape[5] = 213;

  shapeInfo.outShape[0] = 3;
  shapeInfo.outShape[1] = 60;
  shapeInfo.outShape[2] = 6;
  shapeInfo.outShape[3] = 90;
  shapeInfo.outShape[4] = 6;
  shapeInfo.outShape[5] = 213;

  shapeInfo.perm[0] =0;
  shapeInfo.perm[1] =1;
  shapeInfo.perm[2] =3;
  shapeInfo.perm[3] =2;
  shapeInfo.perm[4] =4;
  shapeInfo.perm[5] =5;

  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  utils::OpRunInfo runInfo;
  //SerializeScenario4(runInfo, compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_EQ(shapeInfo.scenario, 5);
}

TEST_F(TransposeTilingTestV2, scenario_5_int_8_2) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 10;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_INT8;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 3;
  shapeInfo.dim = 3;
  shapeInfo.inShape[0] = 65;
  shapeInfo.inShape[1] = 84;
  shapeInfo.inShape[2] = 10;

  shapeInfo.outShape[0] = 10;
  shapeInfo.outShape[1] = 84;
  shapeInfo.outShape[2] = 65;

  shapeInfo.perm[0] = 2;
  shapeInfo.perm[1] = 1;
  shapeInfo.perm[2] = 0;

  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  utils::OpRunInfo runInfo;
  EXPECT_EQ(shapeInfo.scenario, 5);
}

TEST_F(TransposeTilingTestV2, scenario_10_int_16) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_INT16;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 4;
  shapeInfo.dim = 4;
  shapeInfo.inShape[0] = 8;
  shapeInfo.inShape[1] = 128;
  shapeInfo.inShape[2] = 64;
  shapeInfo.inShape[3] = 16;

  shapeInfo.outShape[0] = 128;
  shapeInfo.outShape[1] = 8;
  shapeInfo.outShape[2] = 16;
  shapeInfo.outShape[3] = 64;

  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 0;
  shapeInfo.perm[2] = 3;
  shapeInfo.perm[3] = 2;

  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  utils::OpRunInfo runInfo;
  EXPECT_EQ(shapeInfo.scenario, 10);
}

TEST_F(TransposeTilingTestV2, scenario_10_int_16_2) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_INT16;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 4;
  shapeInfo.dim = 4;
  shapeInfo.inShape[0] = 66;
  shapeInfo.inShape[1] = 25;
  shapeInfo.inShape[2] = 7;
  shapeInfo.inShape[3] = 3;

  shapeInfo.outShape[0] = 25;
  shapeInfo.outShape[1] = 66;
  shapeInfo.outShape[2] = 3;
  shapeInfo.outShape[3] = 7;

  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 0;
  shapeInfo.perm[2] = 3;
  shapeInfo.perm[3] = 2;

  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  utils::OpRunInfo runInfo;
  EXPECT_EQ(shapeInfo.scenario, 5);
}

TEST_F(TransposeTilingTestV2, scenario_10_int_16_3) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_INT16;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 4;
  shapeInfo.dim = 4;
  shapeInfo.inShape[0] = 640;
  shapeInfo.inShape[1] = 50;
  shapeInfo.inShape[2] = 60;
  shapeInfo.inShape[3] = 11;

  shapeInfo.outShape[0] = 60;
  shapeInfo.outShape[1] = 50;
  shapeInfo.outShape[2] = 640;
  shapeInfo.outShape[3] = 11;

  shapeInfo.perm[0] = 2;
  shapeInfo.perm[1] = 0;
  shapeInfo.perm[2] = 1;
  shapeInfo.perm[3] = 3;

  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  utils::OpRunInfo runInfo;
  EXPECT_EQ(shapeInfo.scenario, 5);
}


TEST_F(TransposeTilingTestV2, scenario_5_int_8_axis_union) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  compilerInfo.ubSize = 8192;
  runtimeInfo.dType = ge::DT_INT8;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 3;
  shapeInfo.dim = 2;
  shapeInfo.inShape[0] = 33;
  shapeInfo.inShape[1] = 200;

  shapeInfo.outShape[0] = 200;
  shapeInfo.outShape[1] = 33;

  shapeInfo.perm[0] = 1;
  shapeInfo.perm[1] = 0;

  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  utils::OpRunInfo runInfo;
  //SerializeScenario4(runInfo, compilerInfo, shapeInfo, runtimeInfo);
  EXPECT_EQ(shapeInfo.scenario, 5);
}

TEST_F(TransposeTilingTestV2, last_axis_join_transpose_check_jump_stride) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 32;
  runtimeInfo.dType = ge::DT_FLOAT;
  runtimeInfo.fp16Times = 2;

  shapeInfo.origDim = 7;
  shapeInfo.dim = 7;
  shapeInfo.inShape[0] = 3;
  shapeInfo.inShape[1] = 40;
  shapeInfo.inShape[2] = 5;
  shapeInfo.inShape[3] = 6;
  shapeInfo.inShape[4] = 7;
  shapeInfo.inShape[5] = 8;
  shapeInfo.inShape[6] = 9;

  shapeInfo.outShape[0] = 3;
  shapeInfo.outShape[1] = 7;
  shapeInfo.outShape[2] = 6;
  shapeInfo.outShape[3] = 9;
  shapeInfo.outShape[4] = 8;
  shapeInfo.outShape[5] = 5;
  shapeInfo.outShape[6] = 40;

  shapeInfo.perm[0] = 0;
  shapeInfo.perm[1] = 4;
  shapeInfo.perm[2] = 3;
  shapeInfo.perm[3] = 6;
  shapeInfo.perm[4] = 2;
  shapeInfo.perm[5] = 5;
  shapeInfo.perm[6] = 1;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);

  EXPECT_EQ(runtimeInfo.nJumpAxisNum, 1);
  EXPECT_EQ(runtimeInfo.srcJumpAxisNum, 1);
  EXPECT_EQ(runtimeInfo.dstJumpAxisNum, 5);

  EXPECT_EQ(runtimeInfo.nJumpFactor[0], 3);

  EXPECT_EQ(runtimeInfo.nJumpStrideIn[0], 604800);

  EXPECT_EQ(runtimeInfo.srcJumpStride[0], 15120);

  EXPECT_EQ(runtimeInfo.dstJumpStride[0], 1600);

  EXPECT_EQ(runtimeInfo.infoPerCore[1].infoN.loopOnN, 1);
}

TEST_F(TransposeTilingTestV2, scenario_12) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 2;
  compilerInfo.ubSize = 7936;
  runtimeInfo.ubSizeCouldUse = 7808;
  runtimeInfo.dType = ge::DT_FLOAT16;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 3;
  shapeInfo.dim = 3;
  shapeInfo.inShape[0] = 18;
  shapeInfo.inShape[1] = 16;
  shapeInfo.inShape[2] = 12544;
  shapeInfo.outShape[0] = 16;
  shapeInfo.outShape[1] = 18;
  shapeInfo.outShape[2] = 12544;
  shapeInfo.perm[0] =1;
  shapeInfo.perm[1] =0;
  shapeInfo.perm[2] =2;
  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  utils::OpRunInfo runInfo;
  EXPECT_EQ(shapeInfo.scenario, 12);
}

TEST_F(TransposeTilingTestV2, scenario_12_case1) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 2;
  compilerInfo.ubSize = 8192;
  runtimeInfo.ubSizeCouldUse = 7936;
  runtimeInfo.dType = ge::DT_INT16;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 4;
  shapeInfo.dim = 4;
  shapeInfo.inShape[0] = 2;
  shapeInfo.inShape[1] = 3174;
  shapeInfo.inShape[2] = 3;
  shapeInfo.inShape[3] = 80;

  shapeInfo.outShape[0] = 2;
  shapeInfo.outShape[1] = 3;
  shapeInfo.outShape[2] = 3174;
  shapeInfo.outShape[3] = 80;

  shapeInfo.perm[0] = 0;
  shapeInfo.perm[1] = 2;
  shapeInfo.perm[2] = 1;
  shapeInfo.perm[3] = 3;

  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  utils::OpRunInfo runInfo;
  EXPECT_EQ(shapeInfo.scenario, 12);
}

TEST_F(TransposeTilingTestV2, scenario_12_case2) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 2;
  compilerInfo.ubSize = 8192;
  runtimeInfo.ubSizeCouldUse = 7936;
  runtimeInfo.dType = ge::DT_INT16;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 4;
  shapeInfo.dim = 4;
  shapeInfo.inShape[0] = 2;
  shapeInfo.inShape[1] = 3;
  shapeInfo.inShape[2] = 3174;
  shapeInfo.inShape[3] = 80;

  shapeInfo.outShape[0] = 2;
  shapeInfo.outShape[1] = 3174;
  shapeInfo.outShape[2] = 3;
  shapeInfo.outShape[3] = 80;

  shapeInfo.perm[0] = 0;
  shapeInfo.perm[1] = 2;
  shapeInfo.perm[2] = 1;
  shapeInfo.perm[3] = 3;

  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  utils::OpRunInfo runInfo;
  EXPECT_EQ(shapeInfo.scenario, 12);
}

TEST_F(TransposeTilingTestV2, scenario_5_all_in) {
  TransposeCompilerInfo compilerInfo;
  ShapeInfo shapeInfo;
  RuntimeInfoV2 runtimeInfo;
  compilerInfo.coreNum = 2;
  compilerInfo.ubSize = 8192;
  runtimeInfo.ubSizeCouldUse = 7936;
  runtimeInfo.dType = ge::DT_INT16;
  runtimeInfo.fp16Times = 1;

  shapeInfo.origDim = 5;
  shapeInfo.dim = 5;
  shapeInfo.inShape[0] = 2;
  shapeInfo.inShape[1] = 3;
  shapeInfo.inShape[2] = 4;
  shapeInfo.inShape[3] = 5;
  shapeInfo.inShape[4] = 7;

  shapeInfo.outShape[0] = 2;
  shapeInfo.outShape[1] = 5;
  shapeInfo.outShape[2] = 4;
  shapeInfo.outShape[3] = 3;
  shapeInfo.outShape[4] = 7;

  shapeInfo.perm[0] = 0;
  shapeInfo.perm[1] = 3;
  shapeInfo.perm[2] = 2;
  shapeInfo.perm[3] = 1;
  shapeInfo.perm[4] = 4;

  ReduceAxisV2("Transpose", compilerInfo, shapeInfo, runtimeInfo);
  TransposeCalcTilingData(opTypeV2, compilerInfo, shapeInfo, runtimeInfo);
  utils::OpRunInfo runInfo;
  EXPECT_EQ(shapeInfo.scenario, 5);
}

const int64_t profiling_test_num = 0;
static void run_case(std::vector<int64_t> input_shape, std::vector<int64_t> output_shape,
                     std::vector<int64_t> const_value, std::string data_dtype, std::string compile_info,
                     std::string expect_tiling, std::string case_name) {
  using namespace ut_util;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Transpose");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  int64_t dim_num = input_shape.size();
  std::vector<int64_t> const_shape = {dim_num};
  // get transpose op
  auto test_op = op::Transpose("Transpose");
  TENSOR_INPUT_WITH_SHAPE(test_op, x, input_shape, StringToDtype(data_dtype), FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, perm, const_shape, DT_INT64, FORMAT_ND, const_value);
  TENSOR_OUTPUT_WITH_SHAPE(test_op, y, output_shape, StringToDtype(data_dtype), FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(test_op, iter->second, compile_info, runInfo);
  for (int64_t i = 0; i < profiling_test_num; i++) {
    RUN_TILING_V3(test_op, iter->second, compile_info, runInfo);
  }

  optiling::TransposeCompilerInfo info;
  int32_t tiling_len = 20*1024;
  TILING_PARSE_JSON_TO_COMPILEINFO("Transpose", compile_info, info);
  std::vector<bool> input_const={false, true};
  std::vector<std::string> attrs;
  ATTACH_OPERATOR_TO_HOLDER_CONST(holder, test_op, input_const, attrs, tiling_len, info);
  HOLDER_DO_TILING(holder,"Transpose",ge::GRAPH_SUCCESS);
  if (expect_tiling.size() != 0 ) {
    TILING_DATA_VERIFY_BYTYPE(holder, int64_t, expect_tiling.c_str());
  }
  
}

TEST_F(TransposeTilingTestV2, para_test_32) {
  vector<int64_t> inShape;
  vector<int64_t> outShape;
  inShape.push_back(3);
  inShape.push_back(4);
  outShape.push_back(4);
  outShape.push_back(3);
  vector<int64_t> perm = {1, 0};
  std::string data_dtype = "float16";
  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  run_case(inShape, outShape, perm, data_dtype, compileInfo, "", this->test_info_->name());
}

TEST_F(TransposeTilingTestV2, para_test_64) {
  vector<int64_t> inShape;
  vector<int64_t> outShape;
  inShape.push_back(3);
  inShape.push_back(4);
  outShape.push_back(4);
  outShape.push_back(3);
  vector<int64_t> perm = {1, 0};
  std::string data_dtype = "float16";
  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  run_case(inShape, outShape, perm, data_dtype, compileInfo, "", this->test_info_->name());
}

TEST_F(TransposeTilingTestV2, vcopy) {
  vector<int64_t> inShape;
  vector<int64_t> outShape;
  inShape.push_back(100);
  inShape.push_back(200);
  inShape.push_back(16);
  outShape.push_back(200);
  outShape.push_back(100);
  outShape.push_back(16);
  vector<int64_t> perm = {1, 0, 2};
  std::string data_dtype = "float16";
  std::string compileInfo = "{\"vars\": {\"core_num\":48, \"ub_size\":6144, \"dtype\":\"float16\"}}";
  run_case(inShape, outShape, perm, data_dtype, compileInfo, "", this->test_info_->name());
}

TEST_F(TransposeTilingTestV2, scenario_4_seri) {
  vector<int64_t> inShape;
  vector<int64_t> outShape;
  inShape.push_back(5);
  inShape.push_back(6);
  inShape.push_back(7);
  inShape.push_back(320);
  outShape.push_back(7);
  outShape.push_back(6);
  outShape.push_back(5);
  outShape.push_back(320);
  vector<int64_t> perm = {2, 1, 0, 3};
  std::string data_dtype = "float16";
  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  std::string expected = "9 16 3 1 32 8192 320 40 2 5 1640 0 320 2240 9600 1600 0 0 7 6 2 0 0 2 2 0 2 4 0 "
  "2 6 0 2 1 1 2 3 1 2 5 1 2 0 2 2 2 2 2 4 2 1 6 2 1 0 3 1 1 3 1 2 3 1 3 3 1 4 3 1 5 3 1 6 3 1 0 4 1 1 4 1 "
  "2 4 1 3 4 1 4 4 1 5 4 1 6 4 1 0 5 1 1 5 1 2 5 1 3 5 1 4 5 1 5 5 1 6 5 ";
  run_case(inShape, outShape, perm, data_dtype, compileInfo, "", this->test_info_->name());
}

TEST_F(TransposeTilingTestV2, scenario_7_seri) {
  vector<int64_t> inShape;
  vector<int64_t> outShape;
  inShape.push_back(5);
  inShape.push_back(10000);
  inShape.push_back(20000);
  outShape.push_back(5);
  outShape.push_back(20000);
  outShape.push_back(10000);
  vector<int64_t> perm = {0, 2, 1};
  std::string data_dtype = "float16";
  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  run_case(inShape, outShape, perm, data_dtype, compileInfo, "", this->test_info_->name());
}

TEST_F(TransposeTilingTestV2, scenario_7_seri_fp32) {
  vector<int64_t> inShape;
  vector<int64_t> outShape;
  inShape.push_back(5);
  inShape.push_back(10000);
  inShape.push_back(20000);
  outShape.push_back(5);
  outShape.push_back(20000);
  outShape.push_back(10000);
  vector<int64_t> perm = {0, 2, 1};
  std::string data_dtype = "float32";
  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float32\"}}";
  std::string expected = "7 12 16 0 32 8192 1 1 1 5 200000000 200000000 20000 10000 10000 20000 5 224 89 64 0 0 128 "
  "2 64 0 0 0 0 19936 0 256 5 224 89 64 0 0 128 2 64 320 0 0 0 19936 320 576 5 224 89 64 0 0 128 2 64 640 0 0 0 19936 "
  "640 896 5 224 89 64 0 0 128 2 64 960 0 0 0 19936 960 1216 5 224 89 64 0 0 128 2 64 1280 0 0 0 19936 1280 1536 5 224 "
  "89 64 0 0 128 2 64 1600 0 0 0 19936 1600 1856 5 224 89 64 0 0 128 2 64 1920 0 0 0 19936 1920 2176 5 224 89 64 0 0 128 "
  "2 64 2240 0 0 0 19936 2240 2496 5 224 89 64 0 0 128 2 64 2560 0 0 0 19936 2560 2816 5 224 89 64 0 0 128 2 64 2880 0 0 "
  "0 19936 2880 3136 5 224 89 64 0 0 128 2 64 3200 0 0 0 19936 3200 3456 5 224 89 64 0 0 128 2 64 3520 0 0 0 19936 3520 "
  "3776 5 224 89 64 0 0 128 2 64 3840 0 0 0 19936 3840 4096 5 224 89 64 0 0 128 2 64 4160 0 0 0 19936 4160 4416 5 224 89 "
  "64 0 0 128 2 64 4480 0 0 0 19936 4480 4736 5 224 89 64 0 0 128 2 64 4800 0 0 0 19936 4800 5056 5 224 89 64 0 0 128 2 "
  "64 5120 0 0 0 19936 5120 5376 5 224 89 64 0 0 128 2 64 5440 0 0 0 19936 5440 5696 5 224 89 64 0 0 128 2 64 5760 0 0 "
  "0 19936 5760 6016 5 224 89 64 0 0 128 2 64 6080 0 0 0 19936 6080 6336 5 224 89 64 0 0 128 2 64 6400 0 0 0 19936 6400 "
  "6656 5 224 89 64 0 0 128 2 64 6720 0 0 0 19936 6720 6976 5 224 89 64 0 0 128 2 64 7040 0 0 0 19936 7040 7296 5 224 "
  "89 64 0 0 128 2 64 7360 0 0 0 19936 7360 7616 5 224 89 64 0 0 128 2 64 7680 0 0 0 19936 7680 7936 5 224 89 64 0 0 "
  "128 2 64 8000 0 0 0 19936 8000 8256 5 224 89 64 0 0 128 2 64 8320 0 0 0 19936 8320 8576 5 224 89 64 0 0 128 2 64 "
  "8640 0 0 0 19936 8640 8896 5 224 89 64 0 0 128 2 64 8960 0 0 0 19936 8960 9216 5 224 89 64 0 0 128 2 64 9280 0 0 "
  "0 19936 9280 9536 5 224 89 64 0 0 128 2 64 9600 0 0 0 19936 9600 9856 5 224 89 64 0 0 80 1 0 9920 0 0 0 19936 9920 0 ";
  run_case(inShape, outShape, perm, data_dtype, compileInfo, expected, this->test_info_->name());
}

TEST_F(TransposeTilingTestV2, scenario_9_seri) {
  vector<int64_t> inShape;
  vector<int64_t> outShape;
  inShape.push_back(100);
  inShape.push_back(200);
  inShape.push_back(128);
  outShape.push_back(200);
  outShape.push_back(100);
  outShape.push_back(128);
  vector<int64_t> perm = {1, 0, 2};
  std::string data_dtype = "float16";
  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  run_case(inShape, outShape, perm, data_dtype, compileInfo, "", this->test_info_->name());
}

TEST_F(TransposeTilingTestV2, scenario_11_seri) {
  vector<int64_t> inShape;
  vector<int64_t> outShape;
  inShape.push_back(1024);
  inShape.push_back(768);
  outShape.push_back(768);
  outShape.push_back(1024);
  vector<int64_t> perm = {1, 0};
  std::string data_dtype = "float16";
  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  run_case(inShape, outShape, perm, data_dtype, compileInfo, "", this->test_info_->name());
}

TEST_F(TransposeTilingTestV2, scenario_0_seri_fov_cov) {
  vector<int64_t> inShape;
  vector<int64_t> outShape;
  inShape.push_back(1024);
  outShape.push_back(1024);
  vector<int64_t> perm = {0};
  std::string data_dtype = "float16";
  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  run_case(inShape, outShape, perm, data_dtype, compileInfo, "", this->test_info_->name());
}

TEST_F(TransposeTilingTestV2, scenario_0_seri_all_one_fov_cov) {
  vector<int64_t> inShape;
  vector<int64_t> outShape;
  inShape.push_back(1);
  inShape.push_back(1);
  outShape.push_back(1);
  outShape.push_back(1);
  vector<int64_t> perm = {1,0};
  std::string data_dtype = "float16";
  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  run_case(inShape, outShape, perm, data_dtype, compileInfo, "", this->test_info_->name());
}

TEST_F(TransposeTilingTestV2, spilit_on_n_fp16_axis_for_cov) {
  vector<int64_t> inShape;
  vector<int64_t> outShape;
  inShape.push_back(1000);
  inShape.push_back(257);
  inShape.push_back(129);
  outShape.push_back(1000);
  outShape.push_back(129);
  outShape.push_back(257);
  vector<int64_t> perm = {0,2,1};
  std::string data_dtype = "float16";
  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  run_case(inShape, outShape, perm, data_dtype, compileInfo, "", this->test_info_->name());
}

TEST_F(TransposeTilingTestV2, spilit_on_n_fp32_axis_for_cov) {
  vector<int64_t> inShape;
  vector<int64_t> outShape;
  inShape.push_back(1000);
  inShape.push_back(257);
  inShape.push_back(129);
  outShape.push_back(1000);
  outShape.push_back(129);
  outShape.push_back(257);
  vector<int64_t> perm = {0,2,1};
  std::string data_dtype = "float32";
  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  run_case(inShape, outShape, perm, data_dtype, compileInfo, "", this->test_info_->name());
}

TEST_F(TransposeTilingTestV2, spilit_on_n_int64_axis_for_cov) {
  vector<int64_t> inShape;
  vector<int64_t> outShape;
  inShape.push_back(1000);
  inShape.push_back(22257);
  inShape.push_back(11129);
  outShape.push_back(1000);
  outShape.push_back(11129);
  outShape.push_back(22257);
  vector<int64_t> perm = {0,2,1};
  std::string data_dtype = "float32";
  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  run_case(inShape, outShape, perm, data_dtype, compileInfo, "", this->test_info_->name());
}

TEST_F(TransposeTilingTestV2, scenario_0) {
  vector<int64_t> inShape;
  vector<int64_t> outShape;
  inShape.push_back(11129);
  inShape.push_back(22257);
  outShape.push_back(11129);
  outShape.push_back(22257);
  vector<int64_t> perm = {0,1};
  std::string data_dtype = "float32";
  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  run_case(inShape, outShape, perm, data_dtype, compileInfo, "", this->test_info_->name());
}

TEST_F(TransposeTilingTestV2, scenario_1) {
  vector<int64_t> inShape;
  vector<int64_t> outShape;
  inShape.push_back(64);
  inShape.push_back(32);
  inShape.push_back(22222);
  outShape.push_back(32);
  outShape.push_back(64);
  outShape.push_back(22222);
  vector<int64_t> perm = {1,0,2};
  std::string data_dtype = "float16";
  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  run_case(inShape, outShape, perm, data_dtype, compileInfo, "", this->test_info_->name());
}

TEST_F(TransposeTilingTestV2, scenario_5) {
  vector<int64_t> inShape;
  vector<int64_t> outShape;
  inShape.push_back(9);
  inShape.push_back(8);
  inShape.push_back(7);
  inShape.push_back(6);
  inShape.push_back(5);
  inShape.push_back(4);
  outShape.push_back(4);
  outShape.push_back(5);
  outShape.push_back(6);
  outShape.push_back(7);
  outShape.push_back(8);
  outShape.push_back(9);
  vector<int64_t> perm = {5,4,3,2,1,0};
  std::string data_dtype = "int8";
  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  run_case(inShape, outShape, perm, data_dtype, compileInfo, "", this->test_info_->name());
}

TEST_F(TransposeTilingTestV2, scenario_10) {
  vector<int64_t> inShape;
  vector<int64_t> outShape;
  inShape.push_back(4);
  inShape.push_back(64);
  inShape.push_back(128);
  outShape.push_back(4);
  outShape.push_back(128);
  outShape.push_back(64);
  vector<int64_t> perm = {0,2,1};
  std::string data_dtype = "float16";
  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  run_case(inShape, outShape, perm, data_dtype, compileInfo, "", this->test_info_->name());
}

TEST_F(TransposeTilingTestV2, scenario_all_one) {
  vector<int64_t> inShape;
  vector<int64_t> outShape;
  inShape.push_back(1);
  inShape.push_back(1);
  outShape.push_back(1);
  outShape.push_back(1);
  vector<int64_t> perm = {1, 0};
  std::string data_dtype = "float16";
  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  run_case(inShape, outShape, perm, data_dtype, compileInfo, "", this->test_info_->name());
}

TEST_F(TransposeTilingTestV2, scenario_large_burstlen_core_gt_axis) {
  vector<int64_t> inShape;
  vector<int64_t> outShape;
  inShape.push_back(2);
  inShape.push_back(4);
  inShape.push_back(4096);
  outShape.push_back(4);
  outShape.push_back(2);
  outShape.push_back(4096);
  vector<int64_t> perm = {1, 0, 2};
  std::string data_dtype = "float16";
  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  run_case(inShape, outShape, perm, data_dtype, compileInfo, "", this->test_info_->name());
}

TEST_F(TransposeTilingTestV2, scenario_3) {
  vector<int64_t> inShape;
  vector<int64_t> outShape;
  inShape.push_back(20);
  inShape.push_back(4);
  inShape.push_back(409600);
  outShape.push_back(4);
  outShape.push_back(20);
  outShape.push_back(409600);
  vector<int64_t> perm = {1, 0, 2};
  std::string data_dtype = "float16";
  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  run_case(inShape, outShape, perm, data_dtype, compileInfo, "", this->test_info_->name());
}

TEST_F(TransposeTilingTestV2, scenario_12_seri) {
  vector<int64_t> inShape;
  vector<int64_t> outShape;
  inShape.push_back(2);
  inShape.push_back(16);
  inShape.push_back(64);
  inShape.push_back(12544);
  outShape.push_back(2);
  outShape.push_back(64);
  outShape.push_back(16);
  outShape.push_back(12544);
  vector<int64_t> perm = {0, 2, 1, 3};
  std::string data_dtype = "float16";
  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  run_case(inShape, outShape, perm, data_dtype, compileInfo, "", this->test_info_->name());
}

TEST_F(TransposeTilingTestV2, scenario_12_seri_1) {
  vector<int64_t> inShape;
  vector<int64_t> outShape;
  inShape.push_back(2);
  inShape.push_back(64);
  inShape.push_back(16);
  inShape.push_back(12544);
  outShape.push_back(2);
  outShape.push_back(16);
  outShape.push_back(64);
  outShape.push_back(12544);
  vector<int64_t> perm = {0, 2, 1, 3};
  std::string data_dtype = "float16";
  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  run_case(inShape, outShape, perm, data_dtype, compileInfo, "", this->test_info_->name());
}
