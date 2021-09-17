#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <register/op_tiling.h>
#include "op_tiling/transpose.h"

using namespace std;
using namespace optiling;

string opType = "transpose";

class TransposeTilingTest : public testing::Test {
    protected:
    static void SetUpTestCase() {
            //std::cout << "TransposeTilingTest setup";
        }
    static void TearDownTestCase() {
            //std::cout << "TransposeTilingTest teardown";
        }
};

TEST_F(TransposeTilingTest, reduce_axis_merge) {
    /*
     *  inShape         perm                   reducedInShape       reducedOutShape       reducedPerm
     *  ---------------------------------------------------------------------------------------------------
     *  4,5,6,7         1,0,2,3                4,5,42               5,4,42                1,0,2
     *
     */
    ShapeInfo shapeInfo1;
    shapeInfo1.reducedInShape.push_back(4);
    shapeInfo1.reducedInShape.push_back(5);
    shapeInfo1.reducedInShape.push_back(6);
    shapeInfo1.reducedInShape.push_back(7);
    shapeInfo1.reducedPerm.push_back(1);
    shapeInfo1.reducedPerm.push_back(0);
    shapeInfo1.reducedPerm.push_back(2);
    shapeInfo1.reducedPerm.push_back(3);
    MergeAxis(shapeInfo1);
    EXPECT_EQ(shapeInfo1.reducedInShape.size(), 3);
    EXPECT_EQ(shapeInfo1.reducedOutShape.size(), 3);
    EXPECT_EQ(shapeInfo1.reducedPerm.size(), 3);
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
    ShapeInfo shapeInfo2;
    shapeInfo2.reducedInShape.push_back(2);
    shapeInfo2.reducedInShape.push_back(3);
    shapeInfo2.reducedInShape.push_back(4);
    shapeInfo2.reducedInShape.push_back(5);
    shapeInfo2.reducedInShape.push_back(6);
    shapeInfo2.reducedPerm.push_back(0);
    shapeInfo2.reducedPerm.push_back(4);
    shapeInfo2.reducedPerm.push_back(1);
    shapeInfo2.reducedPerm.push_back(2);
    shapeInfo2.reducedPerm.push_back(3);
    MergeAxis(shapeInfo2);
    for(int i = 0; i < shapeInfo2.reducedPerm.size(); i++) {
        cout<<"xxxxx : "<<shapeInfo2.reducedPerm[i]<<endl;
    }
    EXPECT_EQ(shapeInfo2.reducedInShape.size(), 3);
    EXPECT_EQ(shapeInfo2.reducedOutShape.size(), 3);
    EXPECT_EQ(shapeInfo2.reducedPerm.size(), 3);
    EXPECT_EQ(shapeInfo2.reducedInShape[0], 2);
    EXPECT_EQ(shapeInfo2.reducedInShape[1], 60);
    EXPECT_EQ(shapeInfo2.reducedInShape[2], 6);
    EXPECT_EQ(shapeInfo2.reducedOutShape[0], 2);
    EXPECT_EQ(shapeInfo2.reducedOutShape[1], 6);
    EXPECT_EQ(shapeInfo2.reducedOutShape[2], 60);
    EXPECT_EQ(shapeInfo2.reducedPerm[0], 0);
    EXPECT_EQ(shapeInfo2.reducedPerm[1], 2);
    EXPECT_EQ(shapeInfo2.reducedPerm[2], 1);

    /*
     *  inShape         perm                   reducedInShape       reducedOutShape       reducedPerm
     *  ---------------------------------------------------------------------------------------------------
     *  2,3,4,5,6       2,3,4,0,1              6,120                120,6                 1,0
     *
     */
    ShapeInfo shapeInfo3;
    shapeInfo3.reducedInShape.push_back(2);
    shapeInfo3.reducedInShape.push_back(3);
    shapeInfo3.reducedInShape.push_back(4);
    shapeInfo3.reducedInShape.push_back(5);
    shapeInfo3.reducedInShape.push_back(6);
    shapeInfo3.reducedPerm.push_back(2);
    shapeInfo3.reducedPerm.push_back(3);
    shapeInfo3.reducedPerm.push_back(4);
    shapeInfo3.reducedPerm.push_back(0);
    shapeInfo3.reducedPerm.push_back(1);
    MergeAxis(shapeInfo3);
    EXPECT_EQ(shapeInfo3.reducedInShape.size(), 2);
    EXPECT_EQ(shapeInfo3.reducedOutShape.size(), 2);
    EXPECT_EQ(shapeInfo3.reducedPerm.size(), 2);
    EXPECT_EQ(shapeInfo3.reducedInShape[0], 6);
    EXPECT_EQ(shapeInfo3.reducedInShape[1], 120);
    EXPECT_EQ(shapeInfo3.reducedOutShape[0],120);
    EXPECT_EQ(shapeInfo3.reducedOutShape[1], 6);
    EXPECT_EQ(shapeInfo3.reducedPerm[0], 1);
    EXPECT_EQ(shapeInfo3.reducedPerm[1], 0);

}

TEST_F(TransposeTilingTest, reduce_axis_remove) {
    /*
     *  inShape         perm                   reducedInShape       reducedOutShape       reducedPerm
     *  ---------------------------------------------------------------------------------------------------
     *  4,1,6,1         0,1,2,3                4,6                  4,6                   0,1
     *
     */
    ShapeInfo shapeInfo1;
    shapeInfo1.inShape.push_back(4);
    shapeInfo1.inShape.push_back(1);
    shapeInfo1.inShape.push_back(6);
    shapeInfo1.inShape.push_back(1);
    shapeInfo1.perm.push_back(0);
    shapeInfo1.perm.push_back(1);
    shapeInfo1.perm.push_back(2);
    shapeInfo1.perm.push_back(3);
    RemoveAxis(shapeInfo1);
    EXPECT_EQ(shapeInfo1.reducedInShape.size(), 2);
    EXPECT_EQ(shapeInfo1.reducedOutShape.size(), 2);
    EXPECT_EQ(shapeInfo1.reducedPerm.size(), 2);
    EXPECT_EQ(shapeInfo1.reducedInShape[0], 4);
    EXPECT_EQ(shapeInfo1.reducedInShape[1], 6);
    EXPECT_EQ(shapeInfo1.reducedOutShape[0],4);
    EXPECT_EQ(shapeInfo1.reducedOutShape[1],6);
    EXPECT_EQ(shapeInfo1.reducedPerm[0], 0);
    EXPECT_EQ(shapeInfo1.reducedPerm[1], 1);

    /*
     *  inShape         perm                   reducedInShape       reducedOutShape       reducedPerm
     *  ---------------------------------------------------------------------------------------------------
     *  4,1,6,1         1,2,0,3                4,6                  6,4                   1,0
     *
     */
    ShapeInfo shapeInfo2;
    shapeInfo2.inShape.push_back(4);
    shapeInfo2.inShape.push_back(1);
    shapeInfo2.inShape.push_back(6);
    shapeInfo2.inShape.push_back(1);
    shapeInfo2.perm.push_back(1);
    shapeInfo2.perm.push_back(2);
    shapeInfo2.perm.push_back(0);
    shapeInfo2.perm.push_back(3);
    RemoveAxis(shapeInfo2);
    EXPECT_EQ(shapeInfo2.reducedInShape.size(), 2);
    EXPECT_EQ(shapeInfo2.reducedOutShape.size(), 2);
    EXPECT_EQ(shapeInfo2.reducedPerm.size(), 2);
    EXPECT_EQ(shapeInfo2.reducedInShape[0], 4);
    EXPECT_EQ(shapeInfo2.reducedInShape[1], 6);
    EXPECT_EQ(shapeInfo2.reducedOutShape[0],6);
    EXPECT_EQ(shapeInfo2.reducedOutShape[1], 4);
    EXPECT_EQ(shapeInfo2.reducedPerm[0], 1);
    EXPECT_EQ(shapeInfo2.reducedPerm[1], 0);


    /*
     *  inShape         perm                   reducedInShape       reducedOutShape       reducedPerm
     *  ---------------------------------------------------------------------------------------------------
     *  1,1,4,6         3,2,0,1                4,6                  6,4                   1,0
     *
     */
    ShapeInfo shapeInfo3;
    shapeInfo3.inShape.push_back(1);
    shapeInfo3.inShape.push_back(1);
    shapeInfo3.inShape.push_back(4);
    shapeInfo3.inShape.push_back(6);
    shapeInfo3.perm.push_back(3);
    shapeInfo3.perm.push_back(2);
    shapeInfo3.perm.push_back(0);
    shapeInfo3.perm.push_back(1);
    RemoveAxis(shapeInfo3);
    EXPECT_EQ(shapeInfo3.reducedInShape.size(), 2);
    EXPECT_EQ(shapeInfo3.reducedOutShape.size(), 2);
    EXPECT_EQ(shapeInfo3.reducedPerm.size(), 2);
    EXPECT_EQ(shapeInfo3.reducedInShape[0], 4);
    EXPECT_EQ(shapeInfo3.reducedInShape[1], 6);
    EXPECT_EQ(shapeInfo3.reducedOutShape[0],6);
    EXPECT_EQ(shapeInfo3.reducedOutShape[1], 4);
    EXPECT_EQ(shapeInfo3.reducedPerm[0], 1);
    EXPECT_EQ(shapeInfo3.reducedPerm[1], 0);

    /*
     *  inShape         perm                   reducedInShape       reducedOutShape       reducedPerm
     *  ---------------------------------------------------------------------------------------------------
     *  4,6,1,1         3,2,0,1                4,6                  4,6                   0,1
     *
     */
    ShapeInfo shapeInfo4;
    shapeInfo4.inShape.push_back(4);
    shapeInfo4.inShape.push_back(6);
    shapeInfo4.inShape.push_back(1);
    shapeInfo4.inShape.push_back(1);
    shapeInfo4.perm.push_back(3);
    shapeInfo4.perm.push_back(2);
    shapeInfo4.perm.push_back(0);
    shapeInfo4.perm.push_back(1);
    RemoveAxis(shapeInfo4);
    EXPECT_EQ(shapeInfo4.reducedInShape.size(), 2);
    EXPECT_EQ(shapeInfo4.reducedOutShape.size(), 2);
    EXPECT_EQ(shapeInfo4.reducedPerm.size(), 2);
    EXPECT_EQ(shapeInfo4.reducedInShape[0], 4);
    EXPECT_EQ(shapeInfo4.reducedInShape[1], 6);
    EXPECT_EQ(shapeInfo4.reducedOutShape[0], 4);
    EXPECT_EQ(shapeInfo4.reducedOutShape[1], 6);
    EXPECT_EQ(shapeInfo4.reducedPerm[0], 0);
    EXPECT_EQ(shapeInfo4.reducedPerm[1], 1);
}

TEST_F(TransposeTilingTest, stride_lt_65535) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(7000);
    shapeInfo.inShape.push_back(32768);
    shapeInfo.inShape.push_back(8);
    shapeInfo.outShape.push_back(32768);
    shapeInfo.outShape.push_back(7000);
    shapeInfo.outShape.push_back(8);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(2);

    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    EXPECT_TRUE(TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo));
}

TEST_F(TransposeTilingTest, identical_shape) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(1000);
    shapeInfo.inShape.push_back(2000);
    shapeInfo.outShape.push_back(1000);
    shapeInfo.outShape.push_back(2000);
    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(1);
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    EXPECT_TRUE(TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo));
}

TEST_F(TransposeTilingTest, small_shape) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(10);
    shapeInfo.inShape.push_back(20);
    shapeInfo.outShape.push_back(20);
    shapeInfo.outShape.push_back(10);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(0);
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    EXPECT_TRUE(TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo));
}

TEST_F(TransposeTilingTest, stride_gt_65535) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(7000);
    shapeInfo.inShape.push_back(32768);
    shapeInfo.inShape.push_back(16);
    shapeInfo.outShape.push_back(32768);
    shapeInfo.outShape.push_back(7000);
    shapeInfo.outShape.push_back(16);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(2);
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    EXPECT_TRUE(TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo));
}

TEST_F(TransposeTilingTest, huge_last_axis) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(7000);
    shapeInfo.inShape.push_back(32768);
    shapeInfo.inShape.push_back(160000);
    shapeInfo.outShape.push_back(32768);
    shapeInfo.outShape.push_back(7000);
    shapeInfo.outShape.push_back(160000);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(2);
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    EXPECT_TRUE(TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo));
}

TEST_F(TransposeTilingTest, last_axis_join_transpose_2d) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(7000);
    shapeInfo.inShape.push_back(6000);
    shapeInfo.outShape.push_back(6000);
    shapeInfo.outShape.push_back(7000);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(0);
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);

    EXPECT_EQ(runtimeInfo.infoPerCore[0].infoCol.colPerMC, 128);
    EXPECT_EQ(runtimeInfo.infoPerCore[0].infoCol.loopOnMC, 46);
    EXPECT_EQ(runtimeInfo.infoPerCore[0].infoCol.colTC, 112);
    EXPECT_EQ(runtimeInfo.infoPerCore[0].infoCol.colOffset, 0);
    EXPECT_EQ(runtimeInfo.infoPerCore[0].infoCol.backStepLeft, 0);

    EXPECT_EQ(runtimeInfo.infoPerCore[0].infoRow.rowPerMR, 128);
    EXPECT_EQ(runtimeInfo.infoPerCore[0].infoRow.loopOnMR, 1);
    EXPECT_EQ(runtimeInfo.infoPerCore[0].infoRow.rowTR, 96);
    EXPECT_EQ(runtimeInfo.infoPerCore[0].infoRow.rowOffset, 0);
    EXPECT_EQ(runtimeInfo.infoPerCore[0].infoRow.backStepUp, 0);

    EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.colPerMC, 128);
    EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.loopOnMC, 46);
    EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.colTC, 112);
    EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.colOffset, 0);
    EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.backStepLeft, 0);

    EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.rowPerMR, 128);
    EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.loopOnMR, 1);
    EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.rowTR, 96);
    EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.rowOffset, 224);
    EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.backStepUp, 0);
}

TEST_F(TransposeTilingTest, last_axis_join_transpose_with_multi_src_axis) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(4);
    shapeInfo.inShape.push_back(128);
    shapeInfo.inShape.push_back(8);
    shapeInfo.inShape.push_back(224);

    shapeInfo.outShape.push_back(224);
    shapeInfo.outShape.push_back(8);
    shapeInfo.outShape.push_back(128);
    shapeInfo.outShape.push_back(4);

    shapeInfo.perm.push_back(3);
    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(0);
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);

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

TEST_F(TransposeTilingTest, last_axis_join_transpose_with_tail_row) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(5);
    shapeInfo.inShape.push_back(128);
    shapeInfo.inShape.push_back(8);
    shapeInfo.inShape.push_back(224);

    shapeInfo.outShape.push_back(224);
    shapeInfo.outShape.push_back(8);
    shapeInfo.outShape.push_back(128);
    shapeInfo.outShape.push_back(5);

    shapeInfo.perm.push_back(3);
    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(0);
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);

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

TEST_F(TransposeTilingTest, last_axis_join_transpose_col_and_row_with_multi_axis) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(2);
    shapeInfo.inShape.push_back(3);
    shapeInfo.inShape.push_back(4);
    shapeInfo.inShape.push_back(128);
    shapeInfo.inShape.push_back(8);
    shapeInfo.inShape.push_back(224);

    shapeInfo.outShape.push_back(2);
    shapeInfo.outShape.push_back(3);
    shapeInfo.outShape.push_back(224);
    shapeInfo.outShape.push_back(8);
    shapeInfo.outShape.push_back(128);
    shapeInfo.outShape.push_back(4);

    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(5);
    shapeInfo.perm.push_back(4);
    shapeInfo.perm.push_back(3);
    shapeInfo.perm.push_back(2);
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);

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

TEST_F(TransposeTilingTest, last_axis_join_transpose) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(5);
    shapeInfo.inShape.push_back(6);
    shapeInfo.inShape.push_back(7);
    shapeInfo.inShape.push_back(8);
    shapeInfo.inShape.push_back(90);

    shapeInfo.outShape.push_back(5);
    shapeInfo.outShape.push_back(90);
    shapeInfo.outShape.push_back(8);
    shapeInfo.outShape.push_back(7);
    shapeInfo.outShape.push_back(6);

    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(4);
    shapeInfo.perm.push_back(3);
    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(1);
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);

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


TEST_F(TransposeTilingTest, last_axis_join_transpose_t2f) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 8192;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(20000);
    shapeInfo.inShape.push_back(7);
    shapeInfo.outShape.push_back(7);
    shapeInfo.outShape.push_back(20000);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(0);
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);

    EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.rowPerMR, 288);
}

TEST_F(TransposeTilingTest, last_axis_join_transpose_t2f_fp16) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 8192;
    compilerInfo.dType ="float16";
    compilerInfo.fp16Times = 1;

    shapeInfo.inShape.push_back(20000);
    shapeInfo.inShape.push_back(7);
    shapeInfo.outShape.push_back(7);
    shapeInfo.outShape.push_back(20000);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(0);
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);

    EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.rowPerMR, 576);
}

TEST_F(TransposeTilingTest, last_axis_join_transpose_f2t) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 8192;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 1;

    shapeInfo.inShape.push_back(2);
    shapeInfo.inShape.push_back(30000);
    shapeInfo.outShape.push_back(30000);
    shapeInfo.outShape.push_back(2);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(0);
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);

    EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.colPerMC, 944);
}

TEST_F(TransposeTilingTest, last_axis_join_transpose_f2t_fp16) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 8192;
    compilerInfo.dType ="float16";
    compilerInfo.fp16Times = 1;

    shapeInfo.inShape.push_back(2);
    shapeInfo.inShape.push_back(60000);
    shapeInfo.outShape.push_back(60000);
    shapeInfo.outShape.push_back(2);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(0);
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);

    EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.colPerMC, 944*2);
}

TEST_F(TransposeTilingTest, small_shape_check_split_with_unit_ele_per_block) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 8192;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(8);
    shapeInfo.inShape.push_back(6);
    shapeInfo.inShape.push_back(1);
    shapeInfo.inShape.push_back(3);
    shapeInfo.outShape.push_back(1);
    shapeInfo.outShape.push_back(8);
    shapeInfo.outShape.push_back(6);
    shapeInfo.outShape.push_back(3);
    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(3);
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);

    EXPECT_EQ(runtimeInfo.infoPerCoreIdentical[0].base, 0);
    EXPECT_EQ(runtimeInfo.infoPerCoreIdentical[0].eleNum, 8);
    EXPECT_EQ(runtimeInfo.infoPerCoreIdentical[1].base, 8);
    EXPECT_EQ(runtimeInfo.infoPerCoreIdentical[1].eleNum, 8);
    EXPECT_EQ(runtimeInfo.infoPerCoreIdentical[19].base, 0);
    EXPECT_EQ(runtimeInfo.infoPerCoreIdentical[19].eleNum, 0);
}

TEST_F(TransposeTilingTest, small_shape_check_split_with_no_minus) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 1;
    compilerInfo.ubSize = 8192;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(6);
    shapeInfo.inShape.push_back(4);
    shapeInfo.inShape.push_back(9);
    shapeInfo.inShape.push_back(1);
    shapeInfo.outShape.push_back(9);
    shapeInfo.outShape.push_back(6);
    shapeInfo.outShape.push_back(4);
    shapeInfo.outShape.push_back(1);
    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(3);
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);

    EXPECT_EQ(runtimeInfo.infoPerCoreLastAxisNT[0].num, 216);
}

TEST_F(TransposeTilingTest, less_than_one_block) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 8192;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(1);
    shapeInfo.inShape.push_back(1);
    shapeInfo.inShape.push_back(7);
    shapeInfo.inShape.push_back(1);
    shapeInfo.outShape.push_back(7);
    shapeInfo.outShape.push_back(1);
    shapeInfo.outShape.push_back(1);
    shapeInfo.outShape.push_back(1);
    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(3);
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);

    EXPECT_EQ(runtimeInfo.infoPerCoreIdentical[0].eleNum, 7);
    EXPECT_EQ(runtimeInfo.infoPerCoreIdentical[1].eleNum, 0);
}

TEST_F(TransposeTilingTest, borrow_src_1_dst_1_axis) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 8192;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(100);
    shapeInfo.inShape.push_back(20);
    shapeInfo.inShape.push_back(3);
    shapeInfo.inShape.push_back(400);
    shapeInfo.inShape.push_back(8);

    shapeInfo.outShape.push_back(400);
    shapeInfo.outShape.push_back(3);
    shapeInfo.outShape.push_back(20);
    shapeInfo.outShape.push_back(100);
    shapeInfo.outShape.push_back(8);

    shapeInfo.perm.push_back(3);
    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(4);

    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);
	//EXPECT_EQ(shapeInfo.reducedInShape.size(), 1);
}

TEST_F(TransposeTilingTest, borrow_src_2_dst_2_dup_axis) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 8192;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(10);
    shapeInfo.inShape.push_back(3000);
    shapeInfo.inShape.push_back(20);
    shapeInfo.inShape.push_back(8);

    shapeInfo.outShape.push_back(20);
    shapeInfo.outShape.push_back(3000);
    shapeInfo.outShape.push_back(10);
    shapeInfo.outShape.push_back(8);

    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(3);

    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);
    //EXPECT_EQ(shapeInfo.reducedInShape.size(), 1);
}

TEST_F(TransposeTilingTest, borrow_src_2_dst_2_dup_axis_with_tail) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 8192;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(10);
    shapeInfo.inShape.push_back(3003);
    shapeInfo.inShape.push_back(20);
    shapeInfo.inShape.push_back(8);

    shapeInfo.outShape.push_back(20);
    shapeInfo.outShape.push_back(3003);
    shapeInfo.outShape.push_back(10);
    shapeInfo.outShape.push_back(8);

    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(3);

    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);
    //EXPECT_EQ(shapeInfo.reducedInShape.size(), 1);
}

TEST_F(TransposeTilingTest, borrow_src_2_dst_2_dup_axis_vcopy) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 48;
    compilerInfo.ubSize = 6144;
    compilerInfo.dType ="int32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(6);
    shapeInfo.inShape.push_back(4);
    shapeInfo.inShape.push_back(1280);
    shapeInfo.inShape.push_back(8);
    shapeInfo.inShape.push_back(3);

    shapeInfo.outShape.push_back(6);
    shapeInfo.outShape.push_back(8);
    shapeInfo.outShape.push_back(1280);
    shapeInfo.outShape.push_back(4);
    shapeInfo.outShape.push_back(3);

    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(3);
    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(4);

    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);
    //EXPECT_EQ(shapeInfo.reducedInShape.size(), 1);
}

TEST_F(TransposeTilingTest, borrow_src_2_dst_2_dup_axis_data_move) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 8192;
    compilerInfo.dType ="int32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(6);
    shapeInfo.inShape.push_back(8);
    shapeInfo.inShape.push_back(1800);
    shapeInfo.inShape.push_back(8);
    shapeInfo.inShape.push_back(3);

    shapeInfo.outShape.push_back(6);
    shapeInfo.outShape.push_back(8);
    shapeInfo.outShape.push_back(1800);
    shapeInfo.outShape.push_back(8);
    shapeInfo.outShape.push_back(3);

    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(3);
    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(4);

    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);
    //EXPECT_EQ(shapeInfo.reducedInShape.size(), 1);
}

TEST_F(TransposeTilingTest, borrow_dup_one_axis_data_move) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 8192;
    compilerInfo.dType ="int32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(7000);
    shapeInfo.inShape.push_back(9);
    shapeInfo.inShape.push_back(8);

    shapeInfo.outShape.push_back(9);
    shapeInfo.outShape.push_back(7000);
    shapeInfo.outShape.push_back(8);

    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(2);

    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);
    //EXPECT_EQ(shapeInfo.reducedInShape.size(), 1);
}

TEST_F(TransposeTilingTest, borrow_src_2_dst_2_dup_not_aligned) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 8192;
    compilerInfo.dType ="int32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(6);
    shapeInfo.inShape.push_back(4);
    shapeInfo.inShape.push_back(1800);
    shapeInfo.inShape.push_back(8);
    shapeInfo.inShape.push_back(3);

    shapeInfo.outShape.push_back(6);
    shapeInfo.outShape.push_back(8);
    shapeInfo.outShape.push_back(1800);
    shapeInfo.outShape.push_back(4);
    shapeInfo.outShape.push_back(3);

    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(3);
    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(4);

    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);
    //EXPECT_EQ(shapeInfo.reducedInShape.size(), 1);
}


TEST_F(TransposeTilingTest, borrow_src_2_dst_2_axis) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 8192;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(2);
    shapeInfo.inShape.push_back(30);
    shapeInfo.inShape.push_back(40);
    shapeInfo.inShape.push_back(5);
    shapeInfo.inShape.push_back(8);

    shapeInfo.outShape.push_back(5);
    shapeInfo.outShape.push_back(40);
    shapeInfo.outShape.push_back(30);
    shapeInfo.outShape.push_back(2);
    shapeInfo.outShape.push_back(8);

    shapeInfo.perm.push_back(3);
    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(4);

    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);
	//EXPECT_EQ(shapeInfo.reducedInShape.size(), 1);
}

TEST_F(TransposeTilingTest, borrow_src_2_dst_2_axis_with_6_axis) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 8192;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(2);
    shapeInfo.inShape.push_back(50);
    shapeInfo.inShape.push_back(9);
    shapeInfo.inShape.push_back(23);
    shapeInfo.inShape.push_back(40);
    shapeInfo.inShape.push_back(5);
    shapeInfo.inShape.push_back(8);

    shapeInfo.outShape.push_back(5);
    shapeInfo.outShape.push_back(40);
    shapeInfo.outShape.push_back(23);
    shapeInfo.outShape.push_back(9);
    shapeInfo.outShape.push_back(50);
    shapeInfo.outShape.push_back(2);
    shapeInfo.outShape.push_back(8);

    shapeInfo.perm.push_back(5);
    shapeInfo.perm.push_back(4);
    shapeInfo.perm.push_back(3);
    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(6);

    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);
	//EXPECT_EQ(shapeInfo.reducedInShape.size(), 1);
}

TEST_F(TransposeTilingTest, borrow_2dup_axis) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 8192;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(100);
    shapeInfo.inShape.push_back(70);
    shapeInfo.inShape.push_back(9);
    shapeInfo.inShape.push_back(8);

    shapeInfo.outShape.push_back(100);
    shapeInfo.outShape.push_back(9);
    shapeInfo.outShape.push_back(70);
    shapeInfo.outShape.push_back(8);

    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(3);

    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);
	//EXPECT_EQ(shapeInfo.reducedInShape.size(), 1);
}

TEST_F(TransposeTilingTest, temp) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 2;
    compilerInfo.ubSize = 8192;
    compilerInfo.dType ="float16";
    compilerInfo.fp16Times = 1;

    shapeInfo.inShape.push_back(5);
    shapeInfo.inShape.push_back(4);
    shapeInfo.inShape.push_back(500);
    shapeInfo.inShape.push_back(1);
    shapeInfo.inShape.push_back(81);
    shapeInfo.inShape.push_back(1);

    shapeInfo.outShape.push_back(4);
    shapeInfo.outShape.push_back(500);
    shapeInfo.outShape.push_back(1);
    shapeInfo.outShape.push_back(81);
    shapeInfo.outShape.push_back(1);
    shapeInfo.outShape.push_back(5);

    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(3);
    shapeInfo.perm.push_back(4);
    shapeInfo.perm.push_back(5);
    shapeInfo.perm.push_back(0);

    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);
}

TEST_F(TransposeTilingTest, small_shape_all_one) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 8192;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(1);
    shapeInfo.inShape.push_back(1);
    shapeInfo.inShape.push_back(1);
    shapeInfo.inShape.push_back(1);
    shapeInfo.outShape.push_back(1);
    shapeInfo.outShape.push_back(1);
    shapeInfo.outShape.push_back(1);
    shapeInfo.outShape.push_back(1);
    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(3);
    shapeInfo.perm.push_back(2);
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);
	EXPECT_EQ(shapeInfo.reducedInShape.size(), 1);
}

TEST_F(TransposeTilingTest, nchw) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 8192;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(6);
    shapeInfo.inShape.push_back(7);
    shapeInfo.inShape.push_back(4);
    shapeInfo.inShape.push_back(7);
    shapeInfo.outShape.push_back(4);
    shapeInfo.outShape.push_back(6);
    shapeInfo.outShape.push_back(7);
    shapeInfo.outShape.push_back(7);
    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(3);
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);

    //EXPECT_EQ(runtimeInfo.infoPerCoreLastAxisNT[1].base, 6);
    //EXPECT_EQ(runtimeInfo.infoPerCoreLastAxisNT[1].num, 6);
}

TEST_F(TransposeTilingTest, ub310) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 7936;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(4096);
    shapeInfo.inShape.push_back(8192);
    shapeInfo.outShape.push_back(8192);
    shapeInfo.outShape.push_back(4096);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(0);
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    EXPECT_TRUE(TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo));
}

TEST_F(TransposeTilingTest, ublihisi) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 6144;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(4096);
    shapeInfo.inShape.push_back(8192);
    shapeInfo.outShape.push_back(8192);
    shapeInfo.outShape.push_back(4096);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(0);
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    EXPECT_TRUE(TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo));
}

TEST_F(TransposeTilingTest, int8) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 8192;
    compilerInfo.dType ="int8";
    compilerInfo.fp16Times = 1;

    shapeInfo.inShape.push_back(60);
    shapeInfo.inShape.push_back(70);
    shapeInfo.outShape.push_back(70);
    shapeInfo.outShape.push_back(60);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(0);
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    EXPECT_TRUE(TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo));
}

TEST_F(TransposeTilingTest, specific_shape_scenario_4) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 8192;
    compilerInfo.ubSizeCouldUse = 8064;
    compilerInfo.dType ="float16";
    compilerInfo.fp16Times = 1;

    shapeInfo.inShape.push_back(1000);
    shapeInfo.inShape.push_back(5);
    shapeInfo.inShape.push_back(64);
    shapeInfo.inShape.push_back(64);

    shapeInfo.outShape.push_back(1000);
    shapeInfo.outShape.push_back(64);
    shapeInfo.outShape.push_back(5);
    shapeInfo.outShape.push_back(64);

    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(3);

    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    EXPECT_TRUE(TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo));
    EXPECT_EQ(shapeInfo.scenario, 4);
}

TEST_F(TransposeTilingTest, specific_shape_scenario_5) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 8192;
    compilerInfo.ubSizeCouldUse = 8064;
    compilerInfo.dType ="float16";
    compilerInfo.fp16Times = 1;

    shapeInfo.inShape.push_back(2);
    shapeInfo.inShape.push_back(3);
    shapeInfo.inShape.push_back(4);
    shapeInfo.inShape.push_back(2);
    shapeInfo.inShape.push_back(3);
    shapeInfo.inShape.push_back(4);

    shapeInfo.outShape.push_back(4);
    shapeInfo.outShape.push_back(3);
    shapeInfo.outShape.push_back(2);
    shapeInfo.outShape.push_back(4);
    shapeInfo.outShape.push_back(3);
    shapeInfo.outShape.push_back(2);

    shapeInfo.perm.push_back(5);
    shapeInfo.perm.push_back(4);
    shapeInfo.perm.push_back(3);
    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(0);

    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    EXPECT_TRUE(TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo));
    EXPECT_EQ(shapeInfo.scenario, 5);
}

TEST_F(TransposeTilingTest, scenario_9_src_mode) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 8192;
    compilerInfo.ubSizeCouldUse = 8064;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 1;

    shapeInfo.inShape.push_back(1477);
    shapeInfo.inShape.push_back(1477);
    shapeInfo.inShape.push_back(8);
    shapeInfo.inShape.push_back(64);

    shapeInfo.outShape.push_back(1477);
    shapeInfo.outShape.push_back(8);
    shapeInfo.outShape.push_back(1477);
    shapeInfo.outShape.push_back(64);

    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(3);

    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    EXPECT_TRUE(TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo));
    EXPECT_EQ(shapeInfo.scenario, 9);
    EXPECT_EQ(shapeInfo.mteMode, MTE_MODE_SRC);
}

TEST_F(TransposeTilingTest, scenario_9_dst_mode) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 8192;
    compilerInfo.ubSizeCouldUse = 8064;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 1;

    shapeInfo.inShape.push_back(1477);
    shapeInfo.inShape.push_back(8);
    shapeInfo.inShape.push_back(1447);
    shapeInfo.inShape.push_back(64);

    shapeInfo.outShape.push_back(1477);
    shapeInfo.outShape.push_back(1447);
    shapeInfo.outShape.push_back(8);
    shapeInfo.outShape.push_back(64);

    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(3);

    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    EXPECT_TRUE(TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo));
    EXPECT_EQ(shapeInfo.scenario, 9);
    EXPECT_EQ(shapeInfo.mteMode, MTE_MODE_DST);
}

TEST_F(TransposeTilingTest, specific_shape_scenario_9) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 8192;
    compilerInfo.ubSizeCouldUse = 8064;
    compilerInfo.dType ="float16";
    compilerInfo.fp16Times = 1;

    shapeInfo.inShape.push_back(100);
    shapeInfo.inShape.push_back(200);
    shapeInfo.inShape.push_back(128);

    shapeInfo.outShape.push_back(200);
    shapeInfo.outShape.push_back(100);
    shapeInfo.outShape.push_back(128);

    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(2);

    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    EXPECT_TRUE(TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo));
    EXPECT_EQ(shapeInfo.scenario, 9);
}

TEST_F(TransposeTilingTest, fp16_310) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 2;
    compilerInfo.ubSize = 8192;
    compilerInfo.dType ="float16";
    compilerInfo.fp16Times = 1;

    shapeInfo.inShape.push_back(3);
    shapeInfo.inShape.push_back(256000);
    shapeInfo.inShape.push_back(8);
    shapeInfo.outShape.push_back(8);
    shapeInfo.outShape.push_back(256000);
    shapeInfo.outShape.push_back(3);
    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(0);
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);
}

/*
 *
   scenario  in                  out                 perm        reducedIn           reducedOut          reducedPerm  dim  lastAxisLen  lastAxisBurstLen  alignElement
   -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
   7         6,7,7,1             7,6,7,1             2,0,1,3     42,7                7,42                1,0          2    7            1                 1
   
   coreNum    usedCoreNum    ubSize    ubSizeCouldUse
   --------------------------------------------------
   32         0              8192      0
   
   n                   col                 row                 nFactor  colFactor  rowFactor  priority  modelName
   --------------------------------------------------------------------------------------------------------------------------------
                       1                   0                   32       1          1          8         Model008_t2f
                       1                   0                   1        1          1          999       Model003
                       1                   0                   1        1          1          999       Model007_f2t
                       1                   0                   1        1          1          999       Model006
                       1                   0                   1        1          1          999       Model004_f2t
                       1                   0                   1        1          1          999       Model005_t2f
                       1                   0                   1        1          1          999       Model001
                       1                   0                   1        1          1          999       Model002
   
   
   nJumpAxisNum  srcJumpAxisNum  dstJumpAxisNum  nJumpFactor         nJumpStride         srcJumpFactor       srcJumpStride       dstJumpFactor       dstJumpStride       rPartVol
   ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
   0             1               1                                                       42                  7                   7                   42                  294
   
   loopN  nOffsetActual  initNTuple          colPerMC  loopMC  colTC  colOffset  bsl  initDstTuple        tailDstTuple        rowPerMR  loopMR  rowTR  rowOffset  bsu  initSrcTuple        tailSrcTuple
   ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
   1      0                                  7         0       0      0          0    0,0,0,0,0,0,0,0     0,0,0,0,0,0,0,0     40        1       8      0          6    0                   34
   0      0                                  7         0       0      0          0    0,0,0,0,0,0,0,0     0,0,0,0,0,0,0,0     40        1       8      0          6    0                   34
   0      0                                  7         0       0      0          0    0,0,0,0,0,0,0,0     0,0,0,0,0,0,0,0     40        1       8      0          6    0                   34
   0      0                                  7         0       0      0          0    0,0,0,0,0,0,0,0     0,0,0,0,0,0,0,0     40        1       8      0          6    0                   34
   0      0                                  7         0       0      0          0    0,0,0,0,0,0,0,0     0,0,0,0,0,0,0,0     40        1       8      0          6    0                   34
   0      0                                  7         0       0      0          0    0,0,0,0,0,0,0,0     0,0,0,0,0,0,0,0     40        1       8      0          6    0                   34
   0      0                                  7         0       0      0          0    0,0,0,0,0,0,0,0     0,0,0,0,0,0,0,0     40        1       8      0          6    0                   34
   ...
*/
TEST_F(TransposeTilingTest, split_n_with_small_shape_1) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 8192;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(6);
    shapeInfo.inShape.push_back(7);
    shapeInfo.inShape.push_back(7);
    shapeInfo.inShape.push_back(1);
    shapeInfo.outShape.push_back(7);
    shapeInfo.outShape.push_back(6);
    shapeInfo.outShape.push_back(7);
    shapeInfo.outShape.push_back(1);
    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(3);
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);
    EXPECT_EQ(shapeInfo.scenario, 5);
}


/*
  scenario  in                  out                 perm        reducedIn           reducedOut          reducedPerm  dim  lastAxisLen  lastAxisBurstLen  alignElement
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  6         1,2,2,1             2,1,2,1             2,0,1,3     2,2,1               2,2,1               1,0,2        3    1            1                 7
  
  coreNum    usedCoreNum    ubSize    ubSizeCouldUse
  --------------------------------------------------
  32         1              8192      0
  
  backNum  skipEle  srcStrideLogic  srcJumpStride                 dstJumpStride                 dstJumpFactor                 dstJumpFactorMod
  -------------------------------------------------------------------------------------------------------------------------------------------------------
  8        0        1               2,1                           1,2                           2,2                           1,2
  
  base        num        initTuple                     headMajorLoop  headMajorNum  headTailNum  bodyLoopNum  bodymajorLoop  bodyMajorNum  bodyTailNum  tailMajorLoop  tailMajorNum  tailTailNum
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  0           4          0,0                           0              0             0            0            0              0             0            0              0             0
  0           0          0,0                           0              0             0            0            0              0             0            0              0             0
  0           0          0,0                           0              0             0            0            0              0             0            0              0             0
  0           0          0,0                           0              0             0            0            0              0             0            0              0             0
  0           0          0,0                           0              0             0            0            0              0             0            0              0             0
  0           0          0,0                           0              0             0            0            0              0             0            0              0             0
  0           0          0,0                           0              0             0            0            0              0             0            0              0             0
 */
 
TEST_F(TransposeTilingTest, split_n_with_small_shape_2) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.ubSize = 8192;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(1);
    shapeInfo.inShape.push_back(2);
    shapeInfo.inShape.push_back(2);
    shapeInfo.inShape.push_back(1);
    shapeInfo.outShape.push_back(2);
    shapeInfo.outShape.push_back(1);
    shapeInfo.outShape.push_back(2);
    shapeInfo.outShape.push_back(1);
    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(3);
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);
    EXPECT_EQ(runtimeInfo.infoPerCoreLastAxisNT[0].base, 0);
    EXPECT_EQ(runtimeInfo.infoPerCoreLastAxisNT[0].num, 4);
    EXPECT_EQ(runtimeInfo.infoPerCoreLastAxisNT[1].base, 0);
    EXPECT_EQ(runtimeInfo.infoPerCoreLastAxisNT[1].num, 0);
}


TEST_F(TransposeTilingTest, last_axis_join_transpose_check_jump_stride) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.dType ="float32";
    compilerInfo.fp16Times = 2;

    shapeInfo.inShape.push_back(3);
    shapeInfo.inShape.push_back(40);
    shapeInfo.inShape.push_back(5);
    shapeInfo.inShape.push_back(6);
    shapeInfo.inShape.push_back(7);
    shapeInfo.inShape.push_back(8);
    shapeInfo.inShape.push_back(9);

    shapeInfo.outShape.push_back(3);
    shapeInfo.outShape.push_back(7);
    shapeInfo.outShape.push_back(6);
    shapeInfo.outShape.push_back(9);
    shapeInfo.outShape.push_back(8);
    shapeInfo.outShape.push_back(5);
    shapeInfo.outShape.push_back(40);

    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(4);
    shapeInfo.perm.push_back(3);
    shapeInfo.perm.push_back(6);
    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(5);
    shapeInfo.perm.push_back(1);
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);

    EXPECT_EQ(runtimeInfo.nJumpAxisNum, 1);
    EXPECT_EQ(runtimeInfo.srcJumpAxisNum, 1);
    EXPECT_EQ(runtimeInfo.dstJumpAxisNum, 5);

    EXPECT_EQ(runtimeInfo.nJumpFactor[0], 3);

    EXPECT_EQ(runtimeInfo.nJumpStrideIn[0], 604800);

    EXPECT_EQ(runtimeInfo.srcJumpStride[0], 15120);

    EXPECT_EQ(runtimeInfo.dstJumpStride[0], 1600);

    EXPECT_EQ(runtimeInfo.infoPerCore[1].infoN.loopOnN, 1);
}

TEST_F(TransposeTilingTest, para_test_32) {
    using namespace optiling;
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("Transpose");
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

    TeOpParas opParas;
    vector<int64_t> inShape;
    vector<int64_t> outShape;
    inShape.push_back(3);
    inShape.push_back(4);
    outShape.push_back(4);
    outShape.push_back(3);

    TeOpTensorArg tensorInputs;
    TeOpTensor tensorInput;
    tensorInput.shape = inShape;
    tensorInput.dtype = "float16";
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);

    TeOpTensorArg tensorOutputsArg;
    TeOpTensor tensorOutput;
    tensorOutput.shape = outShape;
    tensorOutput.dtype = "float16";
    tensorOutputsArg.tensor.push_back(tensorOutput);
    tensorOutputsArg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputsArg);

    std::vector<int64_t> perm_shape;
    perm_shape.push_back(2);
    ge::Shape ge_shape(perm_shape);
    ge::Tensor const_tensor(ge::TensorDesc(ge_shape, ge::Format::FORMAT_ND, ge::DataType::DT_INT32));
    int32_t buf[2];
    buf[0] = 1;
    buf[1] = 0;
    opParas.const_inputs["perm"] = std::make_tuple((const unsigned char *)buf, sizeof(buf), const_tensor);

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "123456a";

    opParas.op_type = "Transpose";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(TransposeTilingTest, para_test_64) {
    using namespace optiling;
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("Transpose");
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

    TeOpParas opParas;
    vector<int64_t> inShape;
    vector<int64_t> outShape;
    inShape.push_back(3);
    inShape.push_back(4);
    outShape.push_back(4);
    outShape.push_back(3);

    TeOpTensorArg tensorInputs;
    TeOpTensor tensorInput;
    tensorInput.shape = inShape ;
    tensorInput.dtype = "float16";
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);

    TeOpTensorArg tensorOutputsArg;
    TeOpTensor tensorOutput;
    tensorOutput.shape = outShape;
    tensorOutput.dtype = "float16";
    tensorOutputsArg.tensor.push_back(tensorOutput);
    tensorOutputsArg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputsArg);

    std::vector<int64_t> perm_shape;
    perm_shape.push_back(2);
    ge::Shape ge_shape(perm_shape);
    ge::Tensor const_tensor(ge::TensorDesc(ge_shape, ge::Format::FORMAT_ND, ge::DataType::DT_INT64));
    int64_t buf[2];
    buf[0] = 1;
    buf[1] = 0;
    opParas.const_inputs["perm"] = std::make_tuple((const unsigned char *)buf, sizeof(buf), const_tensor);

    std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "123456b";

    opParas.op_type = "Transpose";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}


TEST_F(TransposeTilingTest, 920A_vcopy) {
    using namespace optiling;
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("Transpose");
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

    TeOpParas opParas;
    vector<int64_t> inShape;
    vector<int64_t> outShape;
    inShape.push_back(100);
    inShape.push_back(200);
    inShape.push_back(16);
    outShape.push_back(200);
    outShape.push_back(100);
    outShape.push_back(16);

    TeOpTensorArg tensorInputs;
    TeOpTensor tensorInput;
    tensorInput.shape = inShape ;
    tensorInput.dtype = "float16";
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);

    TeOpTensorArg tensorOutputsArg;
    TeOpTensor tensorOutput;
    tensorOutput.shape = outShape;
    tensorOutput.dtype = "float16";
    tensorOutputsArg.tensor.push_back(tensorOutput);
    tensorOutputsArg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputsArg);

    std::vector<int64_t> perm_shape;
    perm_shape.push_back(3);
    ge::Shape ge_shape(perm_shape);
    ge::Tensor const_tensor(ge::TensorDesc(ge_shape, ge::Format::FORMAT_ND, ge::DataType::DT_INT64));
    int64_t buf[3];
    buf[0] = 1;
    buf[1] = 0;
    buf[2] = 2;
    opParas.const_inputs["perm"] = std::make_tuple((const unsigned char *)buf, sizeof(buf), const_tensor);

    std::string compileInfo = "{\"vars\": {\"core_num\":96, \"ub_size\":6144, \"dtype\":\"float16\"}}";
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "123456c";

    opParas.op_type = "Transpose";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));

}

TEST_F(TransposeTilingTest, scenario_4_seri) {
    using namespace optiling;
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("Transpose");
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

    TeOpParas opParas;
    vector<int64_t> inShape;
    vector<int64_t> outShape;
    inShape.push_back(5);
    inShape.push_back(6);
    inShape.push_back(7);
    inShape.push_back(32);
    outShape.push_back(7);
    outShape.push_back(6);
    outShape.push_back(5);
    outShape.push_back(32);

    TeOpTensorArg tensorInputs;
    TeOpTensor tensorInput;
    tensorInput.shape = inShape ;
    tensorInput.dtype = "float16";
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);

    TeOpTensorArg tensorOutputsArg;
    TeOpTensor tensorOutput;
    tensorOutput.shape = outShape;
    tensorOutput.dtype = "float16";
    tensorOutputsArg.tensor.push_back(tensorOutput);
    tensorOutputsArg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputsArg);

    std::vector<int64_t> perm_shape;
    perm_shape.push_back(4);
    ge::Shape ge_shape(perm_shape);
    ge::Tensor const_tensor(ge::TensorDesc(ge_shape, ge::Format::FORMAT_ND, ge::DataType::DT_INT64));
    int64_t buf[4];
    buf[0] = 2;
    buf[1] = 1;
    buf[2] = 0;
    buf[3] = 3;
    opParas.const_inputs["perm"] = std::make_tuple((const unsigned char *)buf, sizeof(buf), const_tensor);

    std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "123456d";

    opParas.op_type = "Transpose";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}


TEST_F(TransposeTilingTest, scenario_9_seri) {
    using namespace optiling;
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("Transpose");
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

    TeOpParas opParas;
    vector<int64_t> inShape;
    vector<int64_t> outShape;
    inShape.push_back(100);
    inShape.push_back(200);
    inShape.push_back(128);
    outShape.push_back(200);
    outShape.push_back(100);
    outShape.push_back(128);

    TeOpTensorArg tensorInputs;
    TeOpTensor tensorInput;
    tensorInput.shape = inShape ;
    tensorInput.dtype = "float16";
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);

    TeOpTensorArg tensorOutputsArg;
    TeOpTensor tensorOutput;
    tensorOutput.shape = outShape;
    tensorOutput.dtype = "float16";
    tensorOutputsArg.tensor.push_back(tensorOutput);
    tensorOutputsArg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputsArg);

    std::vector<int64_t> perm_shape;
    perm_shape.push_back(3);
    ge::Shape ge_shape(perm_shape);
    ge::Tensor const_tensor(ge::TensorDesc(ge_shape, ge::Format::FORMAT_ND, ge::DataType::DT_INT64));
    int64_t buf[3];
    buf[0] = 1;
    buf[1] = 0;
    buf[2] = 2;
    opParas.const_inputs["perm"] = std::make_tuple((const unsigned char *)buf, sizeof(buf), const_tensor);

    std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "123456e";

    opParas.op_type = "Transpose";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

