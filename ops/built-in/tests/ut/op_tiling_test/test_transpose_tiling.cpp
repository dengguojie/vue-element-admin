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
    compilerInfo.dType ="float";

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
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);
}

TEST_F(TransposeTilingTest, stride_gt_65535) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32; compilerInfo.dType ="float";

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
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);
}

TEST_F(TransposeTilingTest, last_axis_join_transpose_2d) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.dType ="float";

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
    compilerInfo.dType ="float";

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
    compilerInfo.dType ="float";

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
    compilerInfo.dType ="float";

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
    compilerInfo.dType ="float";

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
    EXPECT_EQ(runtimeInfo.infoPerCore[6].infoN.nOffsetActual, 30240);

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
    compilerInfo.dType ="float";

    shapeInfo.inShape.push_back(20000);
    shapeInfo.inShape.push_back(7);
    shapeInfo.outShape.push_back(7);
    shapeInfo.outShape.push_back(20000);
    shapeInfo.perm.push_back(1);
    shapeInfo.perm.push_back(0);
    
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);

    EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.rowPerMR, 280);
}

TEST_F(TransposeTilingTest, last_axis_join_transpose_f2t) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.dType ="float";

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

TEST_F(TransposeTilingTest, last_axis_join_transpose_check_jump_stride) {
    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    compilerInfo.coreNum = 32;
    compilerInfo.dType ="float";

    shapeInfo.inShape.push_back(3);
    shapeInfo.inShape.push_back(4);
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
    shapeInfo.outShape.push_back(4);

    shapeInfo.perm.push_back(0);
    shapeInfo.perm.push_back(4);
    shapeInfo.perm.push_back(3);
    shapeInfo.perm.push_back(6);
    shapeInfo.perm.push_back(2);
    shapeInfo.perm.push_back(5);
    shapeInfo.perm.push_back(1);
    
    ReduceAxis("Transpose", compilerInfo, shapeInfo);
    TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo);

    EXPECT_EQ(runtimeInfo.nJumpAxisNum, 3);
    EXPECT_EQ(runtimeInfo.srcJumpAxisNum, 3);
    EXPECT_EQ(runtimeInfo.dstJumpAxisNum, 1);

    EXPECT_EQ(runtimeInfo.nJumpFactor[0], 6);
    EXPECT_EQ(runtimeInfo.nJumpFactor[1], 7);
    EXPECT_EQ(runtimeInfo.nJumpFactor[2], 3);

    EXPECT_EQ(runtimeInfo.nJumpStride[0], 504);
    EXPECT_EQ(runtimeInfo.nJumpStride[1], 72);
    EXPECT_EQ(runtimeInfo.nJumpStride[2], 60480);

    EXPECT_EQ(runtimeInfo.srcJumpStride[0], 15120);
    EXPECT_EQ(runtimeInfo.srcJumpStride[1], 9);

    EXPECT_EQ(runtimeInfo.dstJumpStride[0], 160);

    EXPECT_EQ(runtimeInfo.infoPerCore[1].infoN.loopOnN, 4);
    EXPECT_EQ(runtimeInfo.infoPerCore[1].infoN.nOffsetActual, 5760);

    EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.colPerMC, 8);
    EXPECT_EQ(runtimeInfo.infoPerCore[1].infoCol.loopOnMC, 1);

    EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.rowPerMR, 128);
    EXPECT_EQ(runtimeInfo.infoPerCore[1].infoRow.loopOnMR, 1);
}

