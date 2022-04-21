#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "array_ops.h"
#define private public
#include "op_tiling/conv2d_fast_tiling.h"

using namespace optiling;

const uint32_t VALID = 0;
const uint32_t INVALID = 1;
const uint32_t NOTSUPPORTNOW = 2;


class CheckTiling {
    /**
     * @brief check tiling result, should satisfy nine path.
     * 
     * judgment path diagram
     * 
     * brief mark
     * fm: feature map
     * weight: w
     * full load: fd
     * L1 is mad's n: 1T
     * L1 is mad's 1/n: nT
     * 
     * fm.fd & w.!fd --> not support now
     * fm.fd & w.fd  --> not supoort now
     * 
     * fm.!fd & w.fd
     *     |----w.fd2l1
     *     |        |----fm 1T (load2d/3d)--> valid
     *     |        |----fm nT (load2d/3d)--> valid
     *     |
     *     |----w.fd2l0b
     *              |----fm 1T (load2d/3d)--> valid
     *              |----fm nT --> not support now
     * 
     * fm.!fd & w.!fd
     *     |----w.reuse --> not supported now
     *     |----fm.reuse
     *              |----fm nT
     *                     |----w(w->L0B) (laod2d) --> valid
     *                     |----w(w->L1->L0B) --> not support
     * 
     *              |----fm 1T
     *                     |----w(w->L0B) (load3d) --> valid
     *                     |----w(w->L1->L0B)
     *                                |----w 1T (load2d) --> valid
     *                                |----w nT --> not support now
     */
    public:
        explicit CheckTiling(Conv2dTiling& tilingOri, Conv2dParams& inputParamsOri) {
            tiling = tilingOri;
            inputParams = inputParamsOri;
            const int64_t ci1 = (inputParams.fmci + 16 - 1) / 16;
            const int64_t ci0 = 16;
            const int64_t kernelCi1 = (inputParams.wci + 16 - 1) / 16;
            const int64_t kernelCi0 = 16;
            // calculate KA
            KA = ((inputParams.kh - 1) * inputParams.dilations_h + 1) *
                 ((inputParams.kw - 1) * inputParams.dilations_w + 1) *
                 ci1 * ci0;
            // calculate KB
            KB = inputParams.kh * inputParams.kw * kernelCi0 * kernelCi1;
            // isLoad2dFlag should also check aType and bType, resnet50 both DT_FLOAT16
            isLoad2dFlag = (inputParams.kh == 1 && inputParams.kw == 1) &&
                (inputParams.padl == 0 && inputParams.padr == 0 && inputParams.padu == 0 && inputParams.padd == 0) &&
                (inputParams.stride_h == 1 && inputParams.stride_w == 1) &&
                inputParams.hi != 1;
        };
        ~CheckTiling() {};

        uint32_t IsValidTiling() {
            if (IsWeightFullLoad() && !IsFMFullLoad()) {
                if (IsWeightFullLoadL1()) {
                    // weight full loaded to L1
                    if (IsKAOneTimeskAL1()) {
                        // load2d and load3d are both support, do not judge
                        return VALID;
                    } else if (IsKAMultiTimeskAL1()){
                        // load2d and load3d are both support, do not judge
                        return VALID;
                    }
                    printf("Tiling strategy is unknwon when weight is fully loadded to L1!\n");
                    return INVALID;
                } else if (IsWeightFullLoadL0()) {
                    // weight full loaded to L0
                    if (IsKAOneTimeskAL1()) {
                        // load2d and load3d are both support, do not judge
                        return VALID;
                    } else if (IsKAMultiTimeskAL1()){
                        printf("Tiling strategy is currently not supported "
                            "when weight is fully loadded to L1 and K_A != kAL1!\n");
                        return NOTSUPPORTNOW;
                    }
                    printf("Tiling strategy is unknwon when weight is fully loadded to L0!\n");
                    return INVALID;
                }
            } else if (!IsWeightFullLoad() && !IsFMFullLoad()) {
                if (IsFMReused()) {
                    // FM reused
                    if (IsKAMultiTimeskAL1()) {
                        if (IsWeightDirectIntoL0B()) {
                            return VALID;
                        }
                        printf("Tiling strategy is currently not supported "
                            "when FM is reused but weight is loaded to L1!\n");
                        return INVALID;
                    } else if (IsKAOneTimeskAL1()) {
                        if (IsWeightDirectIntoL0B()) {
                            return VALID;
                        } else if (!IsWeightDirectIntoL0B()) {
                            if (IsKBOneTimeskBL1()) {
                                return VALID;
                            }
                        }
                    }
                    printf("Tiling strategy is unknwon when FM is reused!\n");
                    return INVALID;
                } else if (IsWeightReused()) {
                    // weight reused
                    printf("Tiling strategy is currently not supported "
                            "when weight is reused!\n");
                    return NOTSUPPORTNOW;
                }
            } else {
                printf("Tiling strategy is currently not supported!\n");
                return NOTSUPPORTNOW;
            }
        };

        void ShowTiling() {
            std::cout<<"tiling batchDim is "<<tiling.batchDim<<std::endl;
            std::cout<<"tiling nDim is "<<tiling.nDim<<std::endl;
            std::cout<<"tiling mDim is "<<tiling.mDim<<std::endl;
            std::cout<<"tiling groupDim is "<<tiling.groupDim<<std::endl;
            std::cout<<"tiling kAL1 is "<<tiling.kAl1<<std::endl;
            std::cout<<"tiling mAL1 is "<<tiling.mAl1<<std::endl;
            std::cout<<"tiling kBL1 is "<<tiling.kBl1<<std::endl;
            std::cout<<"tiling nBL1 is "<<tiling.nBl1<<std::endl;
            std::cout<<"tiling mA is "<<tiling.ma<<std::endl;
            std::cout<<"tiling kA is "<<tiling.ka<<std::endl;
            std::cout<<"tiling kB is "<<tiling.kb<<std::endl;
            std::cout<<"tiling nB is "<<tiling.nb<<std::endl;
            std::cout<<"tiling mC is "<<tiling.mc<<std::endl;
            std::cout<<"tiling nC is "<<tiling.nc<<std::endl;
            std::cout<<"tiling nCFactor is "<<tiling.ncFactor<<std::endl;
            std::cout<<"tiling mcFactor is "<<tiling.mcFactor<<std::endl;
            std::cout<<"tiling kAub is "<<tiling.kAub<<std::endl;
            std::cout<<"tiling mAub is "<<tiling.mAub<<std::endl;
        }

    private:
        Conv2dTiling tiling;
        Conv2dParams inputParams;
        uint32_t KA;
        uint32_t KB;
        bool isLoad2dFlag;

        bool IsFMFullLoad() {
            return tiling.mAl1 == FULL_LOAD && tiling.kAl1 == FULL_LOAD;
        }

        bool IsWeightFullLoad() {
            // Is if weight is full-load
            return IsWeightFullLoadL1() || IsWeightFullLoadL0();
        }

        bool IsWeightFullLoadL1() {
            return tiling.nBl1 == FULL_LOAD && tiling.nb != FULL_LOAD;
        }

        bool IsWeightFullLoadL0() {
            return tiling.nBl1 == 0 && tiling.nb == FULL_LOAD;
        }

        // feature map L1 is mad's N(N is one)
        bool IsKAOneTimeskAL1() {
            return KA == tiling.kAl1;
        }

        // feature map L1 is mad's 1 / N
        bool IsKAMultiTimeskAL1() {
            return (KA / tiling.kAl1 > 1) && (KA % tiling.kAl1 == 0);
        }

        // weight L1 is mad's N(N is one)
        bool IsKBOneTimeskBL1() {
            return KB == tiling.kBl1;
        }

        bool IsFMReused() {
            return tiling.kAl1 >= tiling.kBl1;
        }

        bool IsWeightReused() {
            return tiling.kAl1 < tiling.kBl1;
        }

        bool IsWeightDirectIntoL0B() {
            // weight is not full load and do not pass l1
            return tiling.kBl1 == 0;
        }
};


class Conv2DFastTilingResNetTest920A : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "Conv2DFastTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "Conv2DFastTilingTest TearDown" << std::endl;
    }

    void setHardwareInfoAscend920A() {
        hardwareInfo.aicoreNum = 48;
        hardwareInfo.l2Size = 33554432;
        hardwareInfo.l1Size = 524288;
        hardwareInfo.l0aSize = 65536;
        hardwareInfo.l0bSize = 65536;
        hardwareInfo.l0cSize = 131072;
        hardwareInfo.ubSize = 196608;
        hardwareInfo.btSize = 1024;
        hardwareInfo.ddrReadRate = 32;
        hardwareInfo.ddrWriteRate = 32;
        hardwareInfo.l2Rate = 110;
        hardwareInfo.l2ReadRate = 110;
        hardwareInfo.l2WriteRate = 86;
        hardwareInfo.l1ToL0aRate = 512;
        hardwareInfo.l1ToL0bRate = 256;
        hardwareInfo.l1ToUbRate = 128;
        hardwareInfo.l0cToUbRate = 256;
        hardwareInfo.ubToL2Rate = 64;
        hardwareInfo.ubToDdrRate = 64;
        hardwareInfo.ubToL1Rate = 128;
        hardwareInfo.socVersion = "Ascend920A";
    }

    virtual void SetUp() {
        setHardwareInfoAscend920A();
        std::cout << "SetUp" << std::endl;
    }

    virtual void TearDown() {
         std::cout << "TearDown" << std::endl;

    }
    Conv2dParams inputParams;
    HardwareInfo hardwareInfo;
    Conv2dTiling tiling;
    int64_t targetTime = std::chrono::microseconds(10).count();
};

void get_resnet_tiling_case_00(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 64;
    inputParams.hi = 56;
    inputParams.wi = 56;
    // weight
    inputParams.n = 256;
    inputParams.wci = 64;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 56;
    inputParams.wo = 56;
    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;
    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}

void get_resnet_tiling_case_01(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 256;
    inputParams.hi = 14;
    inputParams.wi = 14;
    // weight
    inputParams.n = 1024;
    inputParams.wci = 256;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 14;
    inputParams.wo = 14;
    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;
    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}

void get_resnet_tiling_case_02(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 128;
    inputParams.hi = 28;
    inputParams.wi = 28;
    // weight
    inputParams.n = 512;
    inputParams.wci = 128;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 28;
    inputParams.wo = 28;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}

void get_resnet_tiling_case_03(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 256;
    inputParams.hi = 56;
    inputParams.wi = 56;
    // weight
    inputParams.n = 512;
    inputParams.wci = 256;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 28;
    inputParams.wo = 28;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 2;
    inputParams.stride_w = 2;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 4;
}

void get_resnet_tiling_case_04(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 4;
    inputParams.hi = 224;
    inputParams.wi = 224;
    // weight
    inputParams.n = 64;
    inputParams.wci = 4;
    inputParams.kh = 7;
    inputParams.kw = 7;
    // out
    inputParams.ho = 112;
    inputParams.wo = 112;

    inputParams.padu = 2;
    inputParams.padd = 3;
    inputParams.padl = 2;
    inputParams.padr = 3;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 2;
    inputParams.stride_w = 2;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 3;
}

void get_resnet_tiling_case_05(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 512;
    inputParams.hi = 7;
    inputParams.wi = 7;
    // weight
    inputParams.n = 2048;
    inputParams.wci = 512;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 7;
    inputParams.wo = 7;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}

void get_resnet_tiling_case_06(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 512;
    inputParams.hi = 28;
    inputParams.wi = 28;
    // weight
    inputParams.n = 1024;
    inputParams.wci = 512;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 14;
    inputParams.wo = 14;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 2;
    inputParams.stride_w = 2;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 4;
}

void get_resnet_tiling_case_07(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 1024;
    inputParams.hi = 14;
    inputParams.wi = 14;
    // weight
    inputParams.n = 2048;
    inputParams.wci = 1024;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 7;
    inputParams.wo = 7;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 2;
    inputParams.stride_w = 2;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 4;
}

void get_resnet_tiling_case_08(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 64;
    inputParams.hi = 56;
    inputParams.wi = 56;
    // weight
    inputParams.n = 64;
    inputParams.wci = 64;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 56;
    inputParams.wo = 56;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 4;
}

void get_resnet_tiling_case_09(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 64;
    inputParams.hi = 56;
    inputParams.wi = 56;
    // weight
    inputParams.n = 256;
    inputParams.wci = 64;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 56;
    inputParams.wo = 56;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 4;
}

void get_resnet_tiling_case_10(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 64;
    inputParams.hi = 56;
    inputParams.wi = 56;
    // weight
    inputParams.n = 64;
    inputParams.wci = 64;
    inputParams.kh = 3;
    inputParams.kw = 3;
    // out
    inputParams.ho = 56;
    inputParams.wo = 56;

    inputParams.padu = 1;
    inputParams.padd = 1;
    inputParams.padl = 1;
    inputParams.padr = 1;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 4;
}

void get_resnet_tiling_case_11(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 256;
    inputParams.hi = 56;
    inputParams.wi = 56;
    // weight
    inputParams.n = 64;
    inputParams.wci = 256;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 56;
    inputParams.wo = 56;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 4;
}

void get_resnet_tiling_case_12(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 64;
    inputParams.hi = 56;
    inputParams.wi = 56;
    // weight
    inputParams.n = 256;
    inputParams.wci = 64;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 56;
    inputParams.wo = 56;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 4;
}

void get_resnet_tiling_case_13(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 256;
    inputParams.hi = 56;
    inputParams.wi = 56;
    // weight
    inputParams.n = 128;
    inputParams.wci = 256;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 56;
    inputParams.wo = 56;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 4;
}

void get_resnet_tiling_case_14(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 128;
    inputParams.hi = 56;
    inputParams.wi = 56;
    // weight
    inputParams.n = 128;
    inputParams.wci = 128;
    inputParams.kh = 3;
    inputParams.kw = 3;
    // out
    inputParams.ho = 28;
    inputParams.wo = 28;

    inputParams.padu = 0;
    inputParams.padd = 1;
    inputParams.padl = 0;
    inputParams.padr = 1;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 2;
    inputParams.stride_w = 2;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 4;
}

void get_resnet_tiling_case_15(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 512;
    inputParams.hi = 28;
    inputParams.wi = 28;
    // weight
    inputParams.n = 128;
    inputParams.wci = 512;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 28;
    inputParams.wo = 28;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 4;
}

void get_resnet_tiling_case_16(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 128;
    inputParams.hi = 28;
    inputParams.wi = 28;
    // weight
    inputParams.n = 128;
    inputParams.wci = 128;
    inputParams.kh = 3;
    inputParams.kw = 3;
    // out
    inputParams.ho = 28;
    inputParams.wo = 28;

    inputParams.padu = 1;
    inputParams.padd = 1;
    inputParams.padl = 1;
    inputParams.padr = 1;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 4;
}

void get_resnet_tiling_case_17(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 512;
    inputParams.hi = 28;
    inputParams.wi = 28;
    // weight
    inputParams.n = 256;
    inputParams.wci = 512;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 28;
    inputParams.wo = 28;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 4;
}

void get_resnet_tiling_case_18(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 128;
    inputParams.hi = 28;
    inputParams.wi = 28;
    // weight
    inputParams.n = 512;
    inputParams.wci = 128;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 28;
    inputParams.wo = 28;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 4;
}

void get_resnet_tiling_case_19(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 256;
    inputParams.hi = 28;
    inputParams.wi = 28;
    // weight
    inputParams.n = 256;
    inputParams.wci = 256;
    inputParams.kh = 3;
    inputParams.kw = 3;
    // out
    inputParams.ho = 14;
    inputParams.wo = 14;

    inputParams.padu = 0;
    inputParams.padd = 1;
    inputParams.padl = 0;
    inputParams.padr = 1;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 2;
    inputParams.stride_w = 2;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 4;
}

void get_resnet_tiling_case_20(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 256;
    inputParams.hi = 14;
    inputParams.wi = 14;
    // weight
    inputParams.n = 256;
    inputParams.wci = 256;
    inputParams.kh = 3;
    inputParams.kw = 3;
    // out
    inputParams.ho = 14;
    inputParams.wo = 14;

    inputParams.padu = 1;
    inputParams.padd = 1;
    inputParams.padl = 1;
    inputParams.padr = 1;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 4;
}

void get_resnet_tiling_case_21(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 1024;
    inputParams.hi = 14;
    inputParams.wi = 14;
    // weight
    inputParams.n = 256;
    inputParams.wci = 1024;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 14;
    inputParams.wo = 14;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 4;
}

void get_resnet_tiling_case_22(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 256;
    inputParams.hi = 14;
    inputParams.wi = 14;
    // weight
    inputParams.n = 1024;
    inputParams.wci = 256;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 14;
    inputParams.wo = 14;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 4;
}

void get_resnet_tiling_case_23(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 512;
    inputParams.hi = 14;
    inputParams.wi = 14;
    // weight
    inputParams.n = 512;
    inputParams.wci = 512;
    inputParams.kh = 3;
    inputParams.kw = 3;
    // out
    inputParams.ho = 7;
    inputParams.wo = 7;

    inputParams.padu = 0;
    inputParams.padd = 1;
    inputParams.padl = 0;
    inputParams.padr = 1;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 2;
    inputParams.stride_w = 2;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 4;
}

void get_resnet_tiling_case_24(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 1024;
    inputParams.hi = 14;
    inputParams.wi = 14;
    // weight
    inputParams.n = 512;
    inputParams.wci = 1024;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 14;
    inputParams.wo = 14;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 4;
}

void get_resnet_tiling_case_25(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 2048;
    inputParams.hi = 7;
    inputParams.wi = 7;
    // weight
    inputParams.n = 512;
    inputParams.wci = 2048;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 7;
    inputParams.wo = 7;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 4;
}

void get_resnet_tiling_case_26(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 512;
    inputParams.hi = 7;
    inputParams.wi = 7;
    // weight
    inputParams.n = 512;
    inputParams.wci = 512;
    inputParams.kh = 3;
    inputParams.kw = 3;
    // out
    inputParams.ho = 7;
    inputParams.wo = 7;

    inputParams.padu = 1;
    inputParams.padd = 1;
    inputParams.padl = 1;
    inputParams.padr = 1;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 4;
}

void get_resnet_tiling_case_27(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 64 * 16;
    inputParams.hi = 14;
    inputParams.wi = 14;
    // weight
    inputParams.n = 256;
    inputParams.wci = 64 * 16;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 14;
    inputParams.wo = 14;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}

void get_resnet_tiling_case_28(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 256;
    inputParams.hi = 56;
    inputParams.wi = 56;
    // weight
    inputParams.n = 512;
    inputParams.wci = 256;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 28;
    inputParams.wo = 28;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 2;
    inputParams.stride_w = 2;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 4;
}

void get_resnet_tiling_case_29(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 128 * 16;
    inputParams.hi = 7;
    inputParams.wi = 7;
    // weight
    inputParams.n = 512;
    inputParams.wci = 128 * 16;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 7;
    inputParams.wo = 7;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}

void get_resnet_tiling_case_30(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 32 * 16;
    inputParams.hi = 14;
    inputParams.wi = 14;
    // weight
    inputParams.n = 1024;
    inputParams.wci = 32 * 16;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 14;
    inputParams.wo = 14;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}

void get_resnet_tiling_case_31(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 8 * 16;
    inputParams.hi = 28;
    inputParams.wi = 28;
    // weight
    inputParams.n = 128;
    inputParams.wci = 8 * 16;
    inputParams.kh = 3;
    inputParams.kw = 3;
    // out
    inputParams.ho = 28;
    inputParams.wo = 28;

    inputParams.padu = 1;
    inputParams.padd = 1;
    inputParams.padl = 1;
    inputParams.padr = 1;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}

void get_resnet_tiling_case_32(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 8 * 16;
    inputParams.hi = 14;
    inputParams.wi = 14;
    // weight
    inputParams.n = 512;
    inputParams.wci = 8 * 16;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 14;
    inputParams.wo = 14;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}

void get_resnet_tiling_case_33(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 64;
    inputParams.hi = 28;
    inputParams.wi = 28;
    // weight
    inputParams.n = 256;
    inputParams.wci = 64;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 28;
    inputParams.wo = 28;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}

void get_resnet_tiling_case_34(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 16;
    inputParams.hi = 224;
    inputParams.wi = 224;
    // weight
    inputParams.n = 64;
    inputParams.wci = 16;
    inputParams.kh = 7;
    inputParams.kw = 7;
    // out
    inputParams.ho = 112;
    inputParams.wo = 112;

    inputParams.padu = 3;
    inputParams.padd = 3;
    inputParams.padl = 3;
    inputParams.padr = 3;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 2;
    inputParams.stride_w = 2;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}
void get_resnet_tiling_case_35(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 64;
    inputParams.hi = 56;
    inputParams.wi = 56;
    // weight
    inputParams.n = 64;
    inputParams.wci = 64;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 56;
    inputParams.wo = 56;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}
void get_resnet_tiling_case_36(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 16 * 16;
    inputParams.hi = 7;
    inputParams.wi = 7;
    // weight
    inputParams.n = 1024;
    inputParams.wci = 16 * 16;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 7;
    inputParams.wo = 7;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}

void get_resnet_tiling_case_37(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 16 * 16;
    inputParams.hi = 14;
    inputParams.wi = 14;
    // weight
    inputParams.n = 256;
    inputParams.wci = 16 * 16;
    inputParams.kh = 3;
    inputParams.kw = 3;
    // out
    inputParams.ho = 14;
    inputParams.wo = 14;

    inputParams.padu = 1;
    inputParams.padd = 1;
    inputParams.padl = 1;
    inputParams.padr = 1;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}
void get_resnet_tiling_case_38(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 32 * 16;
    inputParams.hi = 28;
    inputParams.wi = 28;
    // weight
    inputParams.n = 128;
    inputParams.wci = 32 * 16;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 28;
    inputParams.wo = 28;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}
void get_resnet_tiling_case_39(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 64 * 16;
    inputParams.hi = 7;
    inputParams.wi = 7;
    // weight
    inputParams.n = 512;
    inputParams.wci = 64 * 16;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 7;
    inputParams.wo = 7;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}

void get_resnet_tiling_case_40(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 64 * 16;
    inputParams.hi = 7;
    inputParams.wi = 7;
    // weight
    inputParams.n = 2048;
    inputParams.wci = 64 * 16;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 7;
    inputParams.wo = 7;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}

void get_resnet_tiling_case_41(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 64;
    inputParams.hi = 56;
    inputParams.wi = 56;
    // weight
    inputParams.n = 64;
    inputParams.wci = 64;
    inputParams.kh = 3;
    inputParams.kw = 3;
    // out
    inputParams.ho = 56;
    inputParams.wo = 56;

    inputParams.padu = 1;
    inputParams.padd = 1;
    inputParams.padl = 1;
    inputParams.padr = 1;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}

void get_resnet_tiling_case_42(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 16 * 16;
    inputParams.hi = 28;
    inputParams.wi = 28;
    // weight
    inputParams.n = 512;
    inputParams.wci = 16 * 16;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 28;
    inputParams.wo = 28;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}

void get_resnet_tiling_case_43(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 16 * 16;
    inputParams.hi = 56;
    inputParams.wi = 56;
    // weight
    inputParams.n = 64;
    inputParams.wci = 16 * 16;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 56;
    inputParams.wo = 56;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}

void get_resnet_tiling_case_44(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 32 * 16;
    inputParams.hi = 14;
    inputParams.wi = 14;
    // weight
    inputParams.n = 256;
    inputParams.wci = 32 * 16;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 14;
    inputParams.wo = 14;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}
void get_resnet_tiling_case_45(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 64;
    inputParams.hi = 56;
    inputParams.wi = 56;
    // weight
    inputParams.n = 64;
    inputParams.wci = 64;
    inputParams.kh = 3;
    inputParams.kw = 3;
    // out
    inputParams.ho = 28;
    inputParams.wo = 28;

    inputParams.padu = 1;
    inputParams.padd = 1;
    inputParams.padl = 1;
    inputParams.padr = 1;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 2;
    inputParams.stride_w = 2;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}
void get_resnet_tiling_case_46(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 32 * 16;
    inputParams.hi = 7;
    inputParams.wi = 7;
    // weight
    inputParams.n = 512;
    inputParams.wci = 32 * 16;
    inputParams.kh = 3;
    inputParams.kw = 3;
    // out
    inputParams.ho = 7;
    inputParams.wo = 7;

    inputParams.padu = 1;
    inputParams.padd = 1;
    inputParams.padl = 1;
    inputParams.padr = 1;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}

void get_resnet_tiling_case_47(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 16 * 16;
    inputParams.hi = 14;
    inputParams.wi = 14;
    // weight
    inputParams.n = 256;
    inputParams.wci = 16 * 16;
    inputParams.kh = 3;
    inputParams.kw = 3;
    // out
    inputParams.ho = 7;
    inputParams.wo = 7;

    inputParams.padu = 1;
    inputParams.padd = 1;
    inputParams.padl = 1;
    inputParams.padr = 1;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 2;
    inputParams.stride_w = 2;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}
void get_resnet_tiling_case_48(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 16 * 16;
    inputParams.hi = 28;
    inputParams.wi = 28;
    // weight
    inputParams.n = 128;
    inputParams.wci = 16 * 16;
    inputParams.kh = 1;
    inputParams.kw = 1;
    // out
    inputParams.ho = 28;
    inputParams.wo = 28;

    inputParams.padu = 0;
    inputParams.padd = 0;
    inputParams.padl = 0;
    inputParams.padr = 0;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 1;
    inputParams.stride_w = 1;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}

void get_resnet_tiling_case_49(Conv2dParams& inputParams) {
    // feature map
    inputParams.batch = 1;
    inputParams.fmci = 8 * 16;
    inputParams.hi = 28;
    inputParams.wi = 28;
    // weight
    inputParams.n = 128;
    inputParams.wci = 8 * 16;
    inputParams.kh = 3;
    inputParams.kw = 3;
    // out
    inputParams.ho = 14;
    inputParams.wo = 14;

    inputParams.padu = 1;
    inputParams.padd = 1;
    inputParams.padl = 1;
    inputParams.padr = 1;
    inputParams.dilations_h = 1;
    inputParams.dilations_w = 1;
    inputParams.stride_h = 2;
    inputParams.stride_w = 2;
    inputParams.groups = 1;
    inputParams.biasFlag = true;

    inputParams.preFusionUbUtilize = 1;
    inputParams.postFusionUbUtilize = 1;
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_00) {
    get_resnet_tiling_case_00(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_01) {
    get_resnet_tiling_case_01(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_02) {
    get_resnet_tiling_case_02(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_03) {
    get_resnet_tiling_case_03(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_04) {
    get_resnet_tiling_case_04(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_05) {
    get_resnet_tiling_case_05(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_06) {
    get_resnet_tiling_case_06(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_07) {
    get_resnet_tiling_case_07(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_08) {
    get_resnet_tiling_case_08(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_09) {
    get_resnet_tiling_case_09(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_10) {
    get_resnet_tiling_case_10(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_11) {
    get_resnet_tiling_case_11(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_12) {
    get_resnet_tiling_case_12(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_13) {
    get_resnet_tiling_case_13(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_14) {
    get_resnet_tiling_case_14(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_15) {
    get_resnet_tiling_case_15(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_16) {
    get_resnet_tiling_case_16(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_17) {
    get_resnet_tiling_case_17(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_18) {
    get_resnet_tiling_case_18(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_19) {
    get_resnet_tiling_case_19(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_20) {
    get_resnet_tiling_case_20(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_21) {
    get_resnet_tiling_case_21(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_22) {
    get_resnet_tiling_case_22(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_23) {
    get_resnet_tiling_case_23(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_24) {
    get_resnet_tiling_case_24(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_25) {
    get_resnet_tiling_case_25(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_26) {
    get_resnet_tiling_case_26(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_27) {
    get_resnet_tiling_case_27(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_28) {
    get_resnet_tiling_case_28(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_29) {
    get_resnet_tiling_case_29(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_30) {
    get_resnet_tiling_case_30(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_31) {
    get_resnet_tiling_case_31(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_32) {
    get_resnet_tiling_case_32(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_33) {
    get_resnet_tiling_case_33(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_34) {
    get_resnet_tiling_case_34(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_35) {
    get_resnet_tiling_case_35(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_36) {
    get_resnet_tiling_case_36(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_37) {
    get_resnet_tiling_case_37(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_38) {
    get_resnet_tiling_case_38(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_39) {
    get_resnet_tiling_case_39(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_40) {
    get_resnet_tiling_case_40(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_41) {
    get_resnet_tiling_case_41(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_42) {
    get_resnet_tiling_case_42(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_43) {
    get_resnet_tiling_case_43(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_44) {
    get_resnet_tiling_case_44(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_45) {
    get_resnet_tiling_case_45(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_46) {
    get_resnet_tiling_case_46(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_47) {
    get_resnet_tiling_case_47(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_48) {
    get_resnet_tiling_case_48(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest920A, test_get_tiling_case_49) {
    get_resnet_tiling_case_49(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

class Conv2DFastTilingResNetTest310 : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "Conv2DFastTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "Conv2DFastTilingTest TearDown" << std::endl;
    }

    void setHardwareInfoAscend310() {
        hardwareInfo.aicoreNum = 2;
        hardwareInfo.l2Size = 8388608;
        hardwareInfo.l1Size = 1048576;
        hardwareInfo.l0aSize = 65536;
        hardwareInfo.l0bSize = 65536;
        hardwareInfo.l0cSize = 262144;
        hardwareInfo.ubSize = 253952;
        hardwareInfo.btSize = 0;
        hardwareInfo.ddrReadRate = 67;
        hardwareInfo.ddrWriteRate = 64;
        hardwareInfo.l2Rate = 128;
        hardwareInfo.l2ReadRate = 128;
        hardwareInfo.l2WriteRate = 64;
        hardwareInfo.l1ToL0aRate = 512;
        hardwareInfo.l1ToL0bRate = 256;
        hardwareInfo.l1ToUbRate = 128;
        hardwareInfo.l0cToUbRate = 256;
        hardwareInfo.ubToL2Rate = 64;
        hardwareInfo.ubToDdrRate = 64;
        hardwareInfo.ubToL1Rate = 128;
        hardwareInfo.socVersion = "Ascend310";
    }

    virtual void SetUp() {
        setHardwareInfoAscend310();
        std::cout << "SetUp" << std::endl;
    }

    virtual void TearDown() {
         std::cout << "TearDown" << std::endl;

    }
    Conv2dParams inputParams;
    HardwareInfo hardwareInfo;
    Conv2dTiling tiling;
    int64_t targetTime = std::chrono::microseconds(10).count();
};


TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_00) {
    get_resnet_tiling_case_00(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_01) {
    get_resnet_tiling_case_01(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_02) {
    get_resnet_tiling_case_02(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_03) {
    get_resnet_tiling_case_03(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_04) {
    get_resnet_tiling_case_04(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_05) {
    get_resnet_tiling_case_05(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_06) {
    get_resnet_tiling_case_06(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_07) {
    get_resnet_tiling_case_07(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_08) {
    get_resnet_tiling_case_08(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_09) {
    get_resnet_tiling_case_09(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_10) {
    get_resnet_tiling_case_10(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_11) {
    get_resnet_tiling_case_11(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_12) {
    get_resnet_tiling_case_12(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_13) {
    get_resnet_tiling_case_13(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_14) {
    get_resnet_tiling_case_14(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_15) {
    get_resnet_tiling_case_15(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_16) {
    get_resnet_tiling_case_16(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_17) {
    get_resnet_tiling_case_17(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_18) {
    get_resnet_tiling_case_18(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_19) {
    get_resnet_tiling_case_19(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_20) {
    get_resnet_tiling_case_20(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_21) {
    get_resnet_tiling_case_21(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_22) {
    get_resnet_tiling_case_22(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_23) {
    get_resnet_tiling_case_23(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_24) {
    get_resnet_tiling_case_24(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_25) {
    get_resnet_tiling_case_25(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_26) {
    get_resnet_tiling_case_26(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_27) {
    get_resnet_tiling_case_27(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_28) {
    get_resnet_tiling_case_28(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_29) {
    get_resnet_tiling_case_29(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_30) {
    get_resnet_tiling_case_30(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_31) {
    get_resnet_tiling_case_31(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_32) {
    get_resnet_tiling_case_32(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_33) {
    get_resnet_tiling_case_33(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_34) {
    get_resnet_tiling_case_34(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_35) {
    get_resnet_tiling_case_35(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_36) {
    get_resnet_tiling_case_36(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_37) {
    get_resnet_tiling_case_37(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_38) {
    get_resnet_tiling_case_38(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_39) {
    get_resnet_tiling_case_39(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_40) {
    get_resnet_tiling_case_40(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_41) {
    get_resnet_tiling_case_41(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_42) {
    get_resnet_tiling_case_42(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_43) {
    get_resnet_tiling_case_43(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_44) {
    get_resnet_tiling_case_44(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_45) {
    get_resnet_tiling_case_45(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_46) {
    get_resnet_tiling_case_46(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_47) {
    get_resnet_tiling_case_47(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_48) {
    get_resnet_tiling_case_48(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest310, test_get_tiling_case_49) {
    get_resnet_tiling_case_49(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

class Conv2DFastTilingResNetTest710 : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "Conv2DFastTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "Conv2DFastTilingTest TearDown" << std::endl;
    }

    void setHardwareInfoAscend710() {
        hardwareInfo.aicoreNum = 8;
        hardwareInfo.l2Size = 16777216;
        hardwareInfo.l1Size = 1048576;
        hardwareInfo.l0aSize = 65536;
        hardwareInfo.l0bSize = 65536;
        hardwareInfo.l0cSize = 262144;
        hardwareInfo.ubSize = 262144;
        hardwareInfo.btSize = 0;
        hardwareInfo.ddrReadRate = 17;
        hardwareInfo.ddrWriteRate = 17;
        hardwareInfo.l2Rate = 114;
        hardwareInfo.l2ReadRate = 114;
        hardwareInfo.l2WriteRate = 114;
        hardwareInfo.l1ToL0aRate = 512;
        hardwareInfo.l1ToL0bRate = 256;
        hardwareInfo.l1ToUbRate = 128;
        hardwareInfo.l0cToUbRate = 512;
        hardwareInfo.ubToL2Rate = 114;
        hardwareInfo.ubToDdrRate = 17;
        hardwareInfo.ubToL1Rate = 256;
        hardwareInfo.socVersion = "Ascend710";
    }

    virtual void SetUp() {
        setHardwareInfoAscend710();
        std::cout << "SetUp" << std::endl;
    }

    virtual void TearDown() {
         std::cout << "TearDown" << std::endl;

    }
    Conv2dParams inputParams;
    HardwareInfo hardwareInfo;
    Conv2dTiling tiling;
    int64_t targetTime = std::chrono::microseconds(10).count();
};

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_00) {
    get_resnet_tiling_case_00(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_01) {
    get_resnet_tiling_case_01(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_02) {
    get_resnet_tiling_case_02(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_03) {
    get_resnet_tiling_case_03(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_04) {
    get_resnet_tiling_case_04(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_05) {
    get_resnet_tiling_case_05(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_06) {
    get_resnet_tiling_case_06(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_07) {
    get_resnet_tiling_case_07(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_08) {
    get_resnet_tiling_case_08(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_09) {
    get_resnet_tiling_case_09(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_10) {
    get_resnet_tiling_case_10(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_11) {
    get_resnet_tiling_case_11(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_12) {
    get_resnet_tiling_case_12(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_13) {
    get_resnet_tiling_case_13(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_14) {
    get_resnet_tiling_case_14(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_15) {
    get_resnet_tiling_case_15(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_16) {
    get_resnet_tiling_case_16(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_17) {
    get_resnet_tiling_case_17(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_18) {
    get_resnet_tiling_case_18(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_19) {
    get_resnet_tiling_case_19(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_20) {
    get_resnet_tiling_case_20(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_21) {
    get_resnet_tiling_case_21(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_22) {
    get_resnet_tiling_case_22(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_23) {
    get_resnet_tiling_case_23(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_24) {
    get_resnet_tiling_case_24(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_25) {
    get_resnet_tiling_case_25(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_26) {
    get_resnet_tiling_case_26(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_27) {
    get_resnet_tiling_case_27(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_28) {
    get_resnet_tiling_case_28(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_29) {
    get_resnet_tiling_case_29(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_30) {
    get_resnet_tiling_case_30(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_31) {
    get_resnet_tiling_case_31(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_32) {
    get_resnet_tiling_case_32(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_33) {
    get_resnet_tiling_case_33(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_34) {
    get_resnet_tiling_case_34(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_35) {
    get_resnet_tiling_case_35(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_36) {
    get_resnet_tiling_case_36(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_37) {
    get_resnet_tiling_case_37(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_38) {
    get_resnet_tiling_case_38(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_39) {
    get_resnet_tiling_case_39(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_40) {
    get_resnet_tiling_case_40(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_41) {
    get_resnet_tiling_case_41(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_42) {
    get_resnet_tiling_case_42(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_43) {
    get_resnet_tiling_case_43(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_44) {
    get_resnet_tiling_case_44(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_45) {
    get_resnet_tiling_case_45(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_46) {
    get_resnet_tiling_case_46(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_47) {
    get_resnet_tiling_case_47(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_48) {
    get_resnet_tiling_case_48(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}

TEST_F(Conv2DFastTilingResNetTest710, test_get_tiling_case_49) {
    get_resnet_tiling_case_49(inputParams);
    auto begin = std::chrono::high_resolution_clock::now();
    bool ret = Conv2dFastTiling(inputParams, hardwareInfo, tiling);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t runTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout<<"Con2dFastTiling run time is "<<runTime<<"us"<<std::endl;
    //ASSERT_LE(runTime, targetTime);
    ASSERT_TRUE(ret);
    CheckTiling checker(tiling, inputParams);
    // checker.ShowTiling();
    uint32_t check_ret = checker.IsValidTiling();
    ASSERT_EQ(check_ret, VALID);
}