#include "gtest/gtest.h"
#include "sixteen_in_two_out_layer_with_workspace.hpp"
#include "six_in_two_out_layer_with_workspace.hpp"
#include <algorithm>
#include <vector>

class YOLO_V3_DETECTION_OUTPUT_V2_ST : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "YOLOV3_DETECTION_OUTPUT_V2_ST ST SetUp" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "YOLOV3_DETECTION_OUTPUT_V2_ST ST TearDown" << std::endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp() {
    }
    virtual void TearDown() {
    }
};

string BASE_DIR = "./llt/ops/common/data/yolo_v3_detection_output_v2/";
string KERNEL_DIR = "./llt/ops/common/kernel_bin/yolo_v3_detection_output_v2/";
string OP_NAME = "yolo_v3_detection_output_v2";

void run_case(vector<vector<int>>& boxInfo, uint32_t batch=1, uint32_t boxes=3,
uint32_t classes=80, const string& dataType="float16") {
    // std::string op_name = "yolo_v3_detection_output_d";
    std::string inputSizeStr = ""; //"""169_169_169";
    string dataDir = BASE_DIR + "/";
    uint32_t dataSize = dataType == "float16" ? 2 : 4;
    uint32_t align = 32 / dataSize;
    int yoloNum = boxInfo.size();
    int inputNum = yoloNum * 5 + 1;
    vector<uint32_t> inputSizeList(inputNum);
    vector<string>  inputArrPath(inputNum);
    for (int i = 0; i < yoloNum; i++) {
        auto info = boxInfo[i];
        uint32_t h1 = info[0];
        uint32_t w1 = info[1];
        dataDir += std::to_string(h1 * w1) + "_";
        if (i == yoloNum - 1) {
            inputSizeStr += std::to_string(h1 * w1);
        } else {
            inputSizeStr += std::to_string(h1 * w1) + "_";
        }
        uint32_t adjHw = int((batch * h1 * w1 + align - 1) /
        align) * align;
        uint32_t inputCoordSize = int((batch * boxes * h1 * w1 * 4 + align - 1) /
        align) * align * dataSize;
        string inputCoordFileName =  "coord_data"+to_string(i+1)+".data";
        inputArrPath[i] = inputCoordFileName;
        inputSizeList[i] = inputCoordSize;

        uint32_t inputObjSize = adjHw * boxes * dataSize;
        string inputObjFileName = "input_b" + to_string(i+1) + ".data";
        inputArrPath[yoloNum + i] = inputObjFileName;
        inputSizeList[yoloNum + i] = inputObjSize;

        uint32_t inputClassSize = inputObjSize * classes;
        string inputClassFileName = "input_c" + to_string(i+1) + ".data";
        inputArrPath[yoloNum*2 + i] = inputClassFileName;
        inputSizeList[yoloNum*2 + i] = inputClassSize;

        uint32_t windexSize = adjHw * dataSize;
        string windexFileName = "windex" + to_string(i+1) + ".data";
        inputArrPath[yoloNum*3 + 1 + i] = windexFileName;
        inputSizeList[yoloNum*3 + 1 + i] = windexSize;

        uint32_t hindexSize = adjHw * dataSize;
        string hindexFileName = "hindex" + to_string(i+1) + ".data";
        inputArrPath[yoloNum*4+1+i] = hindexFileName;
        inputSizeList[yoloNum*4+1+i] = hindexSize;
    }
    inputSizeList[yoloNum * 3] = batch * 4;
    inputArrPath[yoloNum * 3] = "img_info.data";
    dataDir += dataType;
    for (int i = 0; i < inputNum; i++) {
        inputArrPath[i] = dataDir + "/" + inputArrPath[i];
        cout << inputArrPath[i]+":"+to_string(inputSizeList[i]) << endl;
    }

    // bbox
    uint32_t bboxSize = batch*6*1024;
    // box num
    uint32_t boxNumSize = batch*1;
    std::string stubFunc =  OP_NAME + "_" + inputSizeStr + "__kernel0";
    // "yolo_v3_detection_output_d_169_169_169__kernel0";
    std::string binPath = KERNEL_DIR + OP_NAME + "_" + inputSizeStr + ".o";
    // "yolo_v3_detection_output_d_169_169_169.o";
    std::string tilingName = stubFunc;
    // "yolo_v3_detection_output_d_169_169_169__kernel0";
    std::string expectBboxPath = dataDir + "/box_out.data";
    std::string expectBoxNumPath = dataDir + "/box_out_num.data";
    cout << stubFunc << endl;
    cout << binPath << endl;
    cout << tilingName << endl;

    float ratios[2] = {0.1 ,0.1};
    if (yoloNum == 3) {
        SixteenInTwoOutLayerWithWorkspace<fp16_t,fp16_t,int32_t> layer{
            OP_NAME, inputSizeStr,
            inputSizeList[0], inputSizeList[1], inputSizeList[2],
            inputSizeList[3], inputSizeList[4], inputSizeList[5],
            inputSizeList[6], inputSizeList[7], inputSizeList[8],
            inputSizeList[9], inputSizeList[10], inputSizeList[11],
            inputSizeList[12], inputSizeList[13], inputSizeList[14],
            inputSizeList[15],
            bboxSize, boxNumSize,
            binPath, tilingName,
            inputArrPath[0], inputArrPath[1], inputArrPath[2], inputArrPath[3],
            inputArrPath[4], inputArrPath[5], inputArrPath[6], inputArrPath[7],
            inputArrPath[8], inputArrPath[9], inputArrPath[10], inputArrPath[11],
            inputArrPath[12], inputArrPath[13], inputArrPath[14],
            inputArrPath[15],
            expectBboxPath, expectBoxNumPath,
            ratios, (void*)stubFunc.c_str()
        };
        bool ret = layer.test();

        if(1) {
            layer.writeBinaryFile((void*)layer.outputData,
            dataDir + "/actual_box_out.data",
            batch * 6 * 1024 * sizeof(fp16_t));
            layer.writeBinaryFile((void*)layer.outputBData,
            dataDir + "/actual_box_out_number.data",
            batch * 1 * sizeof(int32_t));
        }
        // assert(true == ret);
    } else if (yoloNum == 1) {
        SixInTwoOutLayerWithWorkspace<fp16_t,fp16_t,int32_t> layer{
            OP_NAME, inputSizeStr,
            inputSizeList[0], inputSizeList[1], inputSizeList[2],
            inputSizeList[3], inputSizeList[4], inputSizeList[5],
            bboxSize, boxNumSize, binPath, tilingName,
            inputArrPath[0], inputArrPath[1], inputArrPath[2],
            inputArrPath[3], inputArrPath[4], inputArrPath[5],
            expectBboxPath, expectBoxNumPath, ratios, (void*)stubFunc.c_str()
        };
        bool ret = layer.test();
        if (1) {
            layer.writeBinaryFile((void*)layer.outputData,
            dataDir + "/actual_box_out.data",
            batch * 6 * 1024 * sizeof(fp16_t));
            layer.writeBinaryFile((void*)layer.outputBData,
            dataDir + "/actual_box_out_number.data",
            batch * 1 * sizeof(int32_t));
        }
        // assert(true == ret);
    }
}

TEST_F(YOLO_V3_DETECTION_OUTPUT_V2_ST, test_yolo_v3_detection_output_v2_16_16_16_float16) {
    vector<vector<int>> boxInfo = {{4,4},{4,4},{4,4}};
    run_case(boxInfo, 1, 3, 2, "float16");
}

TEST_F(YOLO_V3_DETECTION_OUTPUT_V2_ST, test_yolo_v3_detection_output_v2_16_float16) {
    vector<vector<int>> boxInfo = {{4,4}};
    run_case(boxInfo, 1, 5, 2, "float16");
}

TEST_F(YOLO_V3_DETECTION_OUTPUT_V2_ST, test_yolo_v3_detection_output_v2_169_169_169_float16) {
    vector<vector<int>> boxInfo = {{13,13},{13,13},{13,13}};
    run_case(boxInfo, 1, 3, 1, "float16");
}

TEST_F(YOLO_V3_DETECTION_OUTPUT_V2_ST, test_yolo_v3_detection_output_v2_169_float16) {
    vector<vector<int>> boxInfo = {{13,13}};
    run_case(boxInfo, 1, 5, 2, "float16");
}

TEST_F(YOLO_V3_DETECTION_OUTPUT_V2_ST, test_yolo_v3_detection_output_v2_361_float16) {
    vector<vector<int>> boxInfo = {{19,19}};
    run_case(boxInfo, 1, 5, 10, "float16");
}

TEST_F(YOLO_V3_DETECTION_OUTPUT_V2_ST, test_yolo_v3_detection_output_v2_169_676_2704_float16) {
    vector<vector<int>> boxInfo = {{13,13},{26,26},{52,52}};
    run_case(boxInfo, 1, 3, 20, "float16");
}


