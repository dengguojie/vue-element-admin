/**
* @file op_execute.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <cstdint>
#include <iostream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fstream>
#include <string>
#include <climits>

#include "acl/acl.h"
#include "op_runner.h"

#include "common.h"

bool g_isDevice = false;
bool CreateOpDesc(OpTestDesc &desc)
{
    if (desc.inputShape.size() < 0 ||
        desc.inputShape.size() != desc.inputDataType.size() ||
        desc.inputShape.size() != desc.inputFormat.size()) {
        ERROR_LOG("input opreator desc errror");
        return false;
    }

     if (desc.outputShape.size() <= 0 || 
        desc.outputShape.size() != desc.outputDataType.size() ||
        desc.outputShape.size() != desc.outputFormat.size()) {
        ERROR_LOG("output opreator desc errror");
        return false;
    }

    for (std::size_t i = 0; i < desc.inputShape.size(); i++) {
        desc.AddInputTensorDesc(desc.inputDataType[i], desc.inputShape[i].size(),
            desc.inputShape[i].data(), desc.inputFormat[i]);
    }

    for (std::size_t i = 0; i < desc.outputShape.size(); i++) {
        desc.AddOutputTensorDesc(desc.outputDataType[i], desc.outputShape[i].size(),
            desc.outputShape[i].data(), desc.outputFormat[i]);
    }

    for (auto &x : desc.opAttrVec) {
        desc.AddTensorAttr(x);
    }
    return true;
}

bool SetInputData(OpRunner &runner)
{
    for (size_t i = 0; i < runner.NumInputs(); ++i) {
        size_t fileSize;
        if (runner.GetOpTestDesc().inputFilePath[i] == "") {
            INFO_LOG("Input[%zu] is an optional input.", i);
            continue;
        }
        std::string filePath = runner.GetOpTestDesc().inputFilePath[i] + ".bin";
        bool result = ReadFile(filePath, fileSize, runner.GetInputBuffer<void>(i), runner.GetInputSize(i));
        if (!result) {
            ERROR_LOG("Read input[%zu] failed", i);
            return false;
        }
        char realPath[PATH_MAX];
        if (realpath(filePath.c_str(), realPath)) {
            INFO_LOG("Set input[%zu] from '%s' success.", i, realPath);
        }else {
            ERROR_LOG("The file '%s' is not exist.", filePath.c_str());
            return false;
        }
    }

    return true;
}

bool ProcessOutputData(OpRunner &runner)
{
    for (size_t i = 0; i < runner.NumOutputs(); ++i) {
        INFO_LOG("Output[%zu]:", i);

        std::string filePath = runner.GetOpTestDesc().outputFilePath[i] + ".bin";
        if (!WriteFile(filePath, runner.GetOutputBuffer<void>(i), runner.GetOutputSize(i))) {
            ERROR_LOG("Write output[%zu] failed.", i);
            return false;
        }
        char realPath[PATH_MAX];
        if (realpath(filePath.c_str(), realPath)) {
            INFO_LOG("Write output[%zu] success. output file = %s", i, realPath);
        }else {
            ERROR_LOG("The file '%s' is not exist.", filePath.c_str());
            return false;
        }
    }
    return true;
}

bool DoRunOp(OpTestDesc &opDesc)
{
    // create and init op desc
    if (CreateOpDesc(opDesc) == false) {
        return false;
    }
    // create Runner
    OpRunner opRunner(&opDesc);
    if (!opRunner.Init()) {
        ERROR_LOG("Init OpRunner failed");
        return false;
    }

    // Load inputs
    if (!SetInputData(opRunner)) {
        ERROR_LOG("Set input data failed");
        return false;
    }

    // Run op
    if (!opRunner.RunOp()) {
        ERROR_LOG("Run op failed");
        return false;
    }

    // process output data
    if (!ProcessOutputData(opRunner)) {
        ERROR_LOG("Process output data failed");
        return false;
    }

    INFO_LOG("Run op success");
    return true;
}

bool OpExecuteInit()
{
    static bool hasInited = false;
    if (hasInited == false) {
        hasInited = true;

        std::string output = "./result_files";
        if (access(output.c_str(), 0) == -1) {
            int ret = mkdir(output.c_str(), 0700);
            if (ret == 0) {
                INFO_LOG("make output directory successfully");
            }else {
                ERROR_LOG("make output directory fail");
                return false;
            }
        }

        std::ofstream resultFile;
        resultFile.open("./result_files/result.txt", std::ios::out);
        if (!resultFile.is_open()) {
            ERROR_LOG("prepare result file failed");
            return false;
        }
        resultFile << "Test Result:" << std::endl;
        resultFile.close();

        if (aclInit("test_data/config/acl.json") != ACL_ERROR_NONE) {
            ERROR_LOG("Init acl failed");
            return false;
        }

        if (aclopSetModelDir("op_models") != ACL_ERROR_NONE) {
            std::cerr << "Load single op model failed" << std::endl;
            return false;
        }
    }
    return true;
}

bool OpExecute(OpTestDesc &opDesc, uint32_t deviceId = 0)
{
    uint32_t deviceCount = 0;
    aclError getDeviceStatus = aclrtGetDeviceCount(&deviceCount);
    if (getDeviceStatus != ACL_SUCCESS) {
        ERROR_LOG("Get Device count failed");
        return false;
    }
    if (deviceId >= deviceCount) {
        ERROR_LOG("Device[%d] is out of range, device id maximum is [%d]", deviceId, deviceCount - 1);
        return false;
    }
    if (OpExecuteInit() == false) {
        return false;
    }
    INFO_LOG("------------------Open device[%d]------------------", deviceId);
    if (aclrtSetDevice(deviceId) != ACL_ERROR_NONE) {
        std::cerr << "Open device failed. device id = " << deviceId << std::endl;
        return false;
    }
    INFO_LOG("Open device[%d] success", deviceId);

    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_ERROR_NONE) {
        ERROR_LOG("acl get run mode failed");
        return false;
    }
    g_isDevice = (runMode == ACL_DEVICE);

    if (!DoRunOp(opDesc)) {
        (void) aclrtResetDevice(deviceId);
        return false;
    }

    return true;
}
