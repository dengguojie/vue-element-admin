/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "./common_layer.h"

#include <memory>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

#include "runtime/rt.h"
#include "runtime/mem.h"

using namespace std;

const int BUFFER_ALIGN_SIZE = 32;

std::vector<std::string> SplitString(const std::string &in, const char &delim)
{
    vector<string> result;
    string::size_type i = 0;
    string::size_type j = i;
    while (i < in.size()) {
        if (in[i] == delim) {
            result.push_back(in.substr(j, i - j));
            j = i + 1;
            i++;
        } else {
            i++;
        }
    }
    if (j < in.size()) {
        result.push_back(in.substr(j, in.size() - j));
    }
    return result;
}

uint64_t Str2Int(const std::string &str)
{
    uint64_t num;
    std::istringstream iss(str);
    iss >> num;
    return num;
}

/**
 * get line's part, between ':' and ",", or between ':' and end. line should be like xxx: 123
 * @param line string line
 * @return line's part
 */
string GetValue(const string &line)
{
    auto startPos = line.find(':') + 1;
    auto endPos = line.find(',');
    auto lenValue = endPos == std::string::npos ? line.size() - startPos : endPos - startPos;
    return line.substr(startPos, lenValue);
}


bool ParseOpWorkspaceTypes(uint32_t numWorkspace, std::ifstream &ifs, OpJsonInfo &opJsonInfo)
{
    std::string line;
    std::string value;
    for (uint32_t i = 0; i < numWorkspace; i++) {
        if (std::getline(ifs, line)) {
            line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
            value = line.substr(0, line.find(','));
            uint64_t size = Str2Int(value);
            if (size == 0) {
                OP_ST_LOG(OP_ST_LOG_ERROR, "json error, get zero workspace size!");
                return false;
            }
            opJsonInfo.workspaceTypes.push_back(size);

            std::getline(ifs, line);
            std::getline(ifs, line);
            if (line.find("type") == std::string::npos) {
                OP_ST_LOG(OP_ST_LOG_ERROR, "json error, cannot find workspace number!");
                return false;
            }
            std::getline(ifs, line);
            line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
            value = line.substr(0, line.find(','));
            uint64_t type = Str2Int(value);
            if (type == 0) {
                OP_ST_LOG(OP_ST_LOG_ERROR, "json error, get zero workspace type!");
                return false;
            }
            opJsonInfo.workspaceTypes.push_back(type);
        } else {
            OP_ST_LOG(OP_ST_LOG_ERROR, "json error, not enough workspace is defined!");
            return false;
        }
    }
    return true;
}

bool ParseOpWorkspace(std::ifstream &ifs, OpJsonInfo &opJsonInfo)
{
    std::string line;
    std::string value;
    std::getline(ifs, line);
    uint64_t numWorkspace = Str2Int(GetValue(line));
    std::getline(ifs, line);
    if (line.find("size") == std::string::npos) {
        OP_ST_LOG(OP_ST_LOG_ERROR, "json error, cannot find workspace number!");
        return false;
    }

    if (numWorkspace == 1 && "[" != GetValue(line)) {
        uint64_t size = Str2Int(GetValue(line));
        if (size == 0) {
            OP_ST_LOG(OP_ST_LOG_ERROR, "json error, get zero workspace size!");
            return false;
        }
        opJsonInfo.workspaceSizes.push_back(size);
        return true;
    }

    return ParseOpWorkspaceTypes(numWorkspace, ifs, opJsonInfo);
}

bool ParseOpJsonInfo(string jsonFilePath, OpJsonInfo &opJsonInfo)
{
    OP_ST_LOG(OP_ST_LOG_INFO, "Start parse op json file: %s", jsonFilePath.c_str());
    std::ifstream ifs(jsonFilePath.c_str());
    if (!ifs) {
        OP_ST_LOG(OP_ST_LOG_ERROR, "json file open failed: %s", jsonFilePath.c_str());
        return false;
    }

    std::string line;
    std::string::size_type pos;
    while (std::getline(ifs, line)) {
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        pos = line.find("\":");
        if (pos == std::string::npos) {
            continue;
        }
        std::string key = line.substr(1, pos - 1);
        std::string value;
        if (key == "blockDim") {
            value = GetValue(line);
            OP_ST_LOG(OP_ST_LOG_INFO, "blockDim str: %s", value.c_str());
            opJsonInfo.blockDim = Str2Int(value);
        } else if (key == "workspace") {
            if (ParseOpWorkspace(ifs, opJsonInfo)) {
                return false;
            }
            break;
        }
    }
    return true;
}

bool ReadBinFile(const string &filePath, uint64_t &fileSize, string &contents)
{
    OP_ST_LOG(OP_ST_LOG_INFO, "ReadBinFile file:%s start!", filePath.c_str());

    stringstream contentSs;
    ifstream fileStream(filePath, std::ios::binary);
    if (!fileStream.is_open()) {
        OP_ST_LOG(OP_ST_LOG_ERROR, "Can not open file: %s", filePath.c_str());
        return false;
    }
    contentSs << fileStream.rdbuf();
    contents = contentSs.str();
    fileSize = contents.size();

    return true;
}

CommonLayer::CommonLayer() : inputCnt(0), outputCnt(0), opJsonInfo()
{
}

CommonLayer::~CommonLayer()
{
}

vector<uint64_t> ParserSizeList(const std::string &in)
{
    OP_ST_LOG(OP_ST_LOG_INFO, "ParserStrToLong: %s", in.c_str());
    std::vector<std::string> inputSizeStrList = SplitString(in, ';');
    std::vector<uint64_t> res;
    for (auto sizeStr: inputSizeStrList) {
        uint64_t result = Str2Int(sizeStr);
        OP_ST_LOG(OP_ST_LOG_INFO, "ParserStrToLong value: %lu", result);
        res.push_back(result);
    }
    return res;
}

CommonLayer::CommonLayer(const OpParams &opParams)
{
    inputCnt = opParams.inputCnt;
    inputSizes = ParserSizeList(opParams.inputSizes);
    outputSizes = ParserSizeList(opParams.outputSizes);
    outputCnt = opParams.outputCnt;
    inputDataFilePaths = SplitString(opParams.inputDataPaths, ';');
    outputDataFilePaths = SplitString(opParams.outputDataPaths, ';');
    binFilePath = opParams.binFilePath;
    kernelFuncName = opParams.kernelFuncName;
    ParseOpJsonInfo(opParams.jsonFilePath, opJsonInfo);
}

bool CommonLayer::RegisterBinaryKernel(const string &filePath, const string &kernelFuncKey,
                                       const string &kernelFuncName)
{
    uint64_t bufferSize = 0;
    if (!ReadBinFile(filePath, bufferSize, content)) {
        OP_ST_LOG(OP_ST_LOG_ERROR, "Read op kernel file failed: %s failed!", filePath.c_str());
        return false;
    }
    rtDevBinary_t binary;
    void *binHandle = nullptr;
    binary.data = content.c_str();
    binary.length = bufferSize;
    binary.magic = RT_DEV_BINARY_MAGIC_ELF;
    binary.version = 0;
    rtError_t rtRet = rtDevBinaryRegister(&binary, &binHandle);
    if (rtRet != RT_ERROR_NONE) {
        OP_ST_LOG(OP_ST_LOG_ERROR, "rtDevBinaryRegister: %s failed!", kernelFuncName.c_str());
        return false;
    }

    rtRet = rtFunctionRegister(binHandle, kernelFuncKey.c_str(), kernelFuncName.c_str(), kernelFuncName.c_str(), 0);
    if (rtRet != RT_ERROR_NONE) {
        OP_ST_LOG(OP_ST_LOG_ERROR, "rtFunctionRegister: %s failed!", kernelFuncName.c_str());
        return false;
    }
    return true;
}

uint64_t CommonLayer::AlignSize(uint64_t size)
{
    return (size + BUFFER_ALIGN_SIZE - 1) / BUFFER_ALIGN_SIZE * BUFFER_ALIGN_SIZE + BUFFER_ALIGN_SIZE;
}

CommonLayer::RtResourceCleanHelper::RtResourceCleanHelper() : needCleanResources(), deviceIds()
{
}


void CommonLayer::RtResourceCleanHelper::AddHBMResource(void *resource)
{
    needCleanResources.push_back(resource);
}

bool CommonLayer::RtResourceCleanHelper::SetDevice(int32_t deviceId)
{
    rtError_t rtRet;
    rtRet = rtSetDevice(deviceId);
    if (rtRet != RT_ERROR_NONE) {
        OP_ST_LOG(OP_ST_LOG_ERROR, "rtSetDevice Error");
        return false;
    }
    deviceIds.push_back(deviceId);
    return true;
}

CommonLayer::RtResourceCleanHelper::~RtResourceCleanHelper()
{
    OP_ST_LOG(OP_ST_LOG_INFO, "release rt resources.")
    for (auto resource:needCleanResources) {
        rtFree(resource);
        resource = nullptr;
    }
    rtError_t rtRet;
    for (auto deviceId: deviceIds) {
        rtRet = rtDeviceReset(deviceId);
        if (rtRet != RT_ERROR_NONE) {
            OP_ST_LOG(OP_ST_LOG_ERROR, "rtDeviceReset Error");
        }
    }
}


bool CommonLayer::MallocAndCpyInputToDevice(vector<void *> &inputAddrs, RtResourceCleanHelper &resourceCleanHelper)
{
    rtError_t error;
    bool mallocSuccess = true;
    int inputIdx = 0;
    OP_ST_LOG(OP_ST_LOG_INFO, "rtMalloc and rtMemcpy input hbm start");
    for (auto inputSize: inputSizes) {
        void *hbmInputAddr = nullptr;
        OP_ST_LOG(OP_ST_LOG_INFO, "rtMalloc input %d hbm start, input size: %lu, input malloc size: %lu", inputIdx,
                  inputSize, AlignSize(inputSize));
        error = rtMalloc(&hbmInputAddr, AlignSize(inputSize), RT_MEMORY_HBM);
        if (error != RT_ERROR_NONE) {
            OP_ST_LOG(OP_ST_LOG_ERROR, "rtMalloc failed, malloc size: %lu", inputSize);
            mallocSuccess = false;
            break;
        }
        inputAddrs.push_back(hbmInputAddr);
        resourceCleanHelper.AddHBMResource(hbmInputAddr);

        OP_ST_LOG(OP_ST_LOG_INFO, "malloc input %d hbm success", inputIdx);
        uint64_t inputFileSize = 0;
        OP_ST_LOG(OP_ST_LOG_INFO, "read input %d bin data start", inputIdx);
        string content;
        if (!ReadBinFile(inputDataFilePaths[inputIdx].c_str(), inputFileSize, content)) {
            OP_ST_LOG(OP_ST_LOG_ERROR, "read input bin data failed, input data file: %s",
                      inputDataFilePaths[inputIdx].c_str());
            return false;
        }
        OP_ST_LOG(OP_ST_LOG_INFO, "read input %d bin data end, input data file size: %lu", inputIdx, inputFileSize);
        OP_ST_LOG(OP_ST_LOG_INFO, "rtMemcpy input %d hbm start, input size: %lu, input data file size: %lu", inputIdx,
                  inputSize, inputFileSize);
        error = rtMemcpy(hbmInputAddr, AlignSize(inputSize), content.c_str(), inputFileSize, RT_MEMCPY_HOST_TO_DEVICE);
        if (error != RT_ERROR_NONE) {
            OP_ST_LOG(OP_ST_LOG_ERROR, "rtMemcpy input hbm failed, input_size: %lu", inputSize);
            mallocSuccess = false;
            break;
        }
        OP_ST_LOG(OP_ST_LOG_INFO, "rtMemcpy input %d hbm success", inputIdx);

        inputIdx++;
    }
    if (!mallocSuccess) {
        OP_ST_LOG(OP_ST_LOG_ERROR, "malloc and memcpy input hbm failed.");
        return false;
    }
    OP_ST_LOG(OP_ST_LOG_INFO, "malloc and memcpy input hbm success.");
    return true;
}

bool CommonLayer::MallocOutputBufferInDevice(vector<void *> &outputAddrs, RtResourceCleanHelper &resourceCleanHelper)
{
    rtError_t error;
    bool mallocSuccess = true;
    OP_ST_LOG(OP_ST_LOG_INFO, "malloc output hbm start.");
    int outputIdx = 0;
    for (auto outputSize: outputSizes) {
        void *hbmOutputAddr = nullptr;
        error = rtMalloc(&hbmOutputAddr, AlignSize(outputSize), RT_MEMORY_HBM);
        if (error != RT_ERROR_NONE) {
            OP_ST_LOG(OP_ST_LOG_ERROR, "rtMalloc failed, malloc size: %lu", AlignSize(outputSize));
            mallocSuccess = false;
            break;
        }
        OP_ST_LOG(OP_ST_LOG_INFO, "malloc output %d hbm success, size: %lu", outputIdx, AlignSize(outputSize));
        outputAddrs.push_back(hbmOutputAddr);
        resourceCleanHelper.AddHBMResource(hbmOutputAddr);
        outputIdx++;
    }
    if (!mallocSuccess) {
        OP_ST_LOG(OP_ST_LOG_ERROR, "malloc output hbm failed.");
        return false;
    }
    OP_ST_LOG(OP_ST_LOG_INFO, "malloc output hbm success.");
    return true;
}

bool CommonLayer::MallocWorkspaceBufferInDevice(vector<void *> &workspaceAddrs,
                                                RtResourceCleanHelper &resourceCleanHelper)
{
    rtError_t error;
    bool mallocSuccess = true;
    OP_ST_LOG(OP_ST_LOG_INFO, "malloc workspace hbm start.");
    int workspaceIdx = 0;
    for (auto workspaceSize: opJsonInfo.workspaceSizes) {
        void *workspaceAddr = nullptr;
        error = rtMalloc(&workspaceAddr, AlignSize(workspaceSize), RT_MEMORY_HBM);
        if (error != RT_ERROR_NONE) {
            OP_ST_LOG(OP_ST_LOG_ERROR, "rtMalloc failed, malloc size: %lu", AlignSize(workspaceSize));
            mallocSuccess = false;
            break;
        }
        OP_ST_LOG(OP_ST_LOG_INFO, "malloc workspace %d hbm success, size: %lu", workspaceIdx, AlignSize(workspaceSize));
        workspaceAddrs.push_back(workspaceAddr);
        resourceCleanHelper.AddHBMResource(workspaceAddr);
    }
    if (!mallocSuccess) {
        OP_ST_LOG(OP_ST_LOG_ERROR, "malloc workspace hbm failed.");
        return false;
    }
    OP_ST_LOG(OP_ST_LOG_INFO, "malloc workspace hbm success.");
    return true;
}

bool CommonLayer::PrepareKernelArgs(vector<void *> &inputAddrs, vector<void *> &outputAddrs,
                                    vector<void *> &workspaceAddrs, RtResourceCleanHelper &resourceCleanHelper)
{
    if (!MallocAndCpyInputToDevice(inputAddrs, resourceCleanHelper)) {
        return false;
    }

    if (!MallocOutputBufferInDevice(outputAddrs, resourceCleanHelper)) {
        return false;
    }

    if (!MallocWorkspaceBufferInDevice(workspaceAddrs, resourceCleanHelper)) {
        return false;
    }
    return true;
}

bool CommonLayer::DumpOutputBuffer(vector<void *> &outputAddrs)
{
    rtError_t error;
    for (int i = 0; i < outputCnt; i++) {
        unique_ptr<char[]> outputData(new(std::nothrow) char[outputSizes[i]]);
        if (outputData == nullptr) {
            OP_ST_LOG(OP_ST_LOG_ERROR, "new unique_ptr for outputData failed!");
            return false;
        }
        error = rtMemcpy(outputData.get(), outputSizes[i], outputAddrs[i], outputSizes[i], RT_MEMCPY_DEVICE_TO_HOST);
        if (error != RT_ERROR_NONE) {
            OP_ST_LOG(OP_ST_LOG_ERROR, "rtMemcpy output hbm failed!");
            return false;
        }
        ofstream outputDataFile(outputDataFilePaths[i], ios::out | ios::binary);
        outputDataFile.write(outputData.get(), outputSizes[i]);
        outputDataFile.close();
    }
    return true;
}


bool CommonLayer::RunOnDeviceAndDumpOutput(RtResourceCleanHelper &resourceCleanHelper)
{
    rtError_t error;
    vector<void *> inputAddrs;
    vector<void *> outputAddrs;
    vector<void *> workspaceAddrs;
    if (!PrepareKernelArgs(inputAddrs, outputAddrs, workspaceAddrs, resourceCleanHelper)) {
        return false;
    }
    int argSize = inputAddrs.size() + outputAddrs.size() + workspaceAddrs.size();
    int curIdx = 0;
    void *args[argSize];
    for (auto inputAddr:inputAddrs) {
        args[curIdx++] = inputAddr;
    }
    for (auto outputAddr:outputAddrs) {
        args[curIdx++] = outputAddr;
    }
    for (auto workspaceAddr:workspaceAddrs) {
        args[curIdx++] = workspaceAddr;
    }
    rtStream_t stream;
    error = rtStreamCreate(&stream, 0);
    if (error != RT_ERROR_NONE) {
        OP_ST_LOG(OP_ST_LOG_ERROR, "rtStreamCreate failed!");
        return false;
    }
    error = rtKernelLaunch(kernelFuncName.c_str(), opJsonInfo.blockDim, args, argSize * sizeof(void *), NULL, stream);
    if (error != RT_ERROR_NONE) {
        OP_ST_LOG(OP_ST_LOG_ERROR, "rtKernelLaunch failed!");
        return false;
    }
    error = rtStreamSynchronize(stream);
    if (error != RT_ERROR_NONE) {
        OP_ST_LOG(OP_ST_LOG_ERROR, "rtStreamSynchronize failed!");
        return false;
    }
    if (!DumpOutputBuffer(outputAddrs)) {
        return false;
    }
    error = rtStreamDestroy(stream);
    if (error != RT_ERROR_NONE) {
        return false;
    }
    return true;
}

bool CommonLayer::Run()
{
    RtResourceCleanHelper resourceCleanHelper;
    int32_t device = 0;
    if (!resourceCleanHelper.SetDevice(device)) {
        return false;
    }

    OP_ST_LOG(OP_ST_LOG_INFO, "RegisterBinaryKernel start");
    bool result = RegisterBinaryKernel(binFilePath, kernelFuncName, kernelFuncName);
    if (!result) {
        OP_ST_LOG(OP_ST_LOG_ERROR, "RegisterBinaryKernel failed");
        return false;
    }
    OP_ST_LOG(OP_ST_LOG_INFO, "RegisterBinaryKernel success");

    return RunOnDeviceAndDumpOutput(resourceCleanHelper);
}