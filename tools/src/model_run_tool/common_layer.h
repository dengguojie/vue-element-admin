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

#ifndef OPP_COMMON_LAYER_H
#define OPP_COMMON_LAYER_H

#include <vector>
#include <string>
#include <stdint.h>

#define OP_ST_LOG_FATAL                   ("FATAL")
#define OP_ST_LOG_ERROR                   ("ERROR")
#define OP_ST_LOG_WARNING                 ("WARNING")
#define OP_ST_LOG_INFO                    ("INFO")
#define OP_ST_LOG_DEBUG                   ("DEBUG")

#define OP_ST_LOG(level, format, ...) \
    do {fprintf(stderr, "[%s] [%s] [%s:%d] " format "\n", \
                level, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__);}while(0);

using std::vector;
using std::string;

struct OpParams {
    int inputCnt;
    int outputCnt;
    char *inputSizes;
    char *outputSizes;
    char *inputDataPaths;
    char *outputDataPaths;
    char *binFilePath;
    char *kernelFuncName;
    char *jsonFilePath;
};

struct OpJsonInfo {
    uint32_t blockDim = 0;
    vector<uint64_t> workspaceSizes;
    vector<uint64_t> workspaceTypes;
};


class CommonLayer {
public:
    class RtResourceCleanHelper {
    public:
        RtResourceCleanHelper();

        ~RtResourceCleanHelper();

        void AddHBMResource(void *resource);

        bool SetDevice(int32_t deviceId);

    private:
        vector<void *> needCleanResources;
        vector<int32_t> deviceIds;
    };

    int inputCnt;
    int outputCnt;
    vector<uint64_t> inputSizes;
    vector<uint64_t> outputSizes;
    vector<string> inputDataFilePaths;
    vector<string> outputDataFilePaths;
    string binFilePath;
    OpJsonInfo opJsonInfo;
    string kernelFuncName;
    string content;

    CommonLayer();

    CommonLayer(const OpParams &opParams);

    ~CommonLayer();

    bool RegisterBinaryKernel(const string &filePath, const string &kernelFuncKey, const string &kernelFuncName);

    uint64_t AlignSize(uint64_t size);

    bool Run();

private:
    bool PrepareKernelArgs(vector<void *> &inputAddrs, vector<void *> &outputAddrs, vector<void *> &workspaceAddrs,
                           RtResourceCleanHelper &resourceCleanHelper);

    bool MallocAndCpyInputToDevice(vector<void *> &inputAddrs, RtResourceCleanHelper &resourceCleanHelper);

    bool MallocOutputBuffeInDevice(vector<void *> &inputAddrs, RtResourceCleanHelper &resourceCleanHelper);

    bool MallocWorkspaceBuffeInDevice(vector<void *> &workspaceAddrs, RtResourceCleanHelper &resourceCleanHelper);

    bool DumpOutputBuffer(vector<void *> &outputAddrs);

    bool RunOnDeviceAndDumpOutput(RtResourceCleanHelper &resourceCleanHelper);
};


#endif //OPP_COMMON_LAYER_H
