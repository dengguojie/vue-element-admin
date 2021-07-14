/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef FUSION_ENGINE_INC_COMMON_FE_ERROR_CODE_H_
#define FUSION_ENGINE_INC_COMMON_FE_ERROR_CODE_H_

#include <string>

namespace fe {

/* args list for error message */
static const std::string EM_VALUE          = "value";
static const std::string EM_OPTION         = "option";
static const std::string EM_AICORE_NUM     = "ai_core_num";
static const std::string EM_NEW_OP         = "new_op";
static const std::string EM_SRC_OP         = "src_op";
static const std::string EM_SRC_FORMAT     = "src_format";
static const std::string EM_DEST_OP        = "dest_op";
static const std::string EM_DEST_FORMAT    = "dest_format";
static const std::string EM_GRAPH_NAME     = "graph_name";
static const std::string EM_FILE           = "file";
static const std::string EM_ERROR_MSG      = "errmsg";
static const std::string EM_PARAM          = "parameter";
static const std::string EM_OP_NAME        = "op_name";
static const std::string EM_OP_TYPE        = "op_type";
static const std::string EM_GRAPH_ID       = "graph_id";
static const std::string EM_THREAD_ID      = "thread_id";
static const std::string EM_TASK_ID        = "task_id";
static const std::string EM_PASS_NAME      = "pass_name";
static const std::string EM_PASS_TYPE      = "pass_type";
static const std::string EM_ATTR_OR_PATTERN_NAME = "attr_or_pattern_name";
static const std::string EM_OPS_STORE_NAME = "ops_store_name";
static const std::string EM_CORRECT_VALUE = "correct_value";
static const std::string EM_INDEX = "correct_value";
static const std::string EM_COMMON_ERROR = "common_error";
static const std::string EM_ORIGIN_DTYPE     = "origin_dtype";
/* Input parameter[--%s]'s value is empty!
 * parameter
 * */
static const std::string EM_INPUT_PARAM_EMPTY = "E10004";

static const std::string EM_INNER_ERROR = "E29999";

/* Failed to precompile op[%s, optype[%s]], when processing the graph_id[%s],
 * please check the precompiling error message.
 * opname,optype,graph_id
 * */
static const std::string EM_PRECOMPLIE_FAILED = "E20001";

/* Failed to precompile op[%s, optype[%s]], when processing the graph_id[%s]
 * in thread[%s] task[%s], please check the precompiling error message.
 * opname,optype,graph_id,thread_id,task_id
 * */
static const std::string EM_PRECOMPLIE_TASK_FAILED = "E20002";

/* Failed to compile op[%s, optype[%s]], when processing the graph_id[%s],
 * please check the compiling error message.
 * opname,optype,graph_id
 * */
static const std::string EM_COMPLIE_FAILED = "E20003";

/* Failed to compile op[%s, optype[%s]], when processing the graph_id[%s] in thread[%s] task[%s],
 * please check the compiling error message.
 * opname,optype,graph_id,thread_id,task_id
 * */
static const std::string EM_COMPLIE_TASK_FAILED = "E20004";

/* Failed to generate task, when generating the task of op[%s,
 * optype[%s]] of graph_id[%s], please check the error message.
 * opname,optype,graph_id
 * */
static const std::string EM_GENERATE_TASK_FAILED = "E20005";

/*
 * Fail to calculate tensor size of op[%s, %s].
 * opname,optype,errmsg
 * */
static const std::string EM_CAL_TENSOR_SIZE_FAILED = "E20006";

/*
 * Run graph fusion pass failed, pass name:[%s], pass type:[%s].
 * pass_name,pass_type
 * */
static const std::string EM_RUN_PASS_FAILED = "E20007";

/* Run pass failed, pass name:[%s], errmsg[%s]
 * pass_name,errmsg
 * */
static const std::string EM_RUN_GRAPH_FUSION_PASS_FAILED = "E20008";


/* The value[%s] of the input option[%s] is not supported，please check again.
 * value,option
 * */
static const std::string EM_INPUT_OPTION_INVALID = "E20101";

/* The value[%s] of the input option[ge.engine_type] is not supported，
 * only AiCore and VectorCore are supported.
 * engine_type
 * */
static const std::string EM_ENGINE_TYPE_INVALID = "E20102";

/* The value[%s] of the input option[ge.aicore_num] is out of range (0, %s].
 * value,ai_core_num
 * */
static const std::string EM_AICORENUM_OUT_OF_RANGE = "E20103";


/* Path[%s]'s realpath is empty, errmsg[%s]
 * path,errmsg
 * */
static const std::string EM_GET_REALPATH_FAILED = "W21000";

/* Open file[%s] failed. %s
 * file,errmsg
 * */
static const std::string EM_OPEN_FILE_FAILED = "E21001";

/* Read file[%s] failed, errmsg[%s]
 * file,errmsg
 * */
static const std::string EM_READ_FILE_FAILED = "E21002";

/* Node[%s] file path is empty, errmsg[%s]
 * op_name,errmsg
 * */
static const std::string EM_GET_FILEPATH_FAILED = "E21003";

static const std::string EM_FAILED_TO_TOPO_SORTING = "E2100C";

static const std::string EM_GRAPH_FUSION_FAILED = "E2100D";

static const std::string EM_GRAPH_PASS_OWNER_INVALID = "E2100E";

static const std::string EM_TAG_NO_CONST_FOLDING_FAILED = "E2100F";

static const std::string EM_COMMON_NULL_PTR = "E21010";

static const std::string EM_INVALID_IMPLEMENTATION = "E21011";

static const std::string EM_INNER_ERROR_1 = "E21012";

static const std::string EM_INVALID_OUTPUT_NAME_INDEX = "E21013";

static const std::string EM_INVALID_TENSOR_NAME_INDEX = "E21014";

static const std::string EM_FAILED_TO_ASSEMBLE_TBE_INFO = "E21015";

static const std::string EM_FAILED_TO_ASSEMBLE_INPUT_INFO = "E21016";

static const std::string EM_FAILED_TO_ASSEMBLE_OUTPUT_INFO = "E21017";

static const std::string EM_INVALID_TENSOR_DATA_TYPE = "E21018";

static const std::string EM_INVALID_ATTR_DATA_TYPE = "E21019";

static const std::string EM_SELECT_OP_FORMAT = "E21019";

static const std::string EM_FAILED_TO_PARSE_FORMAT_JSON = "E2101A";

static const std::string EM_FAILED_TO_CONVERT_FORMAT_FROM_JSON = "E2101B";

static const std::string EM_FORMAT_VECTOR_SIZE_INVALID = "E2101C";

static const std::string EM_INVALID_FORMAT_IN_JSON = "E2101D";

static const std::string EM_INVALID_DTYPE_IN_JSON = "E2101E";

static const std::string EM_ORIGINAL_DATATYPE_IS_NOT_SUPPORTED = "E21020";
}

#endif  // FUSION_ENGINE_INC_COMMON_FE_ERROR_CODE_H_
