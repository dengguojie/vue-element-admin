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

#ifndef FUSION_ENGINE_INC_COMMON_CONFIGURATION_H_
#define FUSION_ENGINE_INC_COMMON_CONFIGURATION_H_

#include <map>
#include <set>
#include <mutex>
#include <string>
#include <vector>
#include <mutex>
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"
#include "common/aicore_util_types.h"
#include "graph/types.h"

namespace fe {
/** @brief Configuration.
* Used to manage all the configuration data within the fusion engine module. */
class Configuration {
 public:
  Configuration(const Configuration &) = delete;
  Configuration &operator=(const Configuration &) = delete;

  /**
   * Get the Singleton Instance by engine name
   */
  static Configuration &Instance(const std::string &engine_name);

  /**
   * Initialize the content_map and ops_store_info_vector_
   * Read the content from the config file, the save the data into content_map
   * Find the data about the Op Store Info from the content_map
   * and build the ops_store_info_vector_ with them.
   * @return Whether the object has been initialized successfully.
   */
  Status Initialize(const std::map<std::string, std::string> &options, std::string &soc_version);

  Status Finalize();

  /**
   * Find the FEOpsStoreInfo object with the OpImplType.
   * @param op_impl_type ImplType of OP storeinfo
   * @param op_store_info output value.
   *    if the object has been found, the op_store_info will refer to this object
   * @return Whether the FEOpsStoreInfo object has been found.
   * Status SUCCESS:found, FAILED:not found
   */
  Status GetOpStoreInfoByImplType(OpImplType op_impl_type, FEOpsStoreInfo &op_store_info) const;

  /*
   * to get the OpsStoreInfo out of the current configuration object
   * @return the OpsStoreInfo
   */
  const std::vector<FEOpsStoreInfo> &GetOpsStoreInfo() const;

  void SetOpsStoreInfo(const FEOpsStoreInfo &fe_ops_store_info);

  /*
   *  get small channel
   */
  bool GetEnableSmallChannel();

  /*
   *  get compress weight
   */
  bool GetEnableCompressWeight();

  /*
   * get is_auto_mix_precision switch, default value is false
   */
  bool GetAutoMixPrecisionSwitch();

  /*
   * to get l1fusion option out of the current configuration object
   * @return true/false
   */
  bool EnableL1Fusion();

  bool EnableL2Fusion();

  bool IsDCorMDCSoc() const;

  bool IsDCSoc() const;

  bool IsMDCSoc() const;

  bool IsCloudSoc();

  bool GetAllowAllScopeId();

  bool GetDuplicationSwitch();

  /*
   * to get switch switch of dump original nodes to fusion node
   * @return true/false
   */
  bool GetDumpOriginalNodesEnable();

  /*
   * to get the soc version out of the current configuration object
   * @return soc version
   */
  const std::string &GetSocVersion() const;

  /**
   * Get the rootdir from configuration file
   * @return root_path
   */
  std::string GetRootPath();

  const std::string &GetPrecisionModeStr();

  const std::string &GetLicenseFusionStr();

  AppendArgsMode GetAppendArgsMode();

  AutoTuneMode GetAutoTuneMode();

  /*
   * to get isa arch version out of the current configuration object
   * @return the value of chipset version
   */
  ISAArchVersion GetIsaArchVer();

  /*
   * to get BufferFusionMode option out of the current configuration object
   * @return BufferFusionMode
   */
  BufferFusionMode GetBufferFusionMode();

  bool IsEnableReuseMem() const;

  std::string GetBuiltInFusionConfigFilePath();

  std::string GetCustomFusionConfigFilePath();

  /**
   * Get the fusionpassmgr.graphpasspath from configuration file
   * @return builtin_pass_file_path
   */
  Status GetBuiltinPassFilePath(std::string &builtin_pass_file_path);

  /**
   * Get the fusionpassmgr.custompasspath from configuration file
   * @return custom_pass_file_path
   */
  Status GetCustomPassFilePath(std::string &custom_pass_file_path);

  /**
   * Get the fusionrulemgr.graphfilepath from configuration file
   * @return graphfilepath
   */
  Status GetGraphFilePath(std::string &graph_file_path);

  /**
   * Get the fusionrulemgr.customfilepath from configuration file
   * @return customfilepath
   */
  Status GetCustomFilePath(std::string &custom_file_path);

  Status InitBufferOptimize();

  Status InitPrecisionMode();

  void InitLicenseFusion(std::string &license_fusion_value);
  bool IsInLicenseControlMap(const std::string &key);

  std::vector<int64_t> GetQualifiedScopeId();
  std::string GetGeContextOptionValue(const std::string &key);

  std::string GetBuildStep();
  std::string GetBuildMode();
  std::string GetFeLibPath() const;
  int32_t GetDataVisitDistThreshold() const;

  void GetLicenseFusionDetailInfo(std::set<std::string> &license_detail_info) const;

  void GetModifyMixlist(std::string &modify_mixlist_path) const;

  void GetOpSelectImplModeStr(std::string &op_select_impl_mode_str) const;

 private:
  Configuration(std::string engine_name);
  ~Configuration();
  static const std::string CONFIG_FILE_RELATIVE_PATH;
  static const std::string CONFIG_OPS_RELATIVE_PATH;
  bool is_init_;
  std::string soc_version_;
  std::string lib_path_;
  std::string ascend_ops_path_;
  std::string engine_name_;
  std::string precision_mode_;
  std::string license_fusion_value_;
  std::set<std::string> license_fusion_detail_value_;
  std::string custom_fusion_config_file_;
  std::map<std::string, std::string> content_map_;
  std::vector<FEOpsStoreInfo> ops_store_info_vector_;
  bool enable_small_channel_;
  bool enable_compress_weight_;
  bool allow_all_scope_id_;
  ISAArchVersion isa_arch_ver_;
  AppendArgsMode append_args_mode_;
  BufferFusionMode buffer_fusion_mode_;
  BufferOptimize buffer_optimize_;
  AutoTuneMode auto_tune_mode_;
  bool is_enable_reuse_mem_ = true;
  std::vector<int64_t> qualified_scope_id_;
  mutable std::mutex ops_store_info_vector_mutex_;
  int32_t data_visit_dist_threshold_;
  std::map<std::string, std::string> op_select_impl_mode_map_;
  std::string modify_mixlist_path_;
  /**
   * Initialize the parameters from options
   * @param options patameters map
   */
  Status InitOptions(const std::map<std::string, std::string> &options);

  template <typename T>
  Status InitFromGeContext(const std::map<std::string, T> str_map, const std::string &key, T &value);

  template <typename T>
  Status InitSingleParamFromOptions(const std::map<std::string, std::string> &options,
                                    const std::map<std::string, T> str_map, const std::string &key, T &value);

  /**
   * Get the real Path of current so lib
   */
  Status InitLibPath();

  /**
   * Get the real Path of ops
   * path of ops is the path of so package + ops_relative_path
   */
  Status InitAscendOpsPath();

  /**
   * Read the content of configuration file(FE_CONFIG_FILE_PATH)
   * Save the data into content_map
   * @return Whether the config file has been loaded successfully.
   */
  Status LoadConfigFile();

  /**
   * Validate whether all the mandatory config items is configured.
   * @return whether they are configured.
   */
  Status ValidateConfigItems();

  /**
   * If the config keys in DEFAULT_CONFIG_ITEM_VALUE is not configured,
   * set the default values for them.
   */
  void FillupDefaultConfigValues();

  /**
   * Find the OpsStoreInfo from the content_map,
   * then use the data to build up ops_store_info_vector.
   * @return Whether the OpsStoreInfoVector has been built up successfully.
   */
  Status AssembleOpsStoreInfoVector();

  Status AssembleEachOpsStoreInfo(std::string &op_store_name, std::vector<std::string> &op_store_vector,
                                 std::vector<OpImplType> &existed_op_impl_types, FEOpsStoreInfo &ops_store_info);

  Status VerifyOpStoreVector(std::vector<std::string> &op_store_vector, const std::string &store_name);

  bool IsIgnoreOpStore(FEOpsStoreInfo &ops_store_info);

  Status CheckOpStoreInfo(const FEOpsStoreInfo &op_store_info);

  /**
   * Check whether the content_map contain the input key.
   * @param key
   * @return Whether the content_map contain the input key.
   */
  bool ContainKey(const std::string &key);

  /**
   * Get the value from the content_map if the content_map contains the input key.
   * @param key config key
   * @param return_value output value. if the value has been found,
   *                    return_value will refer to this value.
   * @return Whether the vale has been found.
   *         Status SUCCESS:found, FAILED:not found
   */
  Status GetStringValue(const std::string &key, std::string &return_value);

  /**
   * Find the value from the content_map by the input key,
   * convert the value to bool type and return the bool value
   * return the input default value if the value is not found.
   * @param key
   * @param default_value
   *   This value will be returned if the input key can be found in content_map.
   * @return bool value
   */
  bool GetBoolValue(const std::string &key, bool default_value);

  /**
   * Find the value from the ge context by the input key,
   * convert the value to bool type and return the bool value
   * return the input default value if the value is not found.
   * @param key
   * @param default_value
   *   This value will be returned if the input key can be found in content_map.
   * @return bool value
   */
  bool GetGeContextBoolValue(const std::string &key, bool default_value);

  Status GetGeContextStringValue(const std::string &key, std::string &option_value);

  void InitParametersOfConfigFile();

  /* to init BufferFusionMode option of the current configuration object
  */
  void InitBufferFusionMode();

  void InitScopeId();

  void InitISAArchVersion();

  void InitAppendArgsMode();

  int32_t ParseDataVisitDistThreshold();

  std::vector<std::string> ParseConfig(const std::string &key, char pattern);

  Status InitOpPrecisionMode(const std::map<std::string, std::string> &options);

  Status InitOpPrecisionModeByPrecisionMode(const std::string &op_precision_mode);

  Status InitOpPrecisionModeByImplModeAll(const std::string &op_select_impl_mode_all);

  Status InitOpPrecisionModeByImplMode(const std::string &op_select_impl_mode,
                                       const std::string &op_type_list_str);

  Status GetOpPrecisonModeStrFromConfigFile(const std::string &file_path);
};
}  // namespace fe
#endif  // FUSION_ENGINE_INC_COMMON_CONFIGURATION_H_
