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

#ifndef FUSION_ENGINE_INC_COMMON_STRING_UTILS_H_
#define FUSION_ENGINE_INC_COMMON_STRING_UTILS_H_

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>
#include "common/fe_log.h"
#include "graph/types.h"
#include "graph/utils/type_utils.h"

namespace fe {
using std::string;

class StringUtils {
 public:
  /** @ingroup domi_common
   *  @brief split string
   *  @link http://stackoverflow.com/questions/236129/how-to-split-a-string-in-c
   *  @param [in] str   : string need to be split
   *  @param [in] pattern : separator
   *  @return string vector after split
   */
  static std::vector<string> Split(const string &str, char pattern) {
    std::vector<string> res_vec;
    if (str.empty()) {
      return res_vec;
    }
    string str_and_pattern = str + pattern;
    size_t pos = str_and_pattern.find(pattern);
    size_t size = str_and_pattern.size();
    while (pos != string::npos) {
      string sub_str = str_and_pattern.substr(0, pos);
      res_vec.push_back(sub_str);
      str_and_pattern = str_and_pattern.substr(pos + 1, size);
      pos = str_and_pattern.find(pattern);
    }
    return res_vec;
  }

  static string &Ltrim(string &s) {
#if __cplusplus >= 201103L
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int c) { return !std::isspace(c); }));
#else
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
#endif
    return s;
  }

  static string &Rtrim(string &s) {
#if __cplusplus >= 201103L
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int c) { return !std::isspace(c); }).base(), s.end());
#else
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
#endif
    return s;
  }

  /** @ingroup domi_common
   *  @brief remove the white space at the begining and end position of string
   *  @param [in] s : the string need to be trim
   *  @return the string after trim
   */
  static string &Trim(string &s) { return Ltrim(Rtrim(s)); }

  /** @ingroup domi_common
  *  @brief whether a string is start with a specific string（prefix）
  *  @param [in] str    : a string to be compared
  *  @param [in] prefix : a string prefix
  *  @return true else false
  */
  static bool StartWith(const string &str, const string prefix) {
    return ((str.size() >= prefix.size()) && (str.compare(0, prefix.size(), prefix) == 0));
  }

  static bool EndWith(const string &str, const string &suffix) {
    if (str.length() < suffix.length()) {
      return false;
    }
    string sub_str = str.substr(str.length() - suffix.length());
    return sub_str == suffix;
  }

  /** @ingroup fe
  *  @brief transfer a string to upper string
  *  @param [in] str : a string to be transfer
  *  @return None
  */
  static void ToUpperString(string &str) { transform(str.begin(), str.end(), str.begin(), (int (*)(int))toupper); }

  /** @ingroup fe
  *  @brief transfer a string to lower string
  *  @param [in] str : a string to be transfer
  *  @return None
  */
  static void ToLowerString(string &str) { transform(str.begin(), str.end(), str.begin(), (int (*)(int))tolower); }

  /** @ingroup fe
   *  @brief whether a string match another string, after triming space
   *  @param [in] left    : a string to be matched
   *  @param [in] right   : a string to be matched
   *  @return true else false
   */
  static bool MatchString(string left, string right) {
    if (Trim(left) == Trim(right)) {
      return true;
    }
    return false;
  }

  static bool MatchWithoutCase(string left, string right) {
    StringUtils::ToLowerString(left);
    StringUtils::ToLowerString(right);
    if (Trim(left) == Trim(right)) {
      return true;
    }
    return false;
  }

  /**
 * check whether the input str is integer
 * @param[in] str input string
 * @param[out] true/false
 */
  static bool IsInteger(const string &str) {
    if (str.empty()) {
      return false;
    }
    for (auto c : str) {
      if (!isdigit(c)) {
        return false;
      }
    }
    return true;
  }

  template <typename T>
  static string IntegerVecToString(const std::vector<T> &integer_vec) {
    string result = "{";
    using DT = typename std::remove_cv<T>::type;
    for (auto ele : integer_vec) {
      string ele_str;
      if (std::is_enum<DT>::value) {
        if (std::is_same<ge::Format, DT>::value) {
          ele_str = ge::TypeUtils::FormatToSerialString(static_cast<ge::Format>(ele));
        } else if (std::is_same<ge::DataType, DT>::value) {
          ele_str = ge::TypeUtils::DataTypeToSerialString(static_cast<ge::DataType >(ele));
        } else {
          ele_str = std::to_string(ele);
        }
      }else {
        ele_str = std::to_string(ele);
      }
      result += ele_str;
      result += ",";
    }
    if (result.empty()) {
      return "";
    }
    /* Get rid of the last comma. */
    if (result.back() == ',') {
      result = result.substr(0, result.length() - 1);
    }
    result += "}";
    return result;
  }

};
}  // namespace fe
#endif  // FUSION_ENGINE_INC_COMMON_STRING_UTILS_H_
