/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * Description: read file from txt file
 */

#include "aicpu_read_file.h"
#include <sstream>
#include <algorithm>

bool StringToBool(const std::string &str, bool &result) {
  result = false;
  std::string buff = str;
  try {
    std::transform(buff.begin(), buff.end(), buff.begin(),
        [](unsigned char c) -> unsigned char {return std::tolower(c);});
    if ((buff == "false") || (buff == "true")) {
      std::istringstream(buff) >> std::boolalpha >> result;
      return true;
    } else {
      std::cout << "convert to bool failed, " << buff
                << " is not a bool value." << std::endl;
      return false;
    }
  } catch (std::exception &e) {
    std::cout << "convert " << str << " to bool failed, "
              << e.what() << std::endl;
  }
  return false;
}

template<>
bool ReadFile(std::string file_name, std::vector<bool> &output) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
        std::cout << "open file: " << file_name << " failed." <<std::endl;
        return false;
    }
    bool tmp;
    std::string read_str;
    while (in_file >> read_str) {
        if (StringToBool(read_str, tmp)) {
          output.push_back(tmp);
        }
    }
    in_file.close();
  } catch (std::exception &e) {
      std::cout << "read file " << file_name << " failed, "
                << e.what() << std::endl;
    return false;
  }
  return true;
}

template<>
bool ReadFile(std::string file_name, bool output[], uint64_t size) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
        std::cout << "open file: " << file_name << " failed." <<std::endl;
        return false;
    }
    bool tmp;
    std::string read_str;
    uint64_t index = 0;
    while (in_file >> read_str) {
        if (StringToBool(read_str, tmp)) {
          if (index >= size) {
            break;
          }
          output[index] = tmp;
          index++;
        }
    }
    in_file.close();
  } catch (std::exception &e) {
      std::cout << "read file " << file_name << " failed, "
                << e.what() << std::endl;
    return false;
  }
  return true;
}