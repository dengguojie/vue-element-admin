/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * Description: read file from txt file
 */

#ifndef OPS_TEST_UTILS_AICPU_READ_FILE_H_
#define OPS_TEST_UTILS_AICPU_READ_FILE_H_
#include <iostream>
#include <string>
#include <fstream>
#include <exception>
#include <vector>
#include "Eigen/Core"

const std::string ktestcaseFilePath =
    "../../../../../../../ops/built-in/tests/ut/aicpu_test/testcase/";

bool ReadFile(std::string file_name, Eigen::half output[], uint64_t size);

template<typename T>
bool ReadFile(std::string file_name, std::vector<T> &output) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }
    T tmp;
    while (in_file >> tmp) {
      output.push_back(tmp);
    }
    in_file.close();
  } catch (std::exception &e) {
    std::cout << "read file " << file_name << " failed, "
              << e.what() << std::endl;
    return false;
  }
  return true;
}

template<typename T>
bool ReadFile(std::string file_name, T output[], uint64_t size) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }
    T tmp;
    uint64_t index = 0;
    while (in_file >> tmp) {
      if (index >= size) {
        break;
      }
      output[index] = tmp;
      index++;
    }
    in_file.close();
  } catch (std::exception &e) {
    std::cout << "read file " << file_name << " failed, "
              << e.what() << std::endl;
    return false;
  }
  return true;
}

#endif