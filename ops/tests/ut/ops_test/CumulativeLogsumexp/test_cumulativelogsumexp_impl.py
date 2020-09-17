#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("CumulativeLogsumexp", "impl.cumulativelogsumexp_d", "cumulative_logsumexp_d")

ut_case.add_case("all", {
    "params": [
        {"shape": (1,), "dtype": "float16", "format": "ND",
         "ori_shape": (1,), "ori_format": "ND"},
        {"shape": (1,), "dtype": "float16", "format": "ND",
         "ori_shape": (1,), "ori_format": "ND"},
        0, False, False],
    "expect": "success"
})

ut_case.add_case("Ascend910", {
    "params": [
        {"shape": (1, 16, 16), "dtype": "float32", "format": "ND",
         "ori_shape": (1, 16, 16), "ori_format": "ND"},
        {"shape": (1, 16, 16), "dtype": "float32", "format": "ND",
         "ori_shape": (1, 16, 16), "ori_format": "ND"},
        0, False, False],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [
        {"shape": (1, 16, 16), "dtype": "float16", "format": "ND",
         "ori_shape": (1, 16, 16), "ori_format": "ND"},
        {"shape": (1, 16, 16), "dtype": "float16", "format": "ND",
         "ori_shape": (1, 16, 16), "ori_format": "ND"},
        0, True, False],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [
        {"shape": (1, 16, 16), "dtype": "float16", "format": "ND",
         "ori_shape": (1, 16, 16), "ori_format": "ND"},
        {"shape": (1, 16, 16), "dtype": "float16", "format": "ND",
         "ori_shape": (1, 16, 16), "ori_format": "ND"},
        0, False, True],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [
        {"shape": (1, 16, 16), "dtype": "float16", "format": "ND",
         "ori_shape": (1, 16, 16), "ori_format": "ND"},
        {"shape": (1, 16, 16), "dtype": "float16", "format": "ND",
         "ori_shape": (1, 16, 16), "ori_format": "ND"},
        0, True, True],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [
        {"shape": (1, 16, 13), "dtype": "float16", "format": "ND",
         "ori_shape": (1, 16, 13), "ori_format": "ND"},
        {"shape": (1, 16, 13), "dtype": "float16", "format": "ND",
         "ori_shape": (1, 16, 13), "ori_format": "ND"},
        0, True, True],
    "expect": "success"
})

ut_case.add_case("Ascend910", {
    "params": [
        {"shape": (1, 12, 13), "dtype": "float32", "format": "ND",
         "ori_shape": (1, 12, 13), "ori_format": "ND"},
        {"shape": (1, 12, 13), "dtype": "float32", "format": "ND",
         "ori_shape": (1, 12, 13), "ori_format": "ND"},
        0, True, True],
    "expect": "success"
})

ut_case.add_case("Ascend910", {
    "params": [
        {"shape": (1, 12, 13), "dtype": "float32", "format": "ND",
         "ori_shape": (1, 12, 13), "ori_format": "ND"},
        {"shape": (1, 12, 13), "dtype": "float32", "format": "ND",
         "ori_shape": (1, 12, 13), "ori_format": "ND"},
        -2, True, True],
    "expect": "success"
})

ut_case.add_case("Ascend910", {
    "params": [
        {"shape": (1, 12, 13, 15), "dtype": "float32", "format": "ND",
         "ori_shape": (1, 12, 13, 15), "ori_format": "ND"},
        {"shape": (1, 12, 13, 15), "dtype": "float32", "format": "ND",
         "ori_shape": (1, 12, 13, 15), "ori_format": "ND"},
        2, True, True],
    "expect": "success"
})

ut_case.add_case("Ascend910", {
    "params": [
        {"shape": (1, 12, 13, 15), "dtype": "float32", "format": "ND",
         "ori_shape": (1, 12, 13, 15), "ori_format": "ND"},
        {"shape": (1, 12, 13, 15), "dtype": "float32", "format": "ND",
         "ori_shape": (1, 12, 13, 15), "ori_format": "ND"},
        3, True, True],
    "expect": "success"
})

ut_case.add_case("Ascend910", {
    "params": [
        {"shape": (10, 12, 13, 15, 223), "dtype": "float32", "format": "ND",
         "ori_shape": (10, 12, 13, 15, 223), "ori_format": "ND"},
        {"shape": (10, 12, 13, 15, 223), "dtype": "float32", "format": "ND",
         "ori_shape": (10, 12, 13, 15, 223), "ori_format": "ND"},
        1, True, True],
    "expect": "success"
})

ut_case.add_case("Ascend910", {
    "params": [
        {"shape": (10, 12, 13, 15, 223), "dtype": "float32", "format": "ND",
         "ori_shape": (10, 12, 13, 15, 223), "ori_format": "ND"},
        {"shape": (10, 12, 13, 15, 223), "dtype": "float32", "format": "ND",
         "ori_shape": (10, 12, 13, 15, 223), "ori_format": "ND"},
        1, False, True],
    "expect": "success"
})

ut_case.add_case("Ascend910", {
    "params": [
        {"shape": (10, 12, 13, 15, 223), "dtype": "float32", "format": "ND",
         "ori_shape": (10, 12, 13, 15, 223), "ori_format": "ND"},
        {"shape": (10, 12, 13, 15, 223), "dtype": "float32", "format": "ND",
         "ori_shape": (10, 12, 13, 15, 223), "ori_format": "ND"},
        1, True, False],
    "expect": "success"
})
ut_case.add_case("Ascend910", {
    "params": [
        {"shape": (10, 12, 13, 15, 223), "dtype": "float32", "format": "ND",
         "ori_shape": (10, 12, 13, 15, 223), "ori_format": "ND"},
        {"shape": (10, 12, 13, 15, 223), "dtype": "float32", "format": "ND",
         "ori_shape": (10, 12, 13, 15, 223), "ori_format": "ND"},
        1, False, False],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [
        {"shape": (10, 1, 13, 15, 223), "dtype": "float16", "format": "ND",
         "ori_shape": (10, 1, 13, 15, 223), "ori_format": "ND"},
        {"shape": (10, 1, 13, 15, 223), "dtype": "float16", "format": "ND",
         "ori_shape": (10, 1, 13, 15, 223), "ori_format": "ND"},
        1, True, False],
    "expect": "success"
})

ut_case.add_case("Ascend910", {
    "params": [
        {"shape": (35, 1, 13, 15, 223), "dtype": "float32", "format": "ND",
         "ori_shape": (35, 1, 13, 15, 223), "ori_format": "ND"},
        {"shape": (35, 1, 13, 15, 223), "dtype": "float32", "format": "ND",
         "ori_shape": (35, 1, 13, 15, 223), "ori_format": "ND"},
        1, True, False],
    "expect": "success"
})

ut_case.add_case("Ascend910", {
    "params": [
        {"shape": (35, 1, 13, 15, 223), "dtype": "float32", "format": "ND",
         "ori_shape": (35, 1, 13, 15, 223), "ori_format": "ND"},
        {"shape": (35, 1, 13, 15, 223), "dtype": "float32", "format": "ND",
         "ori_shape": (35, 1, 13, 15, 223), "ori_format": "ND"},
        -1, True, False],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [
        {"shape": (1, 2, 3, 4), "dtype": "int16", "format": "ND",
         "ori_shape": (1, 2, 3, 4), "ori_format": "ND"},
        {"shape": (1, 2, 3, 4), "dtype": "int16", "format": "ND",
         "ori_shape": (1, 2, 3, 4), "ori_format": "ND"},
        2, False, False],
    "expect": RuntimeError
})

ut_case.add_case("Ascend910", {
    "params": [
        {"shape": (1, 2, 4, 5), "dtype": "float32", "format": "ND",
         "ori_shape": (1, 2, 3, 4), "ori_format": "ND"},
        {"shape": (1, 2, 3, 4), "dtype": "float32", "format": "ND",
         "ori_shape": (1, 2, 3, 4), "ori_format": "ND"},
        10, False, False],
    "expect": RuntimeError
})

if __name__ == '__main__':
    ut_case.run("Ascend910")
