#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
cmpsel simulator
"""
import tbe.dsl.base.padding.simulator as m_simulator
import tbe.dsl.base.padding.util as util


class CmpGtSimulator(m_simulator.Simulator):
    @classmethod
    def get_type(cls):
        return "elewise_binary_vcmpv_gt"

    def adjust_calc(self):
        util.raise_error("Unsupported.")


class CmpGeSimulator(m_simulator.Simulator):
    @classmethod
    def get_type(cls):
        return "elewise_binary_vcmpv_ge"

    def adjust_calc(self):
        util.raise_error("Unsupported.")


class CmpLtSimulator(m_simulator.Simulator):
    @classmethod
    def get_type(cls):
        return "elewise_binary_vcmpv_lt"

    def adjust_calc(self):
        util.raise_error("Unsupported.")


class CmpLeSimulator(m_simulator.Simulator):
    @classmethod
    def get_type(cls):
        return "elewise_binary_vcmpv_le"

    def adjust_calc(self):
        util.raise_error("Unsupported.")


class CmpEqSimulator(m_simulator.Simulator):
    @classmethod
    def get_type(cls):
        return "elewise_binary_vcmpv_eq"

    def adjust_calc(self):
        util.raise_error("Unsupported.")


class CmpNeSimulator(m_simulator.Simulator):
    @classmethod
    def get_type(cls):
        return "elewise_binary_vcmpv_ne"

    def adjust_calc(self):
        util.raise_error("Unsupported.")


class CmpselGtSimulator(m_simulator.Simulator):
    @classmethod
    def get_type(cls):
        return "elewise_binary_cmpsel_gt"

    def adjust_calc(self):
        util.raise_error("Unsupported.")


class CmpselGeSimulator(m_simulator.Simulator):
    @classmethod
    def get_type(cls):
        return "elewise_binary_cmpsel_ge"

    def adjust_calc(self):
        util.raise_error("Unsupported.")


class CmpselLtSimulator(m_simulator.Simulator):
    @classmethod
    def get_type(cls):
        return "elewise_binary_cmpsel_lt"

    def adjust_calc(self):
        util.raise_error("Unsupported.")


class CmpselLeSimulator(m_simulator.Simulator):
    @classmethod
    def get_type(cls):
        return "elewise_binary_cmpsel_le"

    def adjust_calc(self):
        util.raise_error("Unsupported.")


class CmpselEqSimulator(m_simulator.Simulator):
    @classmethod
    def get_type(cls):
        return "elewise_binary_cmpsel_eq"

    def adjust_calc(self):
        util.raise_error("Unsupported.")


class CmpselNeSimulator(m_simulator.Simulator):
    @classmethod
    def get_type(cls):
        return "elewise_binary_cmpsel_ne"

    def adjust_calc(self):
        util.raise_error("Unsupported.")


class SelSimulator(m_simulator.Simulator):
    @classmethod
    def get_type(cls):
        return "elewise_multiple_sel"

    def adjust_calc(self):
        util.raise_error("Unsupported.")
