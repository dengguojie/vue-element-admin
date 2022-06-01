#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Third-Party Packages
import numpy as np


def calc_expect_func(var, m, v, vhat, beta1_power, beta2_power, lr, grad, var_output,
                     m_output, v_output, vhat_output, beta1, beta2, epsilon, use_locking=False):

    var_input = var["value"]
    m_input = m["value"]
    v_input = v["value"]
    vhat_input = vhat["value"]
    beta1_power_input = beta1_power["value"]
    beta2_power_input = beta2_power["value"]
    lr_input = lr["value"]
    grad_input = grad["value"]

    lr_t = lr_input * np.sqrt(1 - beta2_power_input) / (1 - beta1_power_input)
    m_t = beta1* m_input + (1 - beta1) * grad_input
    v_t = beta2 * v_input + (1 - beta2) * grad_input * grad_input
    vhat_t = np.maximum(vhat_input, v_t)
    var_t = var_input - lr_t * m_t / (np.sqrt(vhat_t) + epsilon)

    var_t = var_t.astype(var["dtype"])
    m_t = m_t.astype(var["dtype"])
    v_t = v_t.astype(var["dtype"])
    vhat_t = vhat_t.astype(var["dtype"])

    return [var_t, m_t, v_t, vhat_t]