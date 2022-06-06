import numpy as np

def mul_no_nan(x1, x2, y, kernel_name='mul_no_nan'):
    x1_data = x1.get('value')
    x2_data = x2.get('value')
    y = np.multiply(x1_data, x2_data)
    return y