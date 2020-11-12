import os
import numpy as np


def gen_data_file(data_shape, value_range, dtype, data_distribution, out_data_file):
    def _gen_data(data_shape, min_value, max_value, dtype,
                  distribution='uniform'):
        if 'uniform' in distribution:
            # Returns the uniform distribution random value.
            # min indicates the random minimum value,
            # and max indicates the random maximum value.
            return np.random.uniform(low=min_value, high=max_value,
                                     size=data_shape).astype(dtype)
        if 'normal' in distribution:
            # Returns the normal (Gaussian) distribution random value.
            # min is the central value of the normal distribution,
            # and max is the standard deviation of the normal distribution.
            # The value must be greater than 0.
            return np.random.normal(loc=min_value,
                                    scale=abs(max_value) + 1e-4,
                                    size=data_shape).astype(dtype)
        if 'beta' in distribution:
            # Returns the beta distribution random value.
            # min is alpha and max is beta.
            # The values of both min and max must be greater than 0.
            return np.random.beta(a=abs(min_value) + 1e-4,
                                  b=abs(max_value) + 1e-4,
                                  size=data_shape).astype(dtype)
        if 'laplace' in distribution:
            # Returns the Laplacian distribution random value.
            # min is the central value of the Laplacian distribution,
            # and max is the exponential attenuation of the Laplacian
            # distribution.  The value must be greater than 0.
            return np.random.laplace(loc=min_value,
                                     scale=abs(max_value) + 1e-4,
                                     size=data_shape).astype(dtype)
        if 'triangular' in distribution:
            # Return the triangle distribution random value.
            # min is the minimum value of the triangle distribution,
            # mode is the peak value of the triangle distribution,
            # and max is the maximum value of the triangle distribution.
            mode = np.random.uniform(low=min_value, high=max_value)
            return np.random.triangular(left=min_value, mode=mode,
                                        right=max_value,
                                        size=data_shape).astype(dtype)
        if 'relu' in distribution:
            # Returns the random value after the uniform distribution
            # and relu activation.
            data_pool = np.random.uniform(low=min_value, high=max_value,
                                          size=data_shape).astype(dtype)
            return np.maximum(0, data_pool)
        if 'sigmoid' in distribution:
            # Returns the random value after the uniform distribution
            # and sigmoid activation.
            data_pool = np.random.uniform(low=min_value, high=max_value,
                                          size=data_shape).astype(dtype)
            return 1 / (1 + np.exp(-data_pool))
        if 'softmax' in distribution:
            # Returns the random value after the uniform distribution
            # and softmax activation.
            data_pool = np.random.uniform(low=min_value, high=max_value,
                                          size=data_shape).astype(dtype)
            return np.exp(data_pool) / np.sum(np.exp(data_pool))
        if 'tanh' in distribution:
            # Returns the random value after the uniform distribution
            # and tanh activation.
            data_pool = np.random.uniform(low=min_value, high=max_value,
                                          size=data_shape).astype(dtype)
            return (np.exp(data_pool) - np.exp(-data_pool)) / \
                   (np.exp(data_pool) + np.exp(-data_pool))
        # Return the uniform distribution random value for True,False.
        # advised to set range in the excel design table to [0, 2].
        return np.random.uniform(low=0, high=2, size=data_shape).astype(
            np.int8).astype(dtype)

    out_data_dir = os.path.dirname(out_data_file)
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)
    data = _gen_data(data_shape, value_range[0], value_range[1], dtype, data_distribution)
    data.tofile(out_data_file)
    return data
