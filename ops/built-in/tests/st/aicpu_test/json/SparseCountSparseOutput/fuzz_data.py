import tensorflow as tf
import numpy as np

def int64_512B():
  rng = np.random.RandomState()
  shape = [1, 64]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j])

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int64)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def int64_1K():
  rng = np.random.RandomState()
  shape = [2, 64]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int64)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def int64_4K():
  rng = np.random.RandomState()
  shape = [8, 64]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int64)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def int64_8K():
  rng = np.random.RandomState()
  shape = [16, 64]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int64)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def int64_16K():
  rng = np.random.RandomState()
  shape = [32, 64]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int64)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def int64_32K():
  rng = np.random.RandomState()
  shape = [64, 64]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int64)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def int64_64K():
  rng = np.random.RandomState()
  shape = [128, 64]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int64)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def int64_128K():
  rng = np.random.RandomState()
  shape = [256, 64]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int64)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def int64_256K():
  rng = np.random.RandomState()
  shape = [512, 64]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int64)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def int64_512K():
  rng = np.random.RandomState()
  shape = [1024, 64]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int64)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def int64_1M():
  rng = np.random.RandomState()
  shape = [2048, 64]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int64)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def int64_2M():
  rng = np.random.RandomState()
  shape = [4096, 64]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int64)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def int64_8M():
  rng = np.random.RandomState()
  shape = [16384, 64]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int64)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def int32_512B():
  rng = np.random.RandomState()
  shape = [1, 128]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int32)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def int32_1K():
  rng = np.random.RandomState()
  shape = [2, 128]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int32)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def int32_4K():
  rng = np.random.RandomState()
  shape = [8, 128]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int32)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def int32_8K():
  rng = np.random.RandomState()
  shape = [16, 128]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int32)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def int32_16K():
  rng = np.random.RandomState()
  shape = [32, 128]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int32)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def int32_32K():
  rng = np.random.RandomState()
  shape = [64, 128]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int32)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def int32_64K():
  rng = np.random.RandomState()
  shape = [128, 128]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int32)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def int32_128K():
  rng = np.random.RandomState()
  shape = [256, 128]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int32)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def int32_256K():
  rng = np.random.RandomState()
  shape = [512, 128]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int32)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def int32_512K():
  rng = np.random.RandomState()
  shape = [1024, 128]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int32)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def int32_1M():
  rng = np.random.RandomState()
  shape = [2048, 128]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int32)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def int32_2M():
  rng = np.random.RandomState()
  shape = [4096, 128]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int32)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def int32_8M():
  rng = np.random.RandomState()
  shape = [16384, 128]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int32)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def example_int64():
  rng = np.random.RandomState()
  shape = [32, 32]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int64)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }

def example_int32():
  rng = np.random.RandomState()
  shape = [32, 32]
  indices = []
  for i in range(shape[0]):
    for j in range(shape[1]):
      indices.append([i, j]) if rng.randint(0, 2) else ...

  values = rng.randint(1, 1000, size=len(indices)).astype(np.int32)

  return {
    "input_desc": {
      "indices": {"value": indices, "shape": [len(indices), 2]},
      "values": {"value": values.tolist(), "shape": [len(values)]},
      "dense_shape": {"value": shape, "shape": [len(shape)]},
      "weights": {"value": values.tolist(), "shape": [len(values)]}
    }
  }
