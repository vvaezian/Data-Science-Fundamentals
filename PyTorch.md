#### Tensor Initialization
```py
# direct
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# random or constant data
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# from numpy
x_np = torch.from_numpy(np_array)

# from another tensor (retains the properties of source tensor unless overridden)
x_ones = torch.ones_like(x_data)
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of 
```

#### Tensor Attributes
```py
tensor.shape
tensor.dtype
tensor.device  # cou/gpu/tpu
```

#### Tensor Operations
```py
# numpi-like indexing and slicing
tensor = torch.ones(4, 4)
tensor[:,1] = 0  # sets the second column to zero

# joining tensors (also see tensor.stack)
t = torch.cat([t1, t2, t3], dim=1)

# element-wise product
t1.mul(t2)  # alternatively `t1 * t2`

# matric multiplication
t1.matmul(t2)  # alternatily, `t1 @ t2`

# transpose
t.T
```
