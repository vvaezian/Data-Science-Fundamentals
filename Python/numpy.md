Creating random data
````Python
x = np.arange(0, 1, .1)  # like range but can have non-integer step-size

# just random uniform distributions in differnt range
y1 = np.random.uniform(10,15,10)
y2 = np.random.uniform(20,25,10)
y3 = np.random.uniform(0,5,10)

y = np.concatenate((y1,y2,y3))
y = y[:,None]
````

Difference between list and numpy array:  
* Lists don’t support “vectorized” operations like elementwise addition and multiplication.
* In numpy arrays all elements have the same type which make it faster
* Shallow copy vs. view (explained further below)
````Python
np.random.standard_normal(size=(2, 3))  # creates a 2x3 matrix with values from standard normal distribution
np.random.randn(2,3)                    # same as above
arr = np.arange(-2, 2, .5)              # array([-2.0 -1.5 -1.0 -0.5  0.0  0.5  1.0  1.5])
arr.reshape((4, 2))                     # creates a two-dimentional array of 4 rows and 2 cols
np.dot(m1, m2)                          # matrix multiplication, or vector dot product
np.multiply(m1, m2)                     # element-wise multiplication
arr = np.random.randint(10, size=(3, 4))  # Two-dimensional array
````
#### Array Slicing
````Python
arr[rowStart:rowEnd:step, colStart:colEnd:step]
````
Accessing a single row or col
````Python
arr[:, 0] # first column
arr[2, :] # third row. Can be shortened using `arr[2]`
````
Sub-arryas are 'views' not shallow copies like sub-lists. So if we change a subarray, the actual array will be altered. 
This is useful when working with large datasets. To avoid this we can use arr.copy()
````Python
subArr = arr[:2, :2]
subArrCopy = arr[:2, :2].copy()
````
#### Concatenate and Split
````Python
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
np.concatenate([x, y])  # array([1, 2, 3, 4, 5, 6])
````
For multi-dimential array we can indicate which axis we want the concatenation to occur
````Python
np.concatenate(arr1, arr2, axis=1)  # 1 is for col, 0 is for row (default)
````
For mixed dimentions it is easier to use `np.vstack` (vertical stack) and `np.hstack` (horizontal stack) functions
````Python
np.vstack([arr1, arr2]) # vertivally stack arr1 on arr2
x = np.array([[1], [2]])
grid = np.array([[9, 8, 7],
                 [6, 5, 4]])
np.hstack([grid, x]) # [[9, 8, 7, 1]
                     #  [6, 5, 4, 2]]
````
`np.dstack` will stack arrays along the third axis.  

For splitting arrays (opposite of concatenating) `np.split`, `np.hsplit`, `np.vsplit` and `np.dsplit` can be used.
