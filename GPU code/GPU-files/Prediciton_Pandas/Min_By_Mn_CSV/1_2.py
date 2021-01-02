import cupy
x = cupy.arange(3)
x[[1, 3]] = 10
print(x)
#array([10, 10,  2])
