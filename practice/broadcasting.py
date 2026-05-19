import numpy as np

arr1 = np.array([[1, 2, 4], [5, 6, 7]])
arr2 = np.array([[10], [20]])

res = arr1 + arr2

print(res)

a = np.arange(9)
a = a.reshape(3, 3)

print(a)

b = np.arange(100)
b = b.reshape(10, 10)
print(b)
print(b.reshape(4, 5, 5))
print(b.reshape(-1, 5, 5)) # -1 trick lets numpy figure out the size of the dimension / you can only use once in the reshape function

b = np.arange(9).reshape(-1, 3)
print(b)
print()
print(b.ravel())
print(b.flatten())
print(b.reshape(-1))

c = a.flatten()
print(f"the base of c is {c.base}")
b = a.ravel()
print(f"the base of b is {b.base}")
d = a.reshape(-1)
print(f"the base of d is {d.base}")

a = np.arange(9)
print(a)
print(a.shape)
print()

# add one dimension to a
b = a[:, np.newaxis] # np.newaxis is an alias for None, more readable
c = a[:, None] # same as above

print(b)
print(b.shape)
print()
print(c)
print(c.shape)

