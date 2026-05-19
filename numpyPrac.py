import numpy as np

arr1 = np.array([[2, 3],
                 [4, 5]])
arr2 = np.array([[6, 7],
                 [3, 9]])

arr_result = np.multiply(arr1, arr2)

print(f"Element-wise product of arr1 and arr2 is: \n{arr_result}")

arr_result = np.matmul(arr1, arr2)

print(f"Matrix product of arr1 and arr2 is: \n{arr_result}")

arr_result = arr2 @ arr1

print(f"Matrix product of arr2 and arr1 is: \n{arr_result}")

arr_result = np.dot(arr1, arr2)

print(f"Dot product of arr1 and arr2 is: \n{arr_result}")

arr_result = np.dot(arr2, arr1)

print(f"Dot product of arr2 and arr1 is: \n{arr_result}")