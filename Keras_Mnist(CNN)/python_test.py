import numpy as np
b=np.arange(start=1, stop=9, step=1)


# print(b.reshape(1))
print(b)
print("-------------------")
print(b.reshape(2,4))
print("-------------------")
print(b.reshape(2,2,2))
print("-------------------")
print(b.reshape(1,2,2,2))