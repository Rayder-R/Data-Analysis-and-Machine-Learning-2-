import matplotlib.pyplot as plt
import numpy as np

# Create a 6x6 matrix
matrix = np.array([[1, 2, 3, 4, 5, 6],
                   [7, 8, 9, 10, 11, 12],
                   [13, 14, 15, 16, 17, 18],
                   [19, 20, 21, 22, 23, 24],
                   [25, 26, 27, 28, 29, 30],
                   [31, 32, 33, 34, 35, 36]])

# Create the plot
fig, ax = plt.subplots()
ax.matshow(matrix, cmap='Blues')

# Add text inside each cell of the matrix
for (i, j), val in np.ndenumerate(matrix):
    ax.text(j, i, str(val), ha='center', va='center', color='white')

ax.set_title("6x6 Matrix Visualization")
plt.show()
