import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def show_data(nparr):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(nparr[:, 0], nparr[:, 1], nparr[:, 2], c='b', marker='.')
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    plt.show()


def eig_show_data(eigen_values, eigen_vectors):
    plt.figure(figsize=(10, 10))
    plt.xlabel("Eigenvectors")
    plt.ylabel("Eigenvalues")
    plt.bar([str(i) for i in eigen_vectors], eigen_values, width=0.4)
    plt.show()


def image_array(image_name):
    filename = image_name
    image = Image.open(filename)
    width, height = image.size
    npimage = np.array(image)
    arr = []
    for y in range(height - 1):
        for x in range(width - 1):
            arr.append(npimage[y, x])
    nparr = np.array(arr, dtype="f")
    nparr = np.matrix.transpose(nparr)
    return nparr, width, height


def normalize_matrix(nparr):
    return nparr - nparr.mean()


def calc_covariance_element(x, y):
    mean_x, mean_y = x.mean(), y.mean()
    n = len(x)
    return sum((x - mean_x) * (y - mean_y)) / n


def covariance_matrix(data):
    rows, cols = data.shape
    cov_mat = np.zeros((cols, cols))
    for i in range(cols):
        for j in range(cols):
            cov_mat[i][j] = calc_covariance_element(data[:, i], data[:, j])

    return cov_mat


def w_matrix(matrix):
    eigvalues, eigvectors = np.linalg.eigh(matrix)
    eig_show_data(eigvalues, eigvectors)
    w1 = np.zeros([3, 1])
    w2 = np.zeros([3, 1])
    if np.argmin(eigvalues) == 0:
        if eigvalues[1] >= eigvalues[2]:
            w1 = eigvectors[1]
            w2 = eigvectors[2]
        else:
            w1 = eigvectors[2]
            w2 = eigvectors[1]
    elif np.argmin(eigvalues) == 1:
        if eigvalues[0] >= eigvalues[2]:
            w1 = eigvectors[0]
            w2 = eigvectors[2]
        else:
            w1 = eigvectors[2]
            w2 = eigvectors[0]
    else:
        if eigvalues[0] >= eigvalues[1]:
            w1 = eigvectors[0]
            w2 = eigvectors[1]
        else:
            w1 = eigvectors[1]
            w2 = eigvectors[0]

    w = np.zeros((3, 2))
    for i in range(3):
        w[i, 0] = w1[i]
    for i in range(3):
        w[i, 1] = w2[i]
    return w


def output_image(data, img_height, img_width):
    image_size = (img_height - 1) * (img_width - 1)
    zero_matrix = np.zeros([image_size, 1])
    hstack_matrix = np.hstack((data, zero_matrix))
    new_image = Image.fromarray(np.reshape(hstack_matrix, (img_height - 1, img_width - 1, 3)).astype('uint8'))
    new_image.save("out.jpg")


image_address = input("Enter address of the image file:")
nparr, width, height = image_array(image_address)
show_data(np.matrix.transpose(nparr))
nparrcov = nparr
nparr = normalize_matrix(nparr)

"""
Calculate Covariance Matrix
"""
covariance = covariance_matrix(np.matrix.transpose(nparrcov))

print("Covariance Matrix:")
print(covariance)

"""
Calculate W Matrix (Principal Components)
"""
w = w_matrix(covariance)
print("W Matrix:")
print(w)

"""
Calculate final reduced data set
"""
reduce_data_matrix = np.matrix.transpose(np.dot(np.matrix.transpose(w), nparr))
print("Reduced Data With PCA:")
print(reduce_data_matrix)

"""
Convert data to image
"""
output_image(reduce_data_matrix, height, width)
