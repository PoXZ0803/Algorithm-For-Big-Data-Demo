import numpy as np
from numpy import linalg
import sys
import matplotlib.pyplot as plt

def svd(A, tol=1e-5):
    # TODO: use linalg to calculate the matrix V
    #____________________________________________#
    
    sing_vals = np.sqrt(eigs)

    # TODO: sort the sigular values and the eigenvectors
    #____________________________________________#

    sing_vals_trunc = sing_vals[sing_vals>tol]
    V = V[:, sing_vals>tol]
    sigma = sing_vals_trunc

    # TODO: calculate U by using A and V
    #____________________________________________#

    return U.real, sigma.real, V.T.real

def truncate(U, S, V, k):
    # TODO: Select the top-k sigular and its corresbonding eigenvectors
    #____________________________________________#
    return U_trunc, S_trunc, V_trunc

img = plt.imread("./grayscale_image.png")[:,:,0]
print(img.shape)

U, S, V = svd(img)


# plot
k = 20
fig, ax = plt.subplots(1, 2, figsize=(30, 15))

plt.ion()
fig.canvas.draw()
U_trunc, S_trunc, Vt_trunc = truncate(U, S, V, k)

my_channel = 255 * U_trunc @ np.diag(S_trunc) @ Vt_trunc

ax[0].title.set_text(f"Original image")
ax[0].imshow(img, cmap='gray')
    
ax[1].title.set_text(f"Custom svd implementation, k={k}")
ax[1].imshow(my_channel, cmap='gray')

plt.show()
fig.savefig('./svd.png')