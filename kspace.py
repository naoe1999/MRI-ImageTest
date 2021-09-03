import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn


def kspace_to_image(k, dim=None, img_shape=None):
    if not dim:
        dim = range(k.ndim)

    img = fftshift(ifftn(ifftshift(k, axes=dim), s=img_shape, axes=dim), axes=dim)
    img *= np.sqrt(np.prod(np.take(img.shape, dim)))
    return img


def image_to_kspace(img, dim=None, k_shape=None):
    if not dim:
        dim = range(img.ndim)

    k = fftshift(fftn(ifftshift(img, axes=dim), s=k_shape, axes=dim), axes=dim)
    k /= np.sqrt(np.prod(np.take(img.shape, dim)))
    return k


if __name__ == '__main__':
    img = cv2.imread('img/transform_src.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img.shape)

    kspace = image_to_kspace(img)
    print(kspace.shape)

    img_rec = kspace_to_image(kspace)
    print(img.shape)

    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

    plt.imshow(np.log(np.abs(kspace)), cmap=plt.cm.gray)
    # plt.imshow(np.abs(kspace), cmap=plt.cm.gray)
    plt.show()

    plt.imshow(abs(img_rec), cmap=plt.cm.gray)
    plt.show()

