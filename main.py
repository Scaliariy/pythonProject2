import numpy as np
import cv2
from matplotlib import pyplot as plt


def read_this(image_file, gray_scale=False):
    image_src = cv2.imread(image_file)
    if gray_scale:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
    else:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
    return image_src


def binarize_lib(image_file, thresh_val=127, with_plot=False, gray_scale=False):
    image_src = read_this(image_file=image_file, gray_scale=gray_scale)

    # th, image_b = cv2.threshold(src=image_src, thresh=thresh_val, maxval=255, type=cv2.THRESH_BINARY)
    # image_array = image_b / 255
    image_array = np.around(np.mean(image_src, axis=1), 0)
    VD = image_array + 10
    ND = image_array - 10
    image_b = []
    for i, v in enumerate(image_src):
        if image_src[i] > 0:
            image_b[i] = 1
        else:
            image_b[i] = 0


    # for i, v in enumerate(image_array):
    #     if image_array[i] < 0.5:
    #         image_array[i] = 0
    #     else:
    #         image_array[i] = 1
    print(image_src)
    print(image_array)
    if with_plot:
        cmap_val = None if not gray_scale else 'gray'
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))

        # ax1.axis("off")
        ax1.title.set_text('Original')

        # ax2.axis("off")
        ax2.title.set_text("Binarized")

        ax1.imshow(image_src, cmap=cmap_val)
        ax2.imshow(image_b, cmap=cmap_val)
        return True
    return image_b


binarize_lib(image_file='pictures/1.bmp', with_plot=True, gray_scale=True)
# binarize_lib(image_file='pictures/2.bmp', with_plot=True, gray_scale=True)

plt.show()
