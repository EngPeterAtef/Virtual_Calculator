import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from skimage.color import rgb2gray
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, skeletonize, thin


def show_images(images, titles=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


path = 'E:/Third Year/First Term/Image Processing/Project/Virtual_Keyboard/Project testing/'
closed_hand = rgb2gray(io.imread(path+'hand_closed.png')[:, :, :3]) > 0.4
open_hand = rgb2gray(io.imread(path+'hand.jpg')) > 0.4
closed = rgb2gray(io.imread(path+'closed_hand.jpg'))
open_hand = rgb2gray(io.imread(path+'open_hand.jpg'))

img1 = skeletonize(closed)
img2 = skeletonize(open_hand)


show_images([closed, open_hand, img1, img2])
