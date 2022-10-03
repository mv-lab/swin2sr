# -----------------------------------------------------------------------------------
# https://github.com/mv-lab/swin2sr
# -----------------------------------------------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_img (filename, debug=False, norm=True, resize=None):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if norm:   
        img = img / 255.
        img = img.astype(np.float32)
    if debug:
        print (img.shape, img.dtype, img.min(), img.max())
        
    if resize:
        img = cv2.resize(img, (resize[0], resize[1]))
        
    return img


def plot_all (images, axis='off', figsize=(16, 8)):
    
    fig = plt.figure(figsize=figsize, dpi=80)
    nplots = len(images)
    for i in range(nplots):
        plt.subplot(1,nplots,i+1)
        plt.axis(axis)
        plt.imshow(images[i])
    plt.show()
