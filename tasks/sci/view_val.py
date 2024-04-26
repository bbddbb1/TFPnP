from until import *
from scipy.io import loadmat
import os
from PIL import Image
import numpy as np
data = loadmat(os.path.join("data/val", "1.mat"))

gt = data['x0']

gt = shift_back_CHW(gt, 2)

gt = gt[1,:,:]

max = np.max(gt)
min = np.min(gt)
map = (gt - min) / (max - min) * 255;

img = Image.fromarray(map)


img.show()




