from facenet_pytorch import MTCNN
import os
from PIL import Image

from shutil import copyfile
from collections import Counter

from tqdm import tqdm
import cv2
import numpy as np
from skimage import transform as trans
from pathlib import Path
import torch

from matplotlib import cm

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=112, device=device)

image_path = os.path.join(os.getcwd(), "experiments", "July", "06-07-2022_22-36-26", "final_results", "final_patch.png")


image = cv2.imread(image_path)

im = Image.fromarray(cv2.imread(image_path))

im.show()

out = mtcnn.detect(image)

print(out)
