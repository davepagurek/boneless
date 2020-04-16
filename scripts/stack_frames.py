from PIL import Image
import numpy as np
import skvideo.io
from skimage.util import img_as_float, img_as_ubyte
import sys
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str, default='out.png')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--interval', type=int, default=70)
parser.add_argument('--ymin', type=int, default=-100)
parser.add_argument('--ymax', type=int, default=-10)
args = parser.parse_args()

videodata = img_as_float(skvideo.io.vread(args.input))
videodata = videodata[args.start:, :, :, :]
frames, h, w, channels = videodata.shape

single_h = args.ymax - args.ymin
num_stacked = math.floor(frames / args.interval)

coalesced = np.zeros((num_stacked*single_h, w, channels))
for i in range(num_stacked):
    coalesced[single_h*i:single_h*(i+1),:,:] = \
            videodata[i*args.interval,args.ymin:args.ymax,:,:]

img = Image.fromarray(img_as_ubyte(coalesced), 'RGB')
img.save(args.output)
img.show()
