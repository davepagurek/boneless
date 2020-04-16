from PIL import Image
import numpy as np
import skvideo.io
from skimage.util import img_as_float, img_as_ubyte
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str, default='out.png')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--interval', type=int, default=70)
args = parser.parse_args()

videodata = img_as_float(skvideo.io.vread(args.input))
videodata = videodata[args.start:, :, :, :]
frames, h, w, channels = videodata.shape

thresholds = np.average(videodata, axis=-1)
ok = thresholds < 0.9
nok = thresholds >= 0.9
thresholds[ok] = 1
thresholds[nok] = 0


coalesced = videodata[0,:,:,:]
for i in range(1, frames):
    if i % args.interval == 0:
        mask = np.expand_dims(thresholds[i,:,:], axis=-1)
    else:
        mask = np.expand_dims(thresholds[i,:,:], axis=-1)*0.015
    coalesced = coalesced*(1 - mask) + videodata[i,:,:,:]*mask

img = Image.fromarray(img_as_ubyte(coalesced), 'RGB')
img.save(args.output)
img.show()
