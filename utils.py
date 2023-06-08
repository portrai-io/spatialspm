import math
import numpy as np

def shape_matching(im, size):
    sz=im.shape
    if sz[0] >size[0]:
        xstart=int(math.ceil(sz[0]/2)- int(size[0]/2-1))
        im=im[xstart-1:xstart+size[0]-1,:]
    else:
        npad1=int(math.floor(float(size[0])/2-float(sz[0])/2))
        npad2=int(math.ceil(float(size[0])/2-float(sz[0])/2))
        im=np.lib.pad(im,((npad1,npad2),
                          (0,0)),'constant')
    if sz[1] >size[1]:
        ystart=int(math.ceil(sz[1]/2)- int(size[1]/2-1))
        im=im[:,ystart-1:ystart+size[1]-1]
    else:
        npad1=int(math.floor(float(size[1])/2-float(sz[1])/2))
        npad2=int(math.ceil(float(size[1])/2-float(sz[1])/2))
        im=np.lib.pad(im,((0,0), (npad1,npad2)),'constant')
    return im