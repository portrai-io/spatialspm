import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.filters import gaussian
from scipy.ndimage import rotate
import math

from dipy.data import get_fnames
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric, CCMetric, EMMetric
import dipy.align.imwarp as imwarp

from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform2D,
                                   RigidTransform2D,
                                   AffineTransform2D)
from dipy.viz import regtools


def mutual_information(ximg, yimg, nbins):
    hgram, x_edges, y_edges = np.histogram2d(
             ximg.ravel(),
             yimg.ravel(),
             bins=nbins)
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
     # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def registration_spatial(imgref, imgmove):
    #Params
    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
    level_iters = [100, 50, 25]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]

    #COM
    identity = np.eye(3)
    c_of_mass = transform_centers_of_mass(imgref, identity,
                                          imgmove, identity)

    transformed = c_of_mass.transform(imgmove)
    #regtools.overlay_images(imgref, transformed, 'Static', 'Overlay', 'Moving')
    print('COM matching Done')

    #Transform
    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors,
                                verbosity = 0)

    transform = TranslationTransform2D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(imgref, imgmove, transform, params0,
                                  identity, identity,
                                  starting_affine=starting_affine)
    transformed = translation.transform(imgmove)
    #regtools.overlay_images(imgref, transformed, 'Static', 'Overlay', 'Moving')
    
    #Rigid
    transform = RigidTransform2D()
    rigid = affreg.optimize(imgref, imgmove, transform, params0,
                        identity, identity,
                         starting_affine=translation.affine)
    transformed = rigid.transform(imgmove)
    #regtools.overlay_images(imgref, transformed, 'Static', 'Overlay', 'Moving')
    print(mutual_information(imgref, transformed, nbins=nbins))
    print('Rigid Transform Done')

    #Affine Transform 
    transform = AffineTransform2D()
    #affreg.level_iters = [100, 50, 25]
    affine = affreg.optimize(imgref, imgmove, transform, params0,
                        identity, identity,
                        starting_affine=rigid.affine)
    transformed = affine.transform(imgmove)
    regtools.overlay_images(imgref, transformed, 'Static', 'Overlay', 'Moving')
    print(mutual_information(imgref, transformed, nbins=nbins))
    print('Affine Registration Done')
    
    
    #Nonlinear Transform
    sigma_diff = 6.0
    radius = 8
    metric = CCMetric(2, sigma_diff, radius)

    level_iters = [40, 20, 10]

    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters, inv_iter=100)
    def callback_CC(sdr, status):
        # Status indicates at which stage of the optimization we currently are
        # For now, we will only react at the end of each resolution of the scale
        # space
        if status == imwarp.RegistrationStages.SCALE_END:
            # get the current images from the metric
            wmoving = sdr.metric.moving_image
            wstatic = sdr.metric.static_image
            # draw the images on top of each other with different colors
            #regtools.overlay_images(wmoving, wstatic, 'Warped moving', 'Overlay',
            #                        'Warped static')
    sdr.callback = callback_CC

    mapping = sdr.optimize(imgref, imgmove,  identity, identity, affine.affine)
    warped = mapping.transform( imgmove)

    regtools.plot_2d_diffeomorphic_map(mapping, 10)
    regtools.overlay_images(imgref, warped, 'Static', 'Overlay', 'Moving')
    return warped, mapping


