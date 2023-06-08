from imagizer_cv import spatial_imagizer, get_spotdist, get_img_coord
from registration import registration_spatial
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import pandas as pd
import cv2
import scanorama
from utils import shape_matching
from scipy.ndimage import gaussian_filter
from scipy import stats


def spatial_normalization(tissue_dirs,
                            sel_no = None,
                            tsimg_output= (600,600),
                            targetsize = (128,128),
                            truncate = 3,
                            alpha= 0.01,
                            idx_ref=  0,
                            verbose =0 ):
    '''
    tissue_dirs: Visium folders; read by scanpy.read_visium
    sel_no : To select HVGs commonly found in tissue_dirs
    tsimg_output: size of imagizer size to be squared. default: (600,600)
    targetsize : bilinear interpolation to reshaped image size. default(128,128)
    truncate: imagizer parameter. Filter size to be considered for range of gaussian distribution
               range of distribution = spotdist * truncate
    alpha: imagizer parameter. (1-alpha) is a probability of distribution of counts for each spot in
         (range of distribution)
    idx_ref: index of reference for registration. a reference image shoud be selected. default = 0.
    '''
    if sel_no is None:
        sel_no = len(tissue_dirs)
        
    adatas=[]
    for tissue_dir in tissue_dirs:
        adata1 = sc.read_visium(tissue_dir)
        adata1.var_names_make_unique()
        sc.pp.normalize_total(adata1, inplace=True)
        sc.pp.log1p(adata1)
        sc.pp.highly_variable_genes(adata1, flavor="seurat", n_top_genes=2000)
        adatas.append(adata1)

    #List of HVG
    hvg_list = []
    for adata in adatas:
        hvg_list.extend(adata.var.index[adata.var.highly_variable].tolist())

    hvg_genes_all= list(set(hvg_list))
    hvg_sel =[]

    for ii in hvg_genes_all:
        c= hvg_list.count(ii)
        if c >= sel_no:
            hvg_sel.append(ii)
    print('Number of HVGs:', len(hvg_sel))

    for adata in adatas:
        sc.tl.score_genes(adata, hvg_sel, score_name = 'hvgall')
    #sc.pl.spatial(adatas[0], color = 'hvgall',img_key = 'lowres', size=2.0)

#   ## Load Data for Imagizer
    ts_meta_coords, tsimgs, imscales = [],[],[]
    metafile = None
    for tissue_dir in tissue_dirs:
        ts_meta_coord1,tsimg1, imscale1 = get_img_coord(tissue_dir,
                                                        metafile,
                                                        barcode_tag = '',
                                                        hires=0)
        ts_meta_coords.append(ts_meta_coord1)
        tsimgs.append(tsimg1)
        imscales.append(imscale1)

    spotdists=[]
    for ts_meta_coord1, imscale1 in zip(ts_meta_coords, imscales):
        spotdists.append(get_spotdist(ts_meta_coord1,imscale1))

    ts_meta_coords_ = []
    for adata1, ts_meta_coord1 in zip(adatas, ts_meta_coords):
        ts_meta_coords_.append(pd.merge(adata1.obs,ts_meta_coord1,how = 'inner',
                         right_on ='barcodes' , left_on=adata1.obs.index  ))

    #Generate HVG images by imagizer
    im_alls= []
    for ii in range(len(adatas)):
        im_alls.append(spatial_imagizer(adatas[ii].obs.hvgall - np.min(adatas[ii].obs.hvgall),
                                       tsimgs[ii],
                                       ts_meta_coords_[ii],
                                       imscales[ii],
                                       radius=spotdists[ii]*truncate,
                                       alpha=alpha))

    # Change to shape_matching ---> (600,600)
    fignum = len(im_alls)
                       
    for ii in range(fignum):
        im_alls[ii] = shape_matching(im_alls[ii],tsimg_output)
    #Resize
    for ii in range(fignum):
        im_alls[ii] = cv2.resize(im_alls[ii], targetsize)

    if verbose:
        plt.figure(figsize=(12,4))
        for ii in range(fignum):
            plt.subplot(1,fignum,ii+1)
            plt.imshow(im_alls[ii], cmap='jet')
            plt.title(im_alls[ii].shape)


    ## Spatial Registration
    im_alls_s = []
    mapping_all = []
    for e, img in enumerate(im_alls): #idx_ref ==> index for reference image
        iimg1, mapping1 = registration_spatial(im_alls[idx_ref], img)
        im_alls_s.append(iimg1)
        mapping_all.append(mapping1)

    if verbose:
        plt.figure(figsize=(12,4))
        for ii in range(fignum):
            plt.subplot(1,fignum,ii+1)
            plt.imshow(im_alls_s[ii], cmap='jet')
            plt.title(im_alls_s[ii].shape)

    return im_alls_s, adatas, tsimgs, ts_meta_coords_, imscales, spotdists, mapping_all


def mol2imgs(adatas, tsimgs,ts_meta_coords_,imscales,spotdists,
             mapping_all,feature,
             truncate = 3, alpha=0.01,
             sigma=4, tsimg_output=(600,600), targetsize=(128,128),
             visualize = 0):
    '''
    sigma: cv2 smoothing parameter
    '''
    molimgs=[]
    for ii in range(len(adatas)):
        ad1_1 = spatial_imagizer(np.asarray(adatas[ii][:,feature].X.todense()),
                                   tsimgs[ii],
                                   ts_meta_coords_[ii],
                                   imscales[ii],
                                   radius=spotdists[ii]*truncate,
                                   alpha=alpha)
        #if ii in inverse_idx:
        #    ad1_1 = np.fliplr(ad1_1)
        ad1_1 = shape_matching(ad1_1, tsimg_output) #To ... Tsimg output
        ad1_1 = cv2.resize(ad1_1, targetsize)  #To Resize

        ad1_1= mapping_all[ii].transform(ad1_1) #Saved Mapping
        ad1_1 = gaussian_filter(ad1_1, sigma)
        molimgs.append(ad1_1)

    if visualize:
        plt.figure(figsize=(18,6))
        for ii in range(len(molimgs)):
            plt.subplot(1,len(molimgs),ii+1)
            plt.imshow(molimgs[ii],cmap='magma',  vmax=np.max(np.stack(molimgs)))
            #plt.title(titles[ii])
            plt.axis('off')
            plt.colorbar(fraction=0.015, orientation="horizontal")
    return molimgs

def get_corr_map(imgs_mol, variables, anatomymap=None, threshold = 0.05,
                method='pearson', visualize=1, verbose = 1):
    '''
    variables: variables for correlation. len(variables) = len(imgs_mol)
    threshold: for generating mask.
    '''
    imgs_mol_sum = np.sum(np.stack(imgs_mol,axis=-1),axis=-1)
    imgs_mol_thr = imgs_mol_sum>0.05
    #plt.imshow(imgs_mol_thr)
    imgs_mol_idx = np.where(imgs_mol_thr)

    imgs_rmap = np.zeros(imgs_mol_sum.shape)
    imgs_pmap = np.zeros(imgs_mol_sum.shape)
    for k in range(len(imgs_mol_idx[0])):
        idxx= imgs_mol_idx[0][k]
        idxy= imgs_mol_idx[1][k]
        if method == 'pearson':
            rhoval, pval = stats.pearsonr([img[idxx,idxy] for img in imgs_mol],variables)
        elif method =='spearman':
            rhoval, pval = stats.spearmanr([img[idxx,idxy] for img in imgs_mol],variables)
        else:
            raise ValueError
        imgs_rmap[idxx,idxy] = rhoval
        imgs_pmap[idxx,idxy] = pval
        if verbose:
            if k%600==0: print("Pixel-wise comparison done for ... ", k)
    if anatomymap is None:
        anatomymap = imgs_mol[0]
    if visualize:
        plt.figure(figsize= (8,4))
        plt.subplot(1,2,1)
        plt.imshow(anatomymap,cmap='gray_r',alpha=0.5)
        plt.imshow(imgs_rmap, cmap='Reds', vmin=0.5, vmax=1,alpha=0.5)
        plt.axis('off')
        plt.colorbar(fraction=0.015, orientation="horizontal")
        plt.subplot(1,2,2)
        plt.imshow(anatomymap,cmap='gray_r',alpha=0.5)
        plt.imshow(imgs_rmap, cmap='Blues_r', vmin=-1, vmax=-0.5,alpha=0.5)
        plt.axis('off')
        plt.colorbar(fraction=0.015, orientation="horizontal")
    return imgs_rmap, imgs_pmap
