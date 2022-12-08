##########################################################################
# Analysis written for Jupyter Notebook 6.1.4
#
# Start the tube_filter.ipynb via the Jupyter Notebook, tube_filter.py
# has to be placed in the same folder.
#
# 1. set folder path (r"XXX")
# 2. set basefilename (r'\XXX')
# 3. define varaiable parameters for CLAHE and the Gaussian filter
#
#	#CLAHE
#		ntiles_x = XXX (eg 32)
#		ntiles_y = XXX (eg 32)
#		cliplimit=XXX (eg 0.01)
#
#	#Gaussian filter
#		Sigma=XXX (eg 0.2)
#
# 4. run skript
#
# 5. Analysis saves a skeleton.png and a node.txt (containing x an y
#    coordinates of the respective nodes)
#
# Version 0.1 (2022/12/07)
#
##########################################################################

import sys
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx

import scipy as sp
import scipy.ndimage
import scipy.misc
import scipy.spatial.distance
import scipy.signal

import skimage.color
import skimage.exposure
import skimage.feature
import skimage.filters
from skimage.util.shape import view_as_windows
from skimage.morphology import skeletonize

import scipy.signal
import sklearn.decomposition
import sklearn.linear_model

##########################################################################

def showstereo(image1,image2,pixelintensities=np.zeros((1,1,1)),
               left_title="",right_title="",show=True,
               outfile=None,outpath="."):
    """
    This receives two images and displays them with shared
        axis (x and y). The images must have the same size in
        dimension 1 and 2 (3 may be different to enable gray AND
        rgb output). An additional array holds values to be shown
    in a histogram for each point (x,y) if the user right
        clicks on a point. Shape of this special array is
        (values,x,y).

    PARAMETERS:

        image1: image that will be shown left
        image2: image that will be shown right
        pixelintensities: data shown at right-click event
        left_title: title of the left image
        right_title: title of the right image
        show: True means result will be shown on screen
        outfile: if other than None will save result as
                         file with this name
        outpath: path where the file is stored at
                         (default is . )

    AUTHOR: nd
    """
    if image1.shape[0] != image2.shape[0] \
                    or image1.shape[1] != image2.shape[1]:
        print("Images do not have the same shape:")
        print(image1.shape, 'and', image2.shape)
        return
   
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,
                                           sharex=True)
    ax1.imshow(image1, interpolation="none",cmap = "gray")
    ax1.set_title(left_title)
    ax2.imshow(image2, interpolation="none",cmap = "gray")
    ax2.set_title(right_title)
    plt.subplots_adjust(wspace=0.03, left=0.05, right=0.96)
    f.set_size_inches(10.8,6.5,forward=True)
        # fits to my screen, modify if needed
    if outfile is not None:
        print("saving stereo image as "+str(outfile))
        print("at "+str(outpath))
        plt.savefig(os.path.join(outpath,outfile))
    if show == True:
        plt.show()

    return



##########################################################################

def tube_filter(imO,sigma=2.0,mode='reflect'):
    """Apply tubeness filter to image.
    Parameters
    ----------
    imO : original two-dimensional image
    sigma : width parameter of tube-like structures
        
    Returns
    -------
    imT : filtered and rescaled image
    """
    imH=skimage.feature.hessian_matrix(imO,sigma=sigma,mode=mode)
   
    imM=skimage.feature.hessian_matrix_eigvals(imH)
    imR=-1.0*imM[1]
   
    imT=(imR-imR.min())/(imR.max()-imR.min())
    return imT
 
##########################################################################

def removeOverlap(maxima, verbose=False):
    """
    In a list of points, this method removes all the entries for which the
    corresponding points would have overlapping circles with their value as
    radius. Only the point (of these) with the highest value will be kept.
    PARAMETERS:
        maxima: array of the shape (samples,data) while data has total three
            fields for value, xpos and ypos
    AUTHOR: nd
    """
    ctr = maxima.shape[0]
    if verbose:
        print("number of nodes = "+str(ctr)+" (before removing overlap)")
    if ctr > 180000: #default 18000 --> changed by nils
        print("error: too many nodes found - too much noise")
        print("(>12GB for storing neccessary)") # 10 for radmat1+radmat2+sumrad
        print("aborting...")
        return
    maxima = maxima[maxima[:,0].argsort()][::-1]
    pos = maxima[:,(1,2)]
    rad = maxima[:,0]
    dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(pos))
    radmat1 = np.zeros((ctr,ctr))
    radmat2 = np.zeros((ctr,ctr))
    radmat1 = np.array([rad*1.0,]*ctr)
    radmat2 = np.transpose(radmat1*1.0)
    sumrad = radmat1 + radmat2
    del radmat1
    del radmat2
    overlap = (dist - sumrad) < 0
    for i in range(ctr):
        if maxima[i,0] == 0:
            continue
        for j in range(i+1,ctr):
            if overlap[i,j]:
                overlap[j,:]=False
                overlap[:,j]=False
                maxima[j,0]=0
    maxima = maxima[maxima[:,0]!=0]
    ctr = maxima.shape[0]
    if verbose:
        print("number of nodes = "+str(ctr)+" (after removing overlap)")
    return maxima

##########################################################################

def getNodes(skeleton, minimal_distance=1, return_crossmap=False):
    """
    This method returns all intersection points of a skeleton.
    PARAMETERS:
        skeleton: binary image to find intersections in
        minimal_distance: minimum required distance between two intersections
        return_overlay: if True it also returns binary image with 1 = cross
    AUTHOR: bg, (nd)
    """
    center = skeleton[1:-1,1:-1]
    sumimg = center * 0     # adding neighboring pixels to get their number
    sumimg += skeleton[0:-2,0:-2]
    sumimg += skeleton[1:-1,0:-2]
    sumimg += skeleton[2:,0:-2]
    sumimg += skeleton[0:-2,1:-1]
    sumimg += skeleton[2:,1:-1]
    sumimg += skeleton[0:-2,2:]
    sumimg += skeleton[1:-1,2:]
    sumimg += skeleton[2:,2:]

    crosses = center * (sumimg > 2)
    crosses = skeletonize(crosses)
    crossdata = np.zeros((int(np.sum(crosses)),3))
    pos = 0
    for i in range(crosses.shape[0]):
        for j in range(crosses.shape[1]):
            if crosses[i,j] == 1:
                crossdata[pos,0]=int(minimal_distance)
                crossdata[pos,1]=int(i)
                crossdata[pos,2]=int(j)
                pos += 1
    crossdata = removeOverlap(crossdata)
    if crossdata is None and return_crossmap:
        return (None,None)
    if crossdata is None and not return_crossmap:
        return None
    crosses = np.zeros(skeleton.shape)
    for i in range(crossdata.shape[0]):
        crosses[int(crossdata[i,1]),int(crossdata[i,2])]=1
    if return_crossmap:
        return crossdata[:,1:], crosses
    return crossdata[:,1:]

##########################################################################




