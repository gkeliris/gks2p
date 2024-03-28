# the way below then you would need to call mkops.mkops(...)
# from ops import mkops # <-- imports the whole file not the function inside

from ops.mkops import *
from ops.datasets import *
from gks2p_utils.preprocess import *
import suite2p
from natsort import natsorted

basepath = '/home/georgioskeliris/Desktop/gkel@NAS/MECP2TUN/'

stavroula_classifier='/mnt/12TB_HDD_6/ThinkmateB_HDD6_Data/Stavroula/Classifier_contrast.npy'
'''
ds = getDataPaths('coh2','w11','contrast')
ds = getOneExpPath('coh1','w11','M19','contrast')
ds = getOneSesPath('coh1', 'w11', 'M10', 'ses1', 'OO')
'''
       
ds = datasetQuery(cohort='coh1', week='w11', mouseID='M10', ses='ses1', experiment='OO')   
ds = datasetQuery(cohort='coh1', week='w11', mouseID='M20', experiment='contrast')

ds = datasetQuery(cohort='coh2', week='w11', experiment='contrast')
ds = datasetQuery(cohort='coh2', week='w11', experiment='contrast', mouseID='M24')
ds = datasetQuery(cohort='coh2', week='w11', experiment='contrast', mouseID='M73')
ds = datasetQuery(cohort='coh2', week='w11', experiment='contrast', mouseID='M77', ses='ses3')
ds = datasetQuery(cohort='coh2', week='w11', experiment='contrast', mouseID='M78')
ds = datasetQuery(cohort='coh2', week='w11', experiment='contrast', mouseID='M81')
ds = datasetQuery(cohort='coh2', week='w11', experiment='contrast', mouseID='M145')
ds = datasetQuery(cohort='coh2', week='w11', experiment='contrast', mouseID='M148')
ds = datasetQuery(cohort='coh2', week='w11', experiment='contrast', mouseID='M149', ses='ses3')
ds = datasetQuery(cohort='coh2', week='w11', experiment='contrast', mouseID='M156')
ds = datasetQuery(cohort='coh2', week='w11', experiment='contrast', mouseID='M159')

ds = datasetQuery(cohort='coh2', week='w22', experiment='contrast')
ds = datasetQuery(cohort='coh2', week='w22', experiment='contrast', mouseID='M24')
ds = datasetQuery(cohort='coh2', week='w22', experiment='contrast', mouseID='M145')
ds = datasetQuery(cohort='coh2', week='w22', experiment='contrast', mouseID='M148',ses='ses1')
ds = datasetQuery(cohort='coh2', week='w22', experiment='contrast', mouseID='M148',ses='ses2')
ds = datasetQuery(cohort='coh2', week='w22', experiment='contrast', mouseID='M148',ses='ses3')
ds = datasetQuery(cohort='coh2', week='w22', experiment='contrast', mouseID='M149')
ds = datasetQuery(cohort='coh2', week='w22', experiment='contrast', mouseID='M156', ses='ses1')
ds = datasetQuery(cohort='coh2', week='w22', experiment='contrast', mouseID='M159')


# ANALYSIS

# In case you need to change some of the default OPS parameters put them here:
db = {
    'pipeline': 'orig',
    'tau': 1.5,
    'spatial_scale': 0
}

# GET THE DATASET(S)


# Potentially check if corrupted
for d in range(0,len(ds)):
    dsetOK = checkDatasets(ds.iloc[d].rawPath)

# And make the ops from the .tif header (if not done)
gks2p_makeOps(ds, basepath, db={})


# this can load the ops instead of remaking it
# ops is a list of dictionaries
ops = gks2p_loadOps(ds, basepath)
        
# CONVERT TO BINARY
basepath = '/home/georgioskeliris/Desktop/gkel@NAS/MECP2TUN/'
gks2p_toBinary(ds, basepath)
gks2p_register(ds, basepath, iplaneList=None)
gks2p_segment(ds, basepath, iplaneList=[10, 11]) # potially can give a list of planes to process
gks2p_combine(ds, basepath)

gks2p_classify(ds, basepath)
gks2p_classify(ds, basepath, classfile=stavroula_classifier)
gks2p_deconvolve(ds,basepath,0.7) # run deconvolution with a different tau if necessary





# correct
gks2p_correctOpsPerPlane(ds, basepath)


stats = np.reshape(
    np.array([
        stat[j][k]
        for j in range(len(stat))
        for k in classifier['keys']
    ]), (len(stat), -1))
