#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 19:08:18 2024

@author: georgioskeliris
"""
import numpy as np
import suite2p
from ops.mkops import *
from ops.datasets import *
from suite2p import registration
import sys
from pathlib import Path 
from natsort import natsorted
import shutil
import os

def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]

def gks2p_makeOps(ds, basepath, db={}):
    for d in range(0,len(ds)):
        print('\n\nPROCESSING:')
        print(ds.iloc[d])
        try:
            ops = mkops(basepath, ds.iloc[d], db)
        except Exception as error:
        # handle the exception
            print('\n****** -> PROBLEM WITH THIS DATASET ****\n')
            print("An exception occurred:", type(error).__name__, "-", error)
    return


def gks2p_loadOps(ds, basepath, pipeline="orig"):
    opsPath = []
    for d in range(len(ds)):
        dat = ds.iloc[d]  # Convert the pandas DataFrame to a pandas Series
        opsPath.append(os.path.join(basepath, 's2p_analysis', dat.cohort,
                                              dat.mouseID, dat.week, dat.session,
                                              dat.expID, 'ops_' + pipeline + '.npy'))
    ops = [np.load(f, allow_pickle=True).item() for f in opsPath]
    return ops

def gks2p_toBinary(ds, basepath):
    opsList = gks2p_loadOps(ds, basepath)
    for d in range(len(ds)):
        dat=ds.iloc[d] # Convert the pandas DataFrame to a pandas Series
        print(dat)
        ops = opsList[d]
        suite2p.run_s2p_toBinary(ops=ops)
        #suite2p.run_planes(ops=ops)
    return


def gks2p_register(ds, basepath, pipeline='orig', iplaneList=None):
    
    opsList = gks2p_loadOps(ds, basepath, pipeline)
    for d in range(len(ds)):
        dat=ds.iloc[d] # Convert the pandas DataFrame to a pandas Series
        print(dat)
        ops = opsList[d]
        
        if iplaneList is None:
            cur_iplaneList=[x for x in range(len(ops['dx']))]
        else:
            cur_iplaneList=iplaneList
        
        for iplane in cur_iplaneList:
            print("\nREGISTERING: plane" + str(iplane))
            opsstr= os.path.join(ops['save_path0'],'suite2p_' + pipeline,'plane' + str(iplane), 'ops.npy')
            opsPlane=np.load(opsstr,allow_pickle=True).item()
            Ly=opsPlane['Ly']
            Lx=opsPlane['Lx']
            
            f1 = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx,
                            filename=os.path.join(ops['fast_disk'],'suite2p',
                            'plane' + str(iplane), 'data_raw.bin'))
            n_frames = f1.shape[0]
            f1_reg = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx,
                            filename=os.path.join(ops['fast_disk'],'suite2p',
                            'plane' + str(iplane), 'data.bin'), n_frames = n_frames)
        
            registration_outputs = suite2p.registration_wrapper(f1_reg, f_raw=f1, f_reg_chan2=None, 
                                                               f_raw_chan2=None, refImg=None, 
                                                               align_by_chan2=False, ops=opsPlane)
            
            suite2p.registration.register.save_registration_outputs_to_ops(registration_outputs, opsPlane)
            # add enhanced mean image
            meanImgE = suite2p.registration.compute_enhanced_mean_image(
                            opsPlane["meanImg"].astype(np.float32), opsPlane)
            opsPlane["meanImgE"] = meanImgE
            np.save(opsstr,opsPlane)
            
            if opsPlane["two_step_registration"] and opsPlane["keep_movie_raw"]:
                print("----------- REGISTRATION STEP 2")
                print("(making mean image (excluding bad frames)")
                nsamps = min(n_frames, 1000)
                inds = np.linspace(0, n_frames, 1 + nsamps).astype(np.int64)[:-1]

                refImg = f1_reg[inds].astype(np.float32).mean(axis=0)
                registration_outputs = suite2p.registration_wrapper(
                    f1_reg, f_raw=None, f_reg_chan2=None, f_raw_chan2=None,
                    refImg=refImg, align_by_chan2=False, ops=opsPlane)
                np.save(opsstr,opsPlane)
            
            # compute metrics for registration
            if ops.get("do_regmetrics", True) and n_frames >= 1500:
                
                # n frames to pick from full movie
                nsamp = min(2000 if n_frames < 5000 or Ly > 700 or Lx > 700 else 5000,
                            n_frames)
                inds = np.linspace(0, n_frames - 1, nsamp).astype("int")
                mov = f1_reg[inds]
                mov = mov[:, opsPlane["yrange"][0]:opsPlane["yrange"][-1],
                          opsPlane["xrange"][0]:opsPlane["xrange"][-1]]
                opsPlane = suite2p.registration.get_pc_metrics(mov, opsPlane)
                np.save(opsstr,opsPlane)
    return

def gks2p_segment(ds, basepath, pipeline='orig', iplaneList=None):

    opsList = gks2p_loadOps(ds, basepath, pipeline)    
    for d in range(len(ds)):
        dat=ds.iloc[d] # Convert the pandas DataFrame to a pandas Series
        print(dat)
        ops = opsList[d]

        if iplaneList is None:
            cur_iplaneList=[x for x in range(len(ops['dx']))]
        else:
            cur_iplaneList=iplaneList
        
       # original = sys.stdout
       # sys.stdout = open(os.path.join(ops['save_path0'], ops['save_folder'], "run.log"), "a")
        
        for iplane in cur_iplaneList:
            print("\nSEGMENTING: plane" + str(iplane))
            pathstr= os.path.join(ops['save_path0'],
                                  'suite2p_' + pipeline,'plane' + str(iplane))
            opsstr=os.path.join(pathstr,'ops.npy')
            opsPlane=np.load(opsstr,allow_pickle=True).item()
            Ly=opsPlane['Ly']
            Lx=opsPlane['Lx']

            # Use default classification file provided by suite2p 
            classfile = suite2p.classification.builtin_classfile
            #np.load(classfile, allow_pickle=True)[()]
            
            f_reg = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, 
                    filename=os.path.join(opsPlane['fast_disk'],'suite2p', 
                                      'plane' + str(iplane), 'data.bin'))
            
            opsPlane, stat = suite2p.detection_wrapper(f_reg=f_reg, 
                                            ops=opsPlane, classfile=classfile)
            
            np.save(opsstr,opsPlane)
            np.save(os.path.join(pathstr,'stat.npy'),stat)
                
            # Fluorescence Extraction
            stat_after_extraction, F, Fneu, F_chan2, Fneu_chan2 = \
                suite2p.extraction_wrapper(stat, f_reg, f_reg_chan2 = None,
                                           ops=opsPlane)
            
            np.save(os.path.join(pathstr,'stat.npy'),stat_after_extraction)
            np.save(os.path.join(pathstr,'F.npy'),F)
            np.save(os.path.join(pathstr,'Fneu.npy'),Fneu)
            
            
            # Cell Classification
            iscell = suite2p.classify(stat=stat_after_extraction, classfile=classfile)
            np.save(os.path.join(pathstr,'iscell.npy'),iscell)
        
            # Spike Deconvolution
            # Correct our fluorescence traces 
            dF = F.copy() - opsPlane['neucoeff']*Fneu
            # Apply preprocessing step for deconvolution
            dF = suite2p.extraction.preprocess(
                    F=dF,
                    baseline=opsPlane['baseline'],
                    win_baseline=opsPlane['win_baseline'],
                    sig_baseline=opsPlane['sig_baseline'],
                    fs=opsPlane['fs'],
                    prctile_baseline=opsPlane['prctile_baseline']
                )
            # Identify spikes
            spks = suite2p.extraction.oasis(F=dF, batch_size=opsPlane['batch_size'], 
                                            tau=opsPlane['tau'], fs=opsPlane['fs'])
            np.save(os.path.join(pathstr,'spks.npy'),spks)
    
    #sys.stdout =  original
    return

def gks2p_classify(ds, basepath, pipeline="orig", iplaneList=None, classfile=None):
    
    if classfile is None:
        classfile = suite2p.classification.builtin_classfile
    np.load(classfile, allow_pickle=True)[()]
    
    opsList = gks2p_loadOps(ds, basepath, pipeline)
    for d in range(len(ds)):
        dat=ds.iloc[d] # Convert the pandas DataFrame to a pandas Series
        print(dat)
        ops = opsList[d]
        
        if iplaneList is None:
            iplaneList=[x for x in range(len(ops['dx']))]
            
        for iplane in iplaneList:
            print("\nCLASSIFYING: plane" + str(iplane))
            pathstr= os.path.join(ops['save_path0'],
                                  'suite2p_' + pipeline,'plane' + str(iplane))
            stat_after_extraction = np.load(os.path.join(pathstr,'stat.npy'), allow_pickle=True)
            iscell = suite2p.classify(stat=stat_after_extraction, classfile=classfile)
            np.save(os.path.join(pathstr,'iscell.npy'),iscell)
    return

def gks2p_deconvolve(ds, basepath, tau, pipeline="orig", iplaneList=None):
    
    opsList = gks2p_loadOps(ds, basepath, pipeline)
    for d in range(len(ds)):
        dat=ds.iloc[d] # Convert the pandas DataFrame to a pandas Series
        print(dat)
        ops = opsList[d]
        ops['tau_deconvolution'] = tau
        np.save(os.path.join(ops['save_path0'], 'ops_' + ops['pipeline']),ops)
        if iplaneList is None:
            iplaneList=[x for x in range(len(ops['dx']))]
            
        for iplane in iplaneList:
            print("\nDECONVOLVING: plane" + str(iplane))
            pathstr= os.path.join(ops['save_path0'],
                                  'suite2p_' + pipeline,'plane' + str(iplane))
            opsstr=os.path.join(pathstr,'ops.npy')
            opsPlane=np.load(opsstr,allow_pickle=True).item()
            opsPlane['tau_deconvolution'] = tau
            np.save(opsstr,opsPlane)
            
            F = np.load(os.path.join(pathstr,'F.npy'))
            Fneu = np.load(os.path.join(pathstr,'Fneu.npy'))
            
            # Spike Deconvolution
            # Correct our fluorescence traces 
            dF = F.copy() - opsPlane['neucoeff']*Fneu
            # Apply preprocessing step for deconvolution
            dF = suite2p.extraction.preprocess(
                    F=dF,
                    baseline=opsPlane['baseline'],
                    win_baseline=opsPlane['win_baseline'],
                    sig_baseline=opsPlane['sig_baseline'],
                    fs=opsPlane['fs'],
                    prctile_baseline=opsPlane['prctile_baseline']
                )
            # Identify spikes
            spks = suite2p.extraction.oasis(F=dF, batch_size=opsPlane['batch_size'], 
                                            tau=tau, fs=opsPlane['fs'])
            np.save(os.path.join(pathstr,'spks.npy'),spks)
        
        print('\nCombining planes...\n')
        out = suite2p.io.combined(os.path.join(ops['save_path0'],ops['save_folder']), save=True)
    return
            
def gks2p_combine(ds, basepath, pipeline="orig"):
    opsList = gks2p_loadOps(ds, basepath)
    for d in range(len(ds)):
        ops = opsList[d]
        out = suite2p.io.combined(os.path.join(ops['save_path0'],ops['save_folder']), save=True)
    return

'''
def gks2p_opsPerPlane(ds, basepath, pipeline="orig", iplaneList=None):
    
    for d in range(len(ds)):
        dat=ds.iloc[d] # Convert the pandas DataFrame to a pandas Series
        print(dat)
        opsPath = os.path.join(basepath, 's2p_analysis', dat.cohort,
                               dat.mouseID, dat.week, dat.session,
                               dat.expID, 'ops_' + pipeline + '.npy')
        ops = np.load(opsPath, allow_pickle=True).item() # Load ops as a dict
        
        if iplaneList is None:
            iplaneList=[x for x in range(len(ops['dx']))]
        
        
        for iplane in iplaneList:
            pathstr= os.path.join(ops['save_path0'],
                                  'suite2p_' + pipeline,'plane' + str(iplane))
            opsstr=os.path.join(pathstr,'ops.npy')
            opsPlane=np.load(opsstr,allow_pickle=True).item()
    
        ops1=suite2p.io.utils.init_ops(ops)
'''
        
        
def gks2p_loadOpsPerPlane(save_folder):
    plane_folders = natsorted([ f.path for f in os.scandir(save_folder) if f.is_dir() and f.name[:5]=='plane'])
    ops1 = [np.load(os.path.join(f, 'ops.npy'), allow_pickle=True).item() for f in plane_folders]
    
    return ops1

def gks2p_correctOpsPerPlane(ds, basepath):
    for d in range(len(ds)):
        ops = mkops(basepath, ds.iloc[d])
        head, n = os.path.split(ops['save_path0'])
        src = os.path.join(head, 'pipeline1', n, 'suite2p')
        dst = os.path.join(head, n, 'suite2p_orig')
        shutil.move(src, dst)
        ops1 = gks2p_loadOpsPerPlane(os.path.join(ops['save_path0'],ops['save_folder']))
        for i in range(len(ops1)):
            pathstr= os.path.join(ops['save_path0'],
                                  'suite2p_orig','plane' + str(i))
            ops1[i]['save_path0']=ops['save_path0']
            ops1[i]['save_folder']=ops['save_folder']
            ops1[i]['save_path']=pathstr
            np.save(os.path.join(ops1[i]['save_path'],'ops.npy'),ops1[i])
    return


# To correct some datasets that were directing to a different fastdisk
'''
ops = gks2p_loadOps(ds, basepath)
ops1=gks2p_loadOpsPerPlane(os.path.join(ops[0]['save_path0'],ops[0]['save_folder']))
for i in range(len(ops1)):
    #ops1[i]['fast_disk'] = os.path.join(basepath, 's2p_binaries', dat.cohort, 
    #                    dat.mouseID, dat.week, dat.session, dat.expID)
    ops1[i]['spatial_scale']=2

    np.save(os.path.join(ops1[i]['save_path'],'ops.npy'),ops1[i])
'''
    
    
