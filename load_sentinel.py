import glob
import numpy as np
from osgeo import gdal

# helpers for quick loading of S1 and S2 data

def normalization_parameters(fn):
    
    S = gdal.Open(fn)
    mns = []
    sds = []
    maxs = []
    
    for b in range(S.RasterCount):
        B = S.GetRasterBand(b+1)
        mn, sd = B.ComputeStatistics(1)[2:4]
        mns.append(mn)
        sds.append(sd)
        maxs.append(B.GetMaximum())
        
    return([mns, sds, maxs])

def sample_1s(filebase, batchsize, tilesize=128, normalize=False, flattened=False, fliprot=True):    
    alsos1 = False    
    if type(filebase) == str:
        multires = False
        S2_10 = gdal.Open(filebase)
        if normalize:
            maxima_10 = np.array(normalization_parameters(filebase)[2])            
            means_10 = np.array(normalization_parameters(filebase)[0])
            sds_10 = np.array(normalization_parameters(filebase)[1])
        
    elif type(filebase) == list:#interpreted as multiple resolutions

        tilesize = tilesize/2
        multires = True
        
        if len(filebase) == 3:
            alsos1 = True
            S1 = gdal.Open(filebase[0])
            S2_10 = gdal.Open(filebase[1])
            S2_20 = gdal.Open(filebase[2])            
            if normalize:
                maxima_S1 = np.array(normalization_parameters(filebase[0])[2])
                means_S1 = np.array(normalization_parameters(filebase[0])[0])
                sds_S1 = np.array(normalization_parameters(filebase[0])[1])
                maxima_10 = np.array(normalization_parameters(filebase[1])[2])
                means_10 = np.array(normalization_parameters(filebase[1])[0])
                sds_10 = np.array(normalization_parameters(filebase[1])[1])
                maxima_20 = np.array(normalization_parameters(filebase[2])[2])
                means_20 = np.array(normalization_parameters(filebase[2])[0])
                sds_20 = np.array(normalization_parameters(filebase[2])[1])
            
        else:
            S2_10 = gdal.Open(filebase[0])
            S2_20 = gdal.Open(filebase[1])            
            if normalize:
                maxima_10 = np.array(normalization_parameters(filebase[0])[2])
                maxima_20 = np.array(normalization_parameters(filebase[1])[2])                
                means_10 = np.array(normalization_parameters(filebase[0])[0])
                sds_10 = np.array(normalization_parameters(filebase[0])[1])                
                means_20 = np.array(normalization_parameters(filebase[1])[0])
                sds_20 = np.array(normalization_parameters(filebase[1])[1])  
                
    samples_S1 = []
    samples_10 = []
    
    if multires:        
        samples_20 = []        
        if fliprot:
            fac_for_aug = 8
        else:
            fac_for_aug = 1    
            
        while len(samples_10) < (batchsize * fac_for_aug):
            np.random.seed()
            RX = np.random.randint(S2_20.RasterXSize-tilesize,size=1)
            RY = np.random.randint(S2_20.RasterYSize-tilesize,size=1)
            
            A_10 = np.transpose(S2_10.ReadAsArray(RX[0] * 2, RY[0] * 2, tilesize * 2, tilesize * 2)).astype(np.float32)
            A_20 = np.transpose(S2_20.ReadAsArray(RX[0], RY[0], tilesize, tilesize)).astype(np.float32)
            S1unmasked = True
            
            if alsos1:
                A_S1 = np.transpose(S1.ReadAsArray(RX[0] * 2, RY[0] * 2, tilesize * 2, tilesize * 2)).astype(np.float32)
                if np.min(A_S1) > 0:
                    S1unmasked = True
                else:
                    S1unmasked = False
            
            if (np.min(A_10) > 0) & (np.min(A_20) > 0) & S1unmasked:
                if normalize == 1:
                    A_10 = A_10 / maxima_10
                    A_20 = A_20 / maxima_20
                    if alsos1:
                        A_S1 = A_S1 / maxima_S1
                elif normalize == 2:
                    A_10 = (A_10 - means_10) / sds_10
                    A_20 = (A_20 - means_20) / sds_20
                    if alsos1:
                        A_S1 = (A_S1 - means_S1) / sds_S1
                        
                if flattened:
                    A_10 = A_10.flatten()
                    A_20 = A_20.flatten()
                    if alsos1:
                        A_S1.flatten()
                        
                if fliprot:
                    for r in range(0,3):
                        samples_10.append(np.rot90(A_10, r))
                        samples_10.append(np.fliplr(np.rot90(A_10, r)))                        
                        samples_20.append(np.rot90(A_20, r))
                        samples_20.append(np.fliplr(np.rot90(A_20, r)))
                        if alsos1:
                            samples_S1.append(np.rot90(A_S1, r))
                            samples_S1.append(np.fliplr(np.rot90(A_S1, r)))
                else:
                    samples_10.append(A_10)
                    samples_20.append(A_20)
                    if alsos1:
                        samples_S1.append(A_S1)
                
        return([np.array(samples_10), np.array(samples_20), np.array(samples_S1)])
                
    else:
        while len(samples_10) < batchsize:
            RX = np.random.randint(S2_10.RasterXSize-tilesize,size=1)
            RY = np.random.randint(S2_10.RasterYSize-tilesize,size=1)
            
            A_10 = np.transpose(S2_10.ReadAsArray(RX[0], RY[0], tilesize, tilesize).astype(np.float32))
            
            if np.min(A_10) > 0:            
                if normalize == 1:
                    A_10 = A_10 / maxima_10
                elif normalize == 2:
                    A_10 = (A_10 - means_10) / sds_10
                if flattened:
                    A_10 = A_10.flatten()
                    
                if fliprot:
                    for r in range(0,3):
                        samples_10.append(np.rot90(A_10, r))
                        samples_10.append(np.fliplr(np.rot90(A_10, r)))
                else:
                    samples_10.append(A_10)
                
        return(np.array(samples_10))
    
def sample_some(n_per_scene=16*10*5, tilesize=32, flattened=False, fliprot=True):
    # S2 only
    X_all_10 = []
    X_all_20 = []
    #direc = '../data'
    direc = '/media/ramdisk'
    for s in ['A', 'B', 'C', 'D', 'E']:
        names = glob.glob(direc + '/S2*' + s + '*tif')
        names.sort()
        print names
        X_temp_10, X_temp_20, _ = sample_1s(names, n_per_scene, tilesize=tilesize, normalize=1, flattened=flattened, fliprot=fliprot)
        X_all_10 = X_all_10 + list(X_temp_10)
        X_all_20 = X_all_20 + list(X_temp_20)
    X_all_10 = np.array(X_all_10)
    X_all_20 = np.array(X_all_20)
    np.random.seed(40)
    np.random.shuffle(X_all_10)
    np.random.seed(40)
    np.random.shuffle(X_all_20)    
    return [X_all_10, X_all_20]