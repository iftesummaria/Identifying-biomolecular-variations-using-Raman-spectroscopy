"""
WDEN denoises the signal using wavelet denoising methods similar to that the "wden" function in MATLAB.
All the functions here are replication of MATLAB functions.
For details first 
import WDEN 
then try help(WDEN)
"""
import numpy as np
from pywt import wavedec, waverec

# Wavelet denoising function wden similar to the one availabel in MATLAB
def wden(x, tsr, sorh, scl, N, wname):
    """_summary_: Returns denoised signal using wavelet shrinkage method similar to the wden function in MATLAB.
        XD = wden(x, tsr, sorh, scal, level, wname) returns a denoised version XD of the input signal x obtained
        by thresholding the wavelet coefficients.

    Args:
        x (_type_: numpy array): input one dimensional signal.
        tsr (_type_: string): Threshold selection rule specified as a string. Supported options for tsr are
            'modwtsqtwolog': uses the maximal overlap discrete wavelet transform (MODWT) to denoise the signal with 
            "Donoho & Johnstone's" universal threshold and level-dependent thresholding.
            THe remaining tsr options use the critically sampled DWT to denoise the signal:
            'rigrsure': uses the principle of Stein's Unbiased Risk.
            'heursure': is heuristic variant of Stein's Unbiased Risk.
            'sqtwolog': uses Donoho & Johnstone's Universal Threshold with DWT.
            'minimaxi': uses minimax thresholding.

        sorh (_type_: string): specifies the soft or hard thresholding with 's' or 'h'.
        scl (_type_: string): defines the type of threshold rescaling:
            'one': for no rescaling.
            'sln': for rescaling using a noise estimate based on the first-level coefficients.
            'mln': for rescaling using level-dependent estimates of the noise.
            
        'mln' is the only option supported for MODWT denoising.

        N (_type_: int): N: Wavelet decomposition level.
        wname (_type_: string): is the wavelet.

    Raises:
        ValueError: _description_

    Returns:
       cf( _type_: list): Wavelet coefficients
       cf_thr(_type_:list): Thresholded coefficients
       xden(_type: numpy array): Denoised signal
    """

    # Define a very small number as the lower threshold 
    eps = 2.2204e-16
    
    # Decompose the signal into its wavelet components.
    cf = wavedec(data = x, level = N, wavelet = wname)
    
    # Define threshold rescaling coefficients.
    if scl == 'one':
        s = np.ones((N))
        
    elif scl == 'sln':
        s = wnoisest(cf) * np.ones((N))
        
    elif scl == 'mln':
        s = wnoisest(cf, level = N)
        
    else:
        raise ValueError('Invalid value for scale, scal = %s' %(scl))
        
    # Wavelet coefficient thresholding
    #thr = np.zeros((N))
    cf_thr = [cf[0]]
    for i in range(N):
        if tsr == 'sqtwolog' or tsr == 'minimaxi':            
            th = thselect(np.concatenate(cf, axis = 0), tsr)
            
        else:
            if s[i] < np.sqrt(eps) * np.max(cf[1+i]):
                th = 0
                
            else:
                th = thselect(cf[1+i] / s[i], tsr)
                
        th = th*s[i]
        
        # Threshold wavelet coefficients using wthresh function.
        cfd = np.array(wthresh(cf[1+i], sorh, th))
        
        # append thresholded coefficients
        cf_thr.append(cfd)
        
    # Reconstruct the denoised data from thresholded coefficients
    xden = waverec(cf_thr, wname)

    return cf, cf_thr, xden


def thselect(x, tptr):
    """
    Returns the threshold selected for denoising depending on selection rules.
    Available threshold selection rules are:
    tptr = 'rigrsure', adaptive threshold selection using principle of Stein's Unbiased Risk Estimate.
    tptr = 'heursure', heuristic variant of the first optin.
    tptr = 'sqtwolog', threshold is sqrt(2*log(length(X))).
    tptr = 'minimaxi', minimax thresholding.
    
    """
    
    # Convert x to an array in case it is not an array.
    x = np.array(x)
    l = len(x)
    
    if tptr == 'rigrsure':
        sx2 = [sx*sx for sx in abs(x)]
        sx2.sort()
        cumsumsx2 = np.cumsum(sx2)
        
        risks = []
        for i in range(0, l):
            risks.append((l-2*(i+1) + (cumsumsx2[i] + (l-1-i)*sx2[i]))/l)
            
        mini = np.argmin(risks)
        th = np.sqrt(sx2[mini])
        
        
    elif tptr == 'heursure':
        hth = np.sqrt(2*np.log(l))
        eta = np.sum(abs(x)**2 - l)/l
        crit = (np.log2(l)**1.5)/np.sqrt(l)
        
        if eta < crit:
            th = hth
            
        else:
            sx2 = [sx**2 for sx in abs(x)]
            sx2.sort()
            cumsumsx2 = np.cumsum(sx2)
            
            risks = []
            for i in range(0, l):
                risks.append((l-2*(i+1) + (cumsumsx2[i] + (l-1-i)*sx2[i]))/l)
                
            mini = np.argmin(risks)
            
            rth = np.sqrt(sx2[mini])
            th = min(hth, rth)
            
    elif tptr == 'sqtwolog':
        th = np.sqrt(2*np.log(l))
        
    elif tptr == 'minimaxi':
        if l < 32:
            th = 0
            
        else:
            th = 0.3936 + 0.1829*np.log2(l)
            
    else:
        raise ValueError('Invalid value for threshold selection rule, tptr = %s' %(tptr))
        
    return th
        

def thvalue(c, tpr):
    """
    Returns the threshold values of the input vector.
    INPUT:
        c: is a vector
        tptr: is the threshold selection rule and the availabel options are 'rigrsure', 'heursure', 'sqtwolog' and 'minimaxi'.
    OUTPUT:
        thr: is the threshold value
    """
    # Epsilon stands for a very smal number
    eps = 2.220446049250313e-16
    
    # Calculate the median of the input vector
    s = np.median(abs(c))/0.6745
    
    if tpr == 'sqtwolog' or tpr == 'minimaxi':
        th = thselect(c, tpr)
        
    else:
        if s < np.sqrt(eps)*max(c):
            th = 0
            
        else:
            th = thselect(c/s, tpr)
            
    thr = th*s
    
    return thr

def wnoisest(coef, level = None):
    """
    Estimates variance of 1-D wavelet coefficeints.
    """
    l = len(coef) - 1
    
    if level == None:
        X = [abs(x) for x in coef[-1]]
        stdc = np.median(X) / 0.6745
        
    else:
        stdc = []
        
        for i in range(l):
            X = [abs(x) for x in coef[1+i]]
            stdc.append(np.median(X) / 0.6745)
    return stdc

def wthresh(x, sorh, t):
    """ It performs soft and hard thresholding.
        INPUT:
            x: input array to be thresholded.
            sorh: thresholding mwthod.
            t: thresholding value.
            
    """
    
    if sorh == 's':
        y = [((e < 0)*-1.0 + (e > 0))*((abs(e) - t)*(abs(e) >= t)) for e in x]
        
    elif sorh == 'h':
        y = [e*(abs(e) >= t) for e in x]
        
    else:
        raise ValueError('Invalid value for thresholding type, sorh = %s' %(sorh))
        
    return y

def wthresh_modified(x, thr, tm):
    """
    Threshold detail coefficients using various thresholding methods.
    INPUT:
        x: x is a vector.
        thr: threshold value.
        tm: Prefered thresholding method for denoising. Available methods are 'soft', 'hard',  'Rui-mei',
        'Poornachandra', 'Zhang', 'Lin'.
    """
    # Define thresholded coefficients
    th_cf = np.zeros((len(x)))
    
    # perform soft and hard thresholding using wthresh function 
    if tm == 'soft':
        th_cf = wthresh(x, 's', thr)
        
    elif tm == 'hard':
        th_cf = wthresh(x, 'h', thr)
        
    for i in range(len(x)):
        
        # Rui-mei thresholding
        if tm == 'Rui_mei':
            if abs(x[i]) < thr:
                th_cf[i] = 0
                
            elif x[i] < -thr:
                th_cf[i] = x[i] + ((2*thr) / (1 + np.exp(thr + x[i])))
                
            else:
                th_cf[i] = x[i] - ((2*thr) / (1 + np.exp(thr - x[i])))
                
            
        if tm == 'Poornachandra':
            if abs(x[i]) < thr:
                th_cf[i] = 0
                
            else:
                th_cf[i] = x[i]*(1 + ((x[i]**2) / 6))
                
        if tm == 'Zhang':
            if abs(x[i]) < thr:
                th_cf[i] = 0
                
            else:
                th_cf[i] = np.sign(x[i])*(abs(x[i]) - (thr / np.exp(abs(x[i]) - thr)))
                
        if tm == 'Lin':
            if abs(x[i]) < thr:
                th_cf[i] = 0
                
            else:
                B = np.exp(-(x[i] - thr)**2)
                th_cf[i] = np.sqrt(1-B**2)*np.sign(x[i])*abs(x[i] - thr) + B*x[i]
    
    return th_cf
