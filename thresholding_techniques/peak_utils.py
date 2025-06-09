import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import grey_opening
from skimage.filters import threshold_otsu

ION_CHANNELS = [
    '23Na','24Mg','26Mg','27Al','28Si','29Si',
    '39K','40Ca','48Ti','54Fe','55Mn','56Fe','60Ni'
]
METHODS = ['global','sigma','rolling','quantile','top_hat','find_peaks','otsu']
COLORS  = ['C0','C1','C2','C3','C4','C5','C6']
MARKERS = ['o','v','^','s','P','X','D']

def global_threshold(df, ch, percentile):
    x=df[ch]; T=np.percentile(x,percentile)
    return df.loc[x>T,'Time (ms)']

def sigma_threshold(df, ch, k, bg_frac):
    x=df[ch]; bg=x[x<=np.quantile(x,bg_frac)]
    T=bg.mean()+k*bg.std(); return df.loc[x>T,'Time (ms)']

def rolling_threshold(df, ch, window, k):
    x=df[ch]; mu=x.rolling(window,center=True).mean()
    sigma=x.rolling(window,center=True).std()
    return df.loc[x>(mu+k*sigma),'Time (ms)']

def quantile_threshold(df, ch, window, quantile):
    x=df[ch]; T=x.rolling(window,center=True).quantile(quantile)
    return df.loc[x>T,'Time (ms)']

def top_hat_threshold(df, ch, struct_sz, k):
    x=df[ch].values; base=grey_opening(x,size=struct_sz)
    res=x-base; T=res.std()*k
    return df.loc[res>T,'Time (ms)']

def find_peaks_threshold(df, ch, prominence, width):
    x=df[ch].values; peaks,_=find_peaks(x,prominence=prominence,width=width)
    return df.iloc[peaks]['Time (ms)']

def otsu_threshold(df, ch, struct_sz):
    x=df[ch].values; base=grey_opening(x,size=struct_sz)
    res=x-base; T=threshold_otsu(res)
    return df.loc[res>T,'Time (ms)']

def detect_all(df,ch,opts):
    return {
        'global':     global_threshold(df,ch,opts.global_p),
        'sigma':      sigma_threshold(df,ch,opts.sigma_k,opts.sigma_bg),
        'rolling':    rolling_threshold(df,ch,opts.roll_w,opts.roll_k),
        'quantile':   quantile_threshold(df,ch,opts.quant_w,opts.quant_q),
        'top_hat':    top_hat_threshold(df,ch,opts.th_hat,opts.th_k),
        'find_peaks': find_peaks_threshold(df,ch,opts.find_prom,opts.find_w),
        'otsu':       otsu_threshold(df,ch,opts.otsu_sz),
    }

def save_all(df,outdir,opts):
    os.makedirs(outdir,exist_ok=True)
    for ch in ION_CHANNELS:
        results=detect_all(df,ch,opts)
        fig,axes=plt.subplots(len(METHODS),1,figsize=(10,2*len(METHODS)),sharex=True)
        for ax,m,c,mk in zip(axes,METHODS,COLORS,MARKERS):
            ax.plot(df['Time (ms)'],df[ch],color='lightgray',linewidth=0.5)
            times=results[m]
            counts=df.set_index('Time (ms)')[ch].reindex(times,method='nearest').values
            ax.scatter(times,counts,c=c,marker=mk,s=10)
            ax.set_ylabel(m,rotation=0,labelpad=40)
        axes[-1].set_xlabel('Time (ms)')
        plt.tight_layout()
        plt.savefig(f"{outdir}/peaks_{ch}_panels.png",dpi=150)
        plt.close()
        pd.to_pickle(results,f"{outdir}/peaks_{ch}.pkl")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("-i","--input",required=True)
    p.add_argument("-o","--outdir",required=True)
    p.add_argument("--global-p", type=float, default=99.0, dest="global_p")
    p.add_argument("--sigma-k",  type=float, default=4.0, dest="sigma_k")
    p.add_argument("--sigma-bg", type=float, default=0.5, dest="sigma_bg")
    p.add_argument("--roll-w",   type=int,   default=500, dest="roll_w")
    p.add_argument("--roll-k",   type=float, default=4.0, dest="roll_k")
    p.add_argument("--quant-w",  type=int,   default=500, dest="quant_w")
    p.add_argument("--quant-q",  type=float, default=0.995, dest="quant_q")
    p.add_argument("--th-hat",   type=int,   default=101, dest="th_hat")
    p.add_argument("--th-k",     type=float, default=3.0, dest="th_k")
    p.add_argument("--find-prom",type=float, default=5.0, dest="find_prom")
    p.add_argument("--find-w",   type=int,   default=3,   dest="find_w")
    p.add_argument("--otsu-sz",  type=int,   default=101, dest="otsu_sz")
    opts=p.parse_args()
    df=pd.read_csv(opts.input)
    save_all(df,opts.outdir,opts)

# python peak_utils_all.py \
#   -i ../data/NPs_BHVO_Oct23_full.csv \
#   -o results/
