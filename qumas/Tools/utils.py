import pandas as pd 
from pathlib import Path

_DEFAULT_filterinfo = Path(__file__).resolve().parent.parent / "Tables" / "filterinfo" / "filter_unique.csv"

filterinfo = pd.read_csv(_DEFAULT_filterinfo)

def clean_name(L):
    "remove leters from the name, caracteres, hopefully we can obtain compare those with previous results"
    if not isinstance(L,str):
        l=[]
        for ii in L:
            for i in [chr(i) for i in range(ord('A'),ord('Z')+1)]:
                    ii=ii.replace(i,"").replace("†","").replace("−","-").replace("*","")
            l.append(ii)
        L=l
    else:
        for i in [chr(i) for i in range(ord('A'),ord('Z')+1)]:
            L=L.replace(i,"").replace("†","").replace("−","-").replace("*","")
    return L
