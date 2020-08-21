import numpy as np
import pandas as pd
path = "C:/Users/hp/Desktop/all/cache/"
traces = ["ts_0",  "wdev_0", "rsrch_0"]
for trace in traces:
    data_txt = np.loadtxt(path+trace+'.txt')
    data_txtDF = pd.DataFrame(data_txt)
    data_txtDF.to_csv(path+trace+'.csv',index=False)
