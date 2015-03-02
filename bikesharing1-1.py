path = '/Users/haslam/OneDrive/OneDrive-Documents/Python/data/hits.csv'
open(path).readline()
import pandas as pd
import datetime as dt
from datetime import datetime as dtdt
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(path)


stopT = df['stoptime']
startT=df['starttime']

startT1=pd.to_datetime(startT)
stopT1=pd.to_datetime(stopT)
diff = stopT1[:]-startT1[:]
ans=diff.mean()
seconds = ans/np.timedelta64(1,'s')

print("average length of trip in seconds", seconds)

df_bike = df.sort_index(by=['bikeid','stoptime'])

endStations = df_bike['end station id']
endStations.index = np.arange(0,len(endStations))
startStations = df_bike['start station id']
startStations.index = np.arange(0,len(startStations))

length = len(df_bike.index)
length = length - 1
i=0
counter = 0
while i < length:
    # match end stations with the next start station. The total number of records minus the matches is the missing data
    j=i+1
    #print "Start: " + endStations[i] + "\t End:" + startStations[j]
    if(endStations[i]==startStations[j]):
        counter = counter + 1
    i = i+1
        
#print counter
est = (counter*1.0)/(length+1)*100
missing = 100 - est
print("percentage of missing data = ", missing)

#To get average time spent a bike spends at each station you would take average of startStation[i+1] - endStation[i]