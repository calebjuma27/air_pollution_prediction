from collections import Counter
from itertools import groupby
from qgis.core import *

#IMPORTANT make sure your layer is selected (underlined)
#the layer to use i.e. selected
locs=iface.activeLayer()

#getting number of weather staions as per locid
#first create a list to store the weather stations
weatherstations=[]
for point in locs.getFeatures():
    weatherstations.append(point[0])#o is the first field, thelocid
    #alternatively you can put'locid'
#    weatherstation.append(point['locid'])
Uniqweatherstations=[attribute for attribute, count in Counter(weatherstations).items() if count >0]
print("the weather stations are locid: ", Uniqweatherstations)
print("the number of weather stations is ", len(Uniqweatherstations))

#getting the number of unique timestamps. 
#btw all timestamps will give the numebr of records/features in our layer
timestamp=[]
for point in locs.getFeatures():
    timestamp.append(point[2])# 2 is the third column and contains field timestamp

print("the total number of records in our weather dataset is:", len(timestamp))
#use counter to get unique timestamps
Uniqtimestamp=[attribute for attribute, count in Counter(timestamp).items() if count >0]
print ('Unique timestamps=',Uniqtimestamp)
print ('No. of Unique timestamps=',len(Uniqtimestamp))

#Uniqweatherstations=[attribute for attribute, count in Counter(weatherstations).items() if count >0]Uuniqweatherstations.sort()

#selecting the unique weather stations on arttribute table Getting Unique weather stations
yea=[]
#print(Uniqweatherstations...yea will again contain all weather stations)
for point in locs.getFeatures():
    if point[0] not in yea:
        locs.select(point.id())#ends up selecting unique weatherstaions ofn attribute table
    yea.append(point[0])
 

#####END