from collections import Counter
from itertools import groupby
from qgis.core import *

#IMPORTANT make sure your layer is selected (underlined)
#the layer to use i.e. selected
locs=iface.activeLayer()#sensors

#selecting the unique Sensors on arttribute table 
all_sensors=[]

for point in locs.getFeatures():
    if point[1] not in all_sensors:#column 2 is the sensorid field
        locs.select(point.id())#ends up selecting unique sensors on attribute table
    all_sensors.append(point[1])
 
Uniqsensors=[attribute for attribute, count in Counter(all_sensors).items() if count >0] 
 
print("the number of sensors is",len(all_sensors))
print("the unique sensors are: ",Uniqsensors )
print("the number of unique sensors is= ",len(Uniqsensors))


#####END