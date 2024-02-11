from collections import Counter
from itertools import groupby
from qgis.core import *


#reference of layer to be searched sensorid. the layer should be active on qgis
sens=iface.activeLayer() #may28th-11thnov dataset

#attribute to be checked sid
sid=sens.fields().indexFromName('sensorid')

#list of all attribute values
listsid=[feature.attributes()[sid] for feature in sens.getFeatures()]

#print the duplicate in the list
showcountlistsid=[attribute for attribute, count in Counter(listsid).items() if count >1]

showcountlistsid.sort()
print(showcountlistsid)
print(len(showcountlistsid))


#now create list with unique time positions
ts=iface.activeLayer()

#attribute to be checked ts
tsid=ts.fields().indexFromName('timestamp')
#list of all attribute values
listts=[feature.attributes()[tsid] for feature in ts.getFeatures()]
#print the duplicate in the list
showcountlistts=[attribute for attribute, count in Counter(listts).items() if count >0]

showcountlistts.sort()
print(showcountlistts)
print(len(showcountlistts))

print("ok now trying to find sensors with similar timestamps")
print("remember, sensorids=",len(showcountlistsid),"and timestamps=", len(showcountlistts))

#Getting the frequency of sensors sending data
#C will have sensor id and frequency of the sensor
C = Counter(listsid)
print("Data format...sensorID: frequency",C)

#create empty list to host sensors whose frequency is greater than 3360
cs=[]#the chosen sensors
#select sensors whose value is 3360 or greater
for key, value in C.items():
    if value >=3360:
        cs.append(key)
        #print(key)
 
print(cs)
print(len(cs))
t=tuple(cs)

for key, value in C.items():
    if value >=3360:
        cs.append(key)
        #print(key)
        query='"sensorid"'+ '=' + '10573


#select the cs sensors from the active layer sens or ts
query='"sensorid"'+ '=' + '10573'
selec=sens.getFeatures(QgsFeatureRequest().setFilterExpression(query))
print(selec)
sens.selectByIds([k.id() for k in selec])


    
