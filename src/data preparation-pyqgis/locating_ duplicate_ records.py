from collections import Counter
#reference of layer to be searched al
sens=iface.activeLayer() #may28th-11thnov dataset

#attribute to be checked sid
sid=sens.fields().indexFromName('sensorid')

#list of all attribute values
list=[feature.attributes()[sid] for feature in sens.getFeatures()]

#print the duplicate in the list
showcountlist=[attribute for attribute, count in Counter(list).items() if count >1]

showcountlist.sort()
print(showcountlist)
print(len(showcountlist))


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


