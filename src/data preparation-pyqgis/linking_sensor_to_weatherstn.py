from collections import Counter
from itertools import groupby
from qgis.core import *
import processing
#hope you note the forward slash/...windows uses backslash \
filteredweather='C:/Users/t-82maca1mpg/Desktop/MSCPG_CalebJuma/data/data_filtered/all_records_for_21_Filtered_Weatherhr_till_11Nov_withconcatField.shp'
filteredweatherstns=QgsVectorLayer(filteredweather,'all_records_for_21_Filtered_Weatherhr_till_11Nov_withconcatField','ogr')

###NOTICE: the target layer (in this case the sensors)should be activated in the layers ToC
locs=iface.activeLayer()

#carrying out a join nbetween a new field created in both layers.
#the new field is a concat of timestamp and locid

jfield='mergeweth'
tfield='mergesens'
####UNDO when you want to execute!!!
#joinObject = QgsVectorLayerJoinInfo() 
#joinObject.setJoinFieldName(jfield) #and joinObject.setJoinFieldName('timestamp')
#joinObject.setTargetFieldName(tfield) #and joinObject.setTargetFieldName('timestamp')
#joinObject.setUsingMemoryCache(True)
#joinObject.setJoinLayer(filteredweatherstns)
#locs.addJoin(joinObject)
#

#only problem, the field titles will be affected but than can be changed in a QGIS edit property session
print("end")
####END