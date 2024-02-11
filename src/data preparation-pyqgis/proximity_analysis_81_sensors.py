from collections import Counter
from itertools import groupby
from qgis.core import *
import processing

load81='C:/Users/t-82maca1mpg/Desktop/MSCPG_CalebJuma/data/data_filtered/81_Unique_Sensors_with_GreaterorEqualto_3360_records.shp'

loadweatherUniq='C:/Users/t-82maca1mpg/Desktop/MSCPG_CalebJuma/data/data_filtered/the Unique_Stations_Weatherhr_till_11Nov.shp'


sensor81=QgsVectorLayer(load81,'81_Unique_Sensors_with_GreaterorEqualto_3360_records','ogr')
Uniqweatherstns=QgsVectorLayer(loadweatherUniq,'the Unique_Stations_Weatherhr_till_11Nov','ogr')

weth=[]#an empy list to be filled with the weather stations closer to the sensors
#getting nearest weather station to sensor

L=[]#will contains the smallest distances for 81 sensors

for f in sensor81.getFeatures():
    po=f.geometry()
    pt=QgsGeometry.asPoint(po)
    pt2=QgsGeometry.fromPointXY(pt)
    smallest= None
#    sensorr.append(pt2)
    for m in Uniqweatherstns.getFeatures():
        mo=m.geometry()
        mt=QgsGeometry.asPoint(mo)
        mt2=QgsGeometry.fromPointXY(mt)
        dist=QgsGeometry.distance(pt2,mt2)
        
#        print(f[1],m[1],m[0],"   distance= ",dist)
#        print("yeap")
        if smallest is None or dist < smallest:
            smallest = dist
            
    L.append(smallest)
    print(f[1], smallest)
    
print(L )  
sensorr=[] #will contain sensor ids
wethid=[] # will be filled with the closest weather station to the sensors in a key value pair arrangement
for f in sensor81.getFeatures():
    po=f.geometry()
    pt=QgsGeometry.asPoint(po)
    pt2=QgsGeometry.fromPointXY(pt)
    
#    sensorr.append(pt2)
    for m in Uniqweatherstns.getFeatures():
        mo=m.geometry()
        mt=QgsGeometry.asPoint(mo)
        mt2=QgsGeometry.fromPointXY(mt)
#        dist=QgsGeometry.distance(pt2,mt2) 
        for s in L:
            dist=QgsGeometry.distance(pt2,mt2)
            if s==QgsGeometry.distance(QgsGeometry.fromPointXY(QgsGeometry.asPoint(f.geometry())),QgsGeometry.fromPointXY(QgsGeometry.asPoint(m.geometry()))):
                print(f[1],m[1],dist,s)#m[1] is the weather station name
                weth.append(m[1])
                sensorr.append(f[1])#f[1] is the sensor id
                wethid.append(m[0])# m[0] is the weather id on list m
           

key,value=sensorr,wethid
tutu=list(zip(key,value))#creating a list of key value pairs having sensor id and closest weatherstan
print("tutu", tutu)

print("all weather stations near to the sensors", weth)
print()

#now getting the stations without repetitions
weth2=[]
for k,v in Counter (weth).items():
    if v>0:
        weth2.append(k)
    
print ("final weatherstations to use are: ", weth2)
print("we will use {} unique weather stations".format(len(weth2)))
################################################################################
####UNHASH this BLOCK (Block1) only after selecting the right layer in the Table of contents/legend
# make inactive if as it will lead to an edition of the attribute table of layer affected

#toedit=iface.activeLayer()
#prov=toedit.dataProvider()
#fieldd=toedit.fields().lookupField('LocidWeth')
#
#for l,m in tutu:
#    for t in toedit.getFeatures():
#        if l==t[1]:
#            atts={fieldd: m}
#            prov.changeAttributeValues({t.id(): atts})
        
#the result will be new entrants in the field LocidWeth..but incase attribute table is open close it..
#..then open again
 #################################################################################### 
        
        
######NOTICE only UNHASH the below block (BLOCK 2) ONLY after sppecifying the active layer on legend
#for it to select on attribute table, need to make the layer active on legend
#for this case, use the whole weather dataset to get all the records pertaining to the 21 weatherstations
#thus only unhash the below block only after selecting active layer on legendi.e needs to be underlined

#actay=iface.activeLayer()
#for w in actay.getFeatures():
#    if w[1] in weth2:
#        actay.select(w.id())
#    

print("end")


####END