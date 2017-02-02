	#Preparation of file with name of images and idx
#Save these names and indexes in csv
import os
import csv


#Read name of images
folder = '/home/asgunzi/CarND-Simulator/IMG'
listImg =os.listdir(folder)
listImg.sort()
lst_names =listImg
#lenImg: number of images
#lenef: Number of center images
lenImg = len(lst_names)
lenref =int(len(lst_names)/3)

#===================================================
#Angle ref
#Save only references to angles greater than equal angleRef
angleRef = 0 #0.01

#AngleOffset to be added to the left or right images
angleOffset = 0.08
#===================================================
#Read angles
p = []
p_names=[]
with open('driving_log.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        p.append(row[3])
        p_names.append(row[0])


#===================================================
#Save csv
out_names =[]
i=0
nr=0
angle = 0

with open("nameIdx.csv","w") as oFile:
	wr = csv.writer(oFile,delimiter = ',', quoting=csv.QUOTE_ALL)

	for filename in listImg:
		idx = i % lenref
		grp = int(i / lenref) #0 = center, 1 = left, 2 = right

		i+=1
		# wr.writerow([idx, filename, float(p[idx+1])])
		#print(float(p[idx+1]),float(angleRef), float(p[idx+1])>float(angleRef), float(p[idx+1])<-float(angleRef))   
		# #if float(p[idx+1]) > float(angleRef) or float(p[idx+1]) < -float(angleRef):
		if grp == 0: #Center
			angle = float(p[idx+1])		
		elif grp == 1: #Left
			angle = (float(p[idx+1]) + angleOffset)
		elif grp == 2: #Right
			angle = (float(p[idx+1]) - angleOffset)


		if  float(p[idx+1])>=float(angleRef) or float(p[idx+1])<-float(angleRef) :
			wr.writerow([idx, filename, angle,p_names[idx+1] ])
			nr+=1

print(nr)
