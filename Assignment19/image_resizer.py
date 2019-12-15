from PIL import Image
import os,sys
import cv2

path="C:/EVA 2.0/session 16/image annotation"

for i,item in enumerate(os.listdir(path)):
	
	img_path=os.path.join(path+"/"+item)
	print(img_path)
	im=Image.open(img_path)
	f,e=os.path.splitext(img_path)
	newim = im.resize((400,400),Image.ANTIALIAS)
	# im.thumbnail((400,400))
	if i in range(0,10):
		newim.save('C:/EVA 2.0/session 16/resized_images'+'/'+'img_00'+str(i+1)+'.jpg','JPEG',quality=90)
	elif i in range(10,100) :
		newim.save('C:/EVA 2.0/session 16/resized_images'+'/'+'img_0'+str(i)+'.jpg','JPEG',quality=90)
	else:
		newim.save('C:/EVA 2.0/session 16/resized_images'+'/'+'img_'+str(i)+'.jpg','JPEG',quality=90)	
