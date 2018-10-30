import cv2 
import numpy as np
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt 
 
def get_classificaton(ratio): 
    ratio =round(ratio,1) 
    toret=""
    if(ratio>=3): 
        toret="Slender"
    elif(ratio>=2.1 and ratio<3): 
        toret="Medium"
    elif(ratio>=1.1 and ratio<2.1): 
        toret="Bold" 
    elif(ratio<=1):
        toret="Round" 
        toret="("+toret+")"
    return toret
mypath=r"C:\Users\subha\Desktop\garbage\work\project\pic"
images = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
image = np.empty(len(images), dtype=object)
for n in range(0, len(images)):
    images[n] = cv2.imread( join(mypath,images[n]),0 )
    ret,binary = cv2.threshold(images[n],160,255,cv2.THRESH_BINARY)#averaging  filter   
    kernel = np.ones ((5,5),np.float32)/25
    dst = cv2.filter2D(binary,-1,kernel)# -1 : depth of the destination image
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) #erosion 
    erosion = cv2.erode(dst,kernel2,iterations = 1) #dilation
    dilation = cv2.dilate(erosion,kernel2,iterations = 1) #edge detection	 
    edges = cv2.Canny(dilation,100,200)### Size detection
    _,contours,hierarchy = cv2.findContours(erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) 
    print ("No. of rice grains=",len(contours)) 
    total_ar=0
    for cnt in contours: 
        x,y,w,h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        if(aspect_ratio<1): 
            aspect_ratio=1/aspect_ratio 
        print (round(aspect_ratio,2),get_classificaton(aspect_ratio))
        total_ar+=aspect_ratio 
        avg_ar=total_ar/len(contours) 
    print ("Average Aspect Ratio=",round(avg_ar,2),get_classificaton(avg_ar)) 
    #plot the images 
    imgs_row=2
    imgs_col=3 
    plt.subplot(imgs_row,imgs_col,1),plt.imshow(images[n],'gray') 
    plt.title("Original image")
    plt.xlabel("Length")
    plt.ylabel("Breadth")
    plt.subplot(imgs_row,imgs_col,2),plt.imshow(binary,'gray')
    plt.title("Binary image")
    plt.xlabel("Length")
    plt.ylabel("Breadth")
    plt.subplot(imgs_row,imgs_col,3),plt.imshow(dst,'gray') 
    plt.title("Filtered image")
    plt.xlabel("Length")
    plt.ylabel("Breadth")
    plt.subplot(imgs_row,imgs_col,4),plt.imshow(erosion,'gray')
    plt.title("Eroded image")
    plt.xlabel("Length")
    plt.ylabel("Breadth")
    plt.subplot(imgs_row,imgs_col,5),plt.imshow(dilation,'gray')
    plt.title("Dialated image")
    plt.xlabel("Length")
    plt.ylabel("Breadth")
    plt.subplot(imgs_row,imgs_col,6),plt.imshow(edges,'gray')
    plt.title("Edge detect")
    plt.xlabel("Length")
    plt.ylabel("Breadth")
    plt.show() 
