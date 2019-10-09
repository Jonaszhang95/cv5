# coding=utf-8 
import cv2
import numpy as np 
import matplotlib.pyplot as plt 

img = cv2.imread('5.jpeg',1)
rows,cols,channels = img.shape

def image_crop(img):
    #需要使用img.shape知道图片的大小，本图片（683，1024,3）
    img_crop = img[150:350,600:780]
    return img_crop 

def color_shift(img):
    b,g,r = cv2.split(img)
    r[r>230] = 255
    r[r<230] = r[r<230]+25
    color_shift = cv2.merge((b,g,r))
    return color_shift

def rotate_transform(img):
    #旋转中心点，旋转角度，缩放因子，需要2个点
    M = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),30,0.5)
    img_rotate = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
    return img_rotate

def affine_transform(img):
    #需要3个点
    pts1 = np.float32([[0,0],[cols-1,0],[0,rows-1]])
    pts2 = np.float32([[cols*2,rows*0.1],[cols*0.9,rows*0.2],[cols*0.1,rows*0.9]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(cols, rows))
    return dst

def persPective_transform(img):
    #图片的四个点，以及变换后的四个点
    pts1 = np.float32([[0,0],[cols-1,0],[0,rows-1],[cols-1,rows-1]])
    pts2 = np.float32([[30,30],[cols-100,20],[30,rows-1],[cols-100,rows-70]])
    M_warp = cv2.getPerspectiveTransform(pts1,pts2)
    img_warp = cv2.warpPerspective(img,M_warp,(cols,rows))
    return img_warp


if __name__ =='__main__': 
    #origin_img
    cv2.namedWindow('origin_img',cv2.WINDOW_NORMAL)
    cv2.imshow('origin_img',img)

    #img_crop
    img_crop = image_crop(img)
    cv2.namedWindow('img_crop', cv2.WINDOW_NORMAL)
    cv2.imshow('img_crop',img_crop)

    #img_color_shift
    img_color_shift = color_shift(img) 
    cv2.namedWindow('img_color_shift',cv2.WINDOW_NORMAL)
    cv2.imshow('img_color_shift',img_color_shift)

    #rotate_transform
    img_rotate = rotate_transform(img)
    cv2.namedWindow('img_rotate',cv2.WINDOW_NORMAL)
    cv2.imshow('img_rotate',img_rotate)

    #affine_transform
    affine_transform = affine_transform(img)
    cv2.namedWindow('affine_transform',cv2.WINDOW_NORMAL)
    cv2.imshow('affine_transform',img_rotate)

    #persPective_transform
    persPective_transform = persPective_transform(img)
    cv2.namedWindow('persPective_transform',cv2.WINDOW_NORMAL)
    cv2.imshow('persPective_transform',persPective_transform)
    

    cv2.waitKey(0)
    cv2.destroyALLWindows()



    
    
    
