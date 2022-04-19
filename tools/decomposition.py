'''
@Autuor: LZ-CH
@Contact: 2443976970@qq.com
'''

import numpy as np
import cv2
def lplas_decomposition(img,level_num = 4):
    #input's type: <numpy>
    #output's type: <list<numpy>>
    G_list = []
    L_list = []
    G_list.append(img)
    for i in range(level_num-1):
        G_list.append(cv2.pyrDown(G_list[i]))
    for j in range(level_num-1):
        L_list.append(G_list[j]-cv2.pyrUp(G_list[j+1],dstsize=(G_list[j].shape[1],G_list[j].shape[0])))
    L_list.append(G_list[level_num-1])
    G_list.reverse()
    L_list.reverse()
    return G_list,L_list
if __name__ =='__main__':
    img =cv2.imread('1.png')
    print(img.shape)
    img = img/255
    g,L = lplas_decomposition(img)
    cv2.imwrite('a.jpg',g[0]*255)
    cv2.imwrite('b.jpg',L[3]*255)


