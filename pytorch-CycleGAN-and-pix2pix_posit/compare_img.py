import sys
import os
from imageio import imread
import numpy as np
def compare_img (img1, img2):
    assert (img1.size == img2.size)
    divider = img2
    divider [divider == 0 ] = 1.0
    return sum(np.abs((img1-img2)/divider))/float(img1.size)
def get_file_list (path):
    file_list  = []

    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if 'fake' in file:
                file_list.append(os.path.join(r, file))
    return file_list
def main(path1, path2):

    # read images as 2D arrays (convert to grayscale for simplicity)
    #img1 = imread(file1).astype(float).flatten()
    #img2 = imread(file2).astype(float).flatten()
    # compare
    img_list1 = get_file_list(path1)
    img_list2 = get_file_list(path2)

    #print (img_list1[:])
    #print (img_list2[:10])
    print (len(img_list1)," ", len(img_list2))
    if(len(img_list1) > len(img_list2)):
        for item in img_list1:
            if item not in img_list2:
                print ("bingo")
                img_list1.remove(item)

    avg_rel_err=[]
    for i in range(len(img_list2)):
        img1 = imread(img_list1[i]).astype(float).flatten()
        img2 = imread(img_list2[i]).astype(float).flatten()
        avg_rel_err.append(compare_img(img1,img2 ))
    print (avg_rel_err)
    print ("avg err",sum(avg_rel_err)/float(len(avg_rel_err)))

if __name__ == '__main__':
    arguments = sys.argv[1:]
    main(arguments[0],arguments[1])
