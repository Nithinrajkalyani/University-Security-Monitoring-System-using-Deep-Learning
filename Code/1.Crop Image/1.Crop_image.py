import cv2
import mtcnn
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import os


def detect_face(list_dir,newpath):
    detector = MTCNN()
    for class_name in list_dir:
        count=0
        print(class_name)
        sub_path="C:\\Mini\\Dataset"+"\\"+class_name
        os.mkdir("C:\\Mini\\Cropped"+"\\"+class_name)
        # print(sub_path)
        sub_dir_list=os.listdir(sub_path)
        # print(sub_dir_list)
        for image in sub_dir_list:
            # print(image)
            count=count+1
            image_path=sub_path+'\\'+image
            # print(image_path)
            image = cv2.imread(image_path)
            #image = cv2.resize(image1,(780,540),interpolation = cv2.INTER_LINEAR)
            
           
            bounding_boxes = detector.detect_faces(image)
            
          
            for box in  bounding_boxes:
                    x1, y1, w, h = box['box']
                    cv2.rectangle(image, (x1, y1), (x1+w,y1+h), (0,255,0), 2)
                    crop_img = image[y1:y1+h, x1:x1+w]

                   
                    new_name=class_name+'\\'+class_name+'_'+str(count)+'.jpg'
                    
                    cv2.imwrite("C:\\Mini\\Cropped"+'\\'+new_name, crop_img)
            
#     return bounding_boxes


# def draw_bounding_boxes(image, bboxes):
#     for box in bboxes:
#         x1, y1, w, h = box['box']
#         cv2.rectangle(image, (x1, y1), (x1+w,y1+h), (0,255,0), 2)
#         crop_img = image[y1:y1+h, x1:x1+w]
#         cv2.imwrite('mousecrop.jpg', crop_img)
        
        
        





# print(mtcnn._version_)

# def detect_face(image):
    
#     return bounding_boxes

# def draw_bounding_boxes(image, bboxes):
#     for box in bboxes:
#         x1, y1, w, h = box['box']
#         cv2.rectangle(image, (x1, y1), (x1+w,y1+h), (0,255,0), 2)

# def mark_key_point(image, keypoint):
#     cv2.circle(image, (keypoint), 1, (0,255,0), 2)
    
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# bboxes = detect_face(image)
# print("Output of MTCNN detector is...\n",bboxes)


oldpath="C:\\Mini\\MCropped"
newpath="C:\\Mini\\Cropped"




dir_list=os.listdir(oldpath)

bboxes=detect_face(dir_list,newpath)