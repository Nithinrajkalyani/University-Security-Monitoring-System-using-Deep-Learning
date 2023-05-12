from keras.preprocessing.image import ImageDataGenerator
from skimage import io
datagen = ImageDataGenerator(        
        rotation_range = 40,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        brightness_range = (0.5, 1.5))
import numpy as np
import os
from PIL import Image
image_directory = r'C:/Mini/MCropped/'
SIZE = 224
dataset = []
my_images = os.listdir(image_directory)
t=0
for i, image_name in enumerate(my_images):
   
    print('Folder Number',image_name)
    t=image_name.split('C')[1]
    tmppath=os.path.join(image_directory,image_name)
    print(tmppath)
    dataset = []
    for image_name in os.listdir(tmppath):
        print(image_name)
        if (image_name.split('.')[1] == 'jpg'):
            image = io.imread(tmppath +'/'+ image_name)        
            image = Image.fromarray(image, 'RGB')        
            image = image.resize((SIZE,SIZE)) 
            dataset.append(np.array(image))
       

    x = np.array(dataset)
    i = 0
    for batch in datagen.flow(x, batch_size=5,
                          save_to_dir= r'C:/Mini/Augmented/C'+str(t),
                          save_prefix='dr',
                          save_format='jpg'): 
        i += 1
        if i > 50:
            break