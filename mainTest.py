import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumorCategorical.h5')

image=cv2.imread('uploads\\pred50.jpg')

img=Image.fromarray(image)

img=img.resize((120,120))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

result=model.predict(input_img)
predicted_class = np.argmax(result, axis=1)
print("class",predicted_class)




