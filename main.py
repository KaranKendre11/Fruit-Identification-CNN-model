from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the models.
model = load_model('cnn_from_scratch_fruits.hdf5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 100, 100, 3), dtype=np.float32)
# Replace this with the path to your image
image = Image.open('fruit.jpg')
#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (100, 100)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
image_array = np.asarray(image)
# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
#print(prediction)
max_predict = {
 0:"Apple",
 1:"Banana",
 2:"Orange"
 }
result_score = 0
result_index = 0
for predict in prediction:
    result_score = max(predict)

for predict in prediction:
    for i in range(0,3):
        if(result_score == predict[i]):
            result_index = i
#print(result_index)

print(max_predict[result_index])
