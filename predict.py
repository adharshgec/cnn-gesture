import numpy as np
import string as str
from keras.preprocessing import image
from keras.models import load_model

model = load_model('gesture.h5')
test_image = image.load_img('dataset/single_prediction/G1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = model.predict(test_image)
#pred_class = list(result).index(max(result))
for res in result[0]:
	#print(str.ascii_uppercase[res])
	#print('=')
	print(res)
#print(pred_class)