from keras.utils import plot_model
from keras.models import load_model

classifier = load_model('gesture.h5')
plot_model(classifier, to_file='model.png')