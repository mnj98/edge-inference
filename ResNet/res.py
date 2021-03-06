from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import sys
import time
model = ResNet50(weights='imagenet')

if len(sys.argv) < 2:
    print("Please specify image")
img_path = sys.argv[1]
#img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

for i in range(5):
    start = time.time()
    preds = model.predict(x)
    print("--- %s seconds ---" % (time.time() - start))
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
