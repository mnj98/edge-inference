from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
import numpy as np
import sys
import time
model = MobileNetV3Large(weights='imagenet')

if len(sys.argv) < 2:
    print("Please specify image")
img_path = sys.argv[1]
#img_path = 'elephant.jpg'
start = time.time()
img = image.load_img(img_path, target_size=(224, 224))
load = time.time() - start
x = image.img_to_array(img)
to_arr = time.time() - start
x = np.expand_dims(x, axis=0)
expand = time.time() - start
x = preprocess_input(x)
process = time.time() - start

prediciton = model.predict(x)
done  = time.time() - start

print('Load time: ', load)
print('Format as array time', to_arr - load)
print('Exapnd dimensions time:', expand - to_arr)
print('Finish preprocessing time', process - expand)
print('Infer time:', done - process)



#for i in range(5):
#    start = time.time()
#    preds = model.predict(x)
#    print("--- %s seconds ---" % (time.time() - start))
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
#    print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
