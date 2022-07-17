# ASL classifier for Sanguage project
# Capstone ECE 454

import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt

os.environ["TFHUB_CACHE_DIR"] = "\\Temp"

image_path = os.path.join(os.path.dirname('c:\\Users\\cazs5\Documents\\ASL_classifier\\asl_alphabet_basic'), 'asl_alphabet_basic')

#test_path = os.path.join(os.path.dirname('c:\\Users\\cazs5\Documents\\ASL_classifier\\asl_basic_test'), 'asl_basic_test')

data = DataLoader.from_folder(image_path, shuffle=True)
#data_test = DataLoader.from_folder(test_path, shuffle=True)

train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)

#test_env_data = data_test

print("Model Created\n")

model = image_classifier.create(train_data, validation_data=validation_data, epochs=5)

model.summary()

print("Beginning Model Evaluation\n")

loss, accuracy = model.evaluate(test_data)

# A helper function that returns 'red'/'black' depending on if its two input
# parameter matches or not.
def get_label_color(val1, val2):
  if val1 == val2:
    return 'black'
  else:
    return 'red'

# Then plot 100 test images and their predicted labels.
# If a prediction result is different from the label provided label in "test"
# dataset, we will highlight it in red color.
plt.figure(figsize=(20, 20))
predicts = model.predict_top_k(test_data)
for i, (image, label) in enumerate(test_data.gen_dataset().unbatch().take(100)):
  ax = plt.subplot(10, 10, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(image.numpy(), cmap=plt.cm.gray)

  predict_label = predicts[i][0][0]
  color = get_label_color(predict_label,
                          test_data.index_to_label[label.numpy()])
  ax.xaxis.label.set_color(color)
  plt.xlabel('Predicted: %s' % predict_label)
plt.show()

print("Exporting Model\n")

model.export(export_dir='model/v2/', tflite_filename='model_v2.tflite', with_metadata=True)
model.export(export_dir='model/v2/', export_format=ExportFormat.LABEL)

print("Evaluating TFlite Model\n")

model.evaluate_tflite('model/v2/model_v2.tflite', test_data)

#model.evaluate_tflite('model/v3/model.tflite', test_env_data)

print("Welcome to the world Image Classifier!\n")
