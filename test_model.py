# Tests a specified SaveModel CNN on any image (for demonstration purposes)

#Single: python test_model.py one models\resnet50_transfer_learning test\lmao.jpg
#Multiple: python test_model.py many models\resnet50_transfer_learning test\smol_test 

import tensorflow as tf
import numpy as np
import sys
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# Unused, specify via cmd
# IMAGE_DIR = "./test"
# MODEL_DIR = "./models/resnet50_transfer_learning"

IMG_DIMS = (227, 227)
CLASSES = ['negative', 'positive']

def eval_model_on_test(model, data_path):
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        class_names=CLASSES,
        seed=42,
        image_size=IMG_DIMS,
        batch_size=128,
    )

    test_labels = []
    predictions = []

    for imgs, labels in tqdm(test_ds, 
                             desc='Predicting on Test Data'):
        batch_preds = model.predict(imgs)
        predictions.extend(batch_preds)
        test_labels.extend(labels)

    predictions = np.array(predictions)
    predictions = predictions.ravel()
    test_labels = np.array(test_labels)

    return test_labels, predictions

def predict_on_dataset(model, data_path):
    y_true, y_pred = eval_model_on_test(model, data_path)
    predicted_labels = np.array([1 if p > 0.5 else 0 for p in y_pred])
    print(classification_report(y_true, predicted_labels, 
                                target_names=CLASSES))

def predict_on_single_image(model, image_path):
    image = tf.keras.preprocessing.image.load_img(image_path)
    image = tf.image.resize(image, IMG_DIMS)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    predicted_labels = np.array(["a" if p > 0.5 else "no" for p in prediction])
    print("The wise algorithm prophet has decreed that there is:", predicted_labels[0], "crack")
    print("Score:", prediction[0])

if __name__ == "__main__":
    if len(sys.argv) == 4:
        model = tf.keras.models.load_model(sys.argv[2])
        if sys.argv[1] == "one":
            predict_on_single_image(model, sys.argv[3])
        elif sys.argv[1] == "many":
            predict_on_dataset(model, sys.argv[3])
        else:
            print("Usage: python test_model.py MODE(one/many) MODEL_PATH DATA_PATH")
    else:
        print("Usage: python test_model.py MODE(one/many) MODEL_PATH DATA_PATH")


