import argparse
import json
import numpy as np
import tensorflow as tf
from PIL import Image


def process_image(image):
    """Resize and normalize image to (224, 224, 3) NumPy array"""
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255.0
    return image.numpy()


def predict(image_path, model, top_k):
    """Return top K probabilities and classes for a given image"""
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_image = process_image(test_image)
    expanded_image = np.expand_dims(processed_image, axis=0)
    predictions = model.predict(expanded_image)
    top_k_probs, top_k_indices = tf.math.top_k(predictions[0], k=top_k)
    top_k_probs = top_k_probs.numpy().tolist()
    top_k_classes = [str(idx) for idx in top_k_indices.numpy().tolist()]
    return top_k_probs, top_k_classes


# Argument parser
parser = argparse.ArgumentParser(description='Flower Species Classifier')
parser.add_argument('image_path', type=str, help='Path to input image')
parser.add_argument('saved_model', type=str, help='Path to saved Keras model (.h5 or .keras)')
parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
parser.add_argument('--category_names', type=str, default=None, help='Path to JSON file mapping labels to flower names')
args = parser.parse_args()

# Load model
model = tf.keras.models.load_model(args.saved_model)

# Predict
probs, classes = predict(args.image_path, model, args.top_k)

# Load category names if provided
if args.category_names:
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
    classes = [class_names[c] for c in classes]

# Print results
print(f"\nTop {args.top_k} Predictions:")
print("-" * 30)
for prob, cls in zip(probs, classes):
    print(f"Class: {cls:<30} Probability: {prob:.4f}")
