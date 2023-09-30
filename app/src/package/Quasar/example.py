from Quasar import Quasar
import numpy as np

# Load data sets for training and testing
# Datasets can be accessed on releases page
train_dataset = np.load("./recycled_32_train.npz")
test_dataset = np.load("./recycled_32_test.npz")

# Create Quasar instance
model = Quasar(train_dataset, test_dataset)

# Preprocess images. It will save the pre-processed images to the directory specified.
model.preprocess("./quanvolution/")

# Load pre-processed images from numpy arrays
# q_train_images and q_test_images are default names
q_train_images = np.load("./quanvolution/q_train_images.npy")
q_test_images = np.load("./quanvolution/q_test_images.npy")

# Load the pre-processed images in the model
model.load(q_train_images, q_test_images)

# Train the model
model.train()

# Evaluate the model
model.evaluate()

# Print the prediction of the model on a single image
# It will output the index of the class with the highest probability and hardcoded name.
# In format {'name': prediction_result}
print(model.predict("./image.jpg"))
# {'Box': #}