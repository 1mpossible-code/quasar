from Quasar import Quasar
import numpy as np

train_dataset = np.load("./recycled_32_train.npz")
test_dataset = np.load("./recycled_32_test.npz")

model = Quasar(train_dataset, test_dataset)

model.preprocess("./quanvolution/")

q_train_images = np.load("./quanvolution/q_train_images.npy")
q_test_images = np.load("./quanvolution/q_test_images.npy")

model.load(q_train_images, q_test_images)

model.train()

model.evaluate()

print(model.predict("./image.jpg"))
# {'Box': #}
