import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import tensorflow as tf
from tensorflow import keras
import os
from PIL import Image


class Quasar:
    def __init__(self) -> None:
        self.index_name = ["Boxes", "Glass Bottles", "Soda Cans", "Crushed Soda Cans", "Plastic Bottles"]

        self.q_train_images = []
        self.q_test_images = []


        # Load the recycled dataset
        train_data = np.load('./notebooks/dataset/recycled_32_train.npz')
        test_data = np.load('./notebooks/dataset/recycled_32_test.npz')

        # Extract the arrays

        # split data and labels
        self.x_train = train_data['x']
        self.y_train = train_data['y']
        self.x_test = test_data['x']
        self.y_test = test_data['y']

        # Preprocess the data
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

        # Reshape the data
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 32, 32, 3)

        self.x_test = self.x_test.reshape(self.x_test.shape[0], 32, 32, 3)

        self.path = "./notebooks/quanvolution/"

        self.model = None


    def preprocess(self) -> None:
        print("Quantum pre-processing of train images:")
        for idx, img in enumerate(self.x_train):
            print("{}/{}        ".format(idx + 1, self.n_train), end="\r")
            self.q_train_images.append(Quasar.quanv(img))
        self.q_train_images = np.asarray(self.q_train_images)

        print("\nQuantum pre-processing of test images:")
        for idx, img in enumerate(self.x_test):
            print("{}/{}        ".format(idx + 1, self.n_test), end="\r")
            self.q_test_images.append(Quasar.quanv(img))
        self.q_test_images = np.asarray(self.q_test_images)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # Save pre-processed images
        np.save(self.path + "q_train_images.npy", self.q_train_images)
        np.save(self.path + "q_test_images.npy", self.q_test_images)

    def load(self, path="./notebooks/quanvolution/") -> None:
        """
        Load pre-processed images from path if they exists. If the images do not exist, then
        pre-process using preprocess method and save them to the path.
        """
        q_train_images = np.load(self.path + "q_train_images.npy")
        q_test_images = np.load(self.path + "q_test_images.npy")

    @staticmethod
    def quanv(image):
        n_qubits = Quasar.get_params()["n_qubits"]    # Number of qubits used
        # Convolves the input image with many applications of the same quantum circuit.
        out = np.zeros((16, 16, n_qubits))

        # Loop over the coordinates of the top-left pixel of 2 by 2 squares
        for j in range(0, 32, int(n_qubits**(0.5))):
            for k in range(0, 32, int(n_qubits**(0.5))):
                # Process a squared 2 by 2 region of the image with a quantum circuit
                q_results = Quasar.circuit(
                    [
                        image[j, k, 0],
                        image[j, k + 1, 0],
                        image[j + 1, k, 0],
                        image[j + 1, k + 1, 0]
                    ]
                )
                # Assign expectation values to different channels of the output pixel (j/2, k/2)
                for c in range(n_qubits):
                    out[j // 2, k // 2, c] = q_results[c]
        return out

    @staticmethod
    @qml.qnode(qml.device("default.qubit", wires=4), interface="autograd")
    def circuit(phi):
        n_layers = Quasar.get_params()["n_layers"]    # Number of random layers
        n_qubits = Quasar.get_params()["n_qubits"]    # Number of qubits used
        # Encoding of 4 classical input values
        for j in range(n_qubits):
            qml.RY(np.pi * phi[j], wires=j)

        rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, n_qubits))
        # Random quantum circuit
        RandomLayers(rand_params, wires=list(range(n_qubits)))

        # Measurement producing 4 classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]

    @staticmethod
    def get_params():
        return {
            "n_qubits": 4,
            "n_layers": 1,
            "n_epochs": 50,
        }
    
    def train(self):
        # Step 1: Compile
        def MyModel():
            model = keras.models.Sequential([
                keras.layers.Flatten(),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dense(5, activation="softmax"),
            ])
            
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            return model

        self.model = MyModel()

        self.model.fit(
            self.q_train_images,
            self.y_train,
            validation_data=(self.q_test_images, self.y_test),
            batch_size=32,
            epochs=Quasar.get_params().n_epochs,
            verbose=2,
        )

    def save(self, path="./notebooks/model/"):
        self.model.save(path + "quantum_model.h5")
    
    def load_model(self, path="./notebooks/model/"):
        self.model = keras.models.load_model(path + "quantum_model.h5")
    
    def evaluate(self):
        acc, loss = self.model.evaluate(self.q_test_images, self.y_test, verbose=2)
        print("Accuracy: {}%".format(acc * 100))
        print ("Loss: {}".format(loss))
    
    def predict(self, image_path):
        # Load the JPG image
        original_image = Image.open(image_path)

        # Resize the loaded image without maintaining aspect ratio
        desired_size = (32, 32)
        original_image.resize(desired_size)

        # Create a NumPy array from the resized image
        image_array = np.array(original_image)

        # Ensure the image has 3 channels (RGB)
        if len(image_array.shape) == 2:
            image_array = np.stack((image_array,) * 3, axis=-1)

        # Optionally, normalize the pixel values to the range [0, 1]
        image_array = image_array / 255.0

        q_image = self.quanv(image_array)
        result_raw = self.model.predict(np.asarray([q_image]))[0]

        result = {}
        
        for index, val in enumerate(result_raw):
            result[self.index_name[index]] = val
        
        return result