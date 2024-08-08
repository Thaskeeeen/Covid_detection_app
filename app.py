import streamlit as st
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
from skimage.metrics import structural_similarity as ssim
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

# Define Quantum Fourier Transform (QFT) and its inverse
def qft(circuit, n):
    for j in range(n):
        circuit.h(j)
        for k in range(j+1, n):
            circuit.cp(np.pi/float(2**(k-j)), j, k)
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def qft_dagger(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    for j in range(n):
        for k in range(j):
            circuit.cp(-np.pi/float(2**(j-k)), k, j)
        circuit.h(j)
    return circuit

# Encryption function
def encrypt_image(image):
    flat_image = image.flatten()
    flat_image = flat_image / 255.0
    phase_angles = 2 * np.pi * flat_image

    num_qubits = int(np.ceil(np.log2(flat_image.size)))
    qc = QuantumCircuit(num_qubits)

    for i, angle in enumerate(phase_angles):
        qc.rz(angle, i % num_qubits)

    qft(qc, num_qubits)

    sim = Aer.get_backend('aer_simulator')
    qc.save_statevector()
    qc = transpile(qc, sim)
    result = sim.run(qc).result()
    statevector = result.get_statevector(qc)

    return statevector

# Decryption function
def decrypt_image(statevector, image_shape):
    num_qubits = int(np.log2(len(statevector)))
    qc = QuantumCircuit(num_qubits)
    qc.initialize(statevector, range(num_qubits))

    qft_dagger(qc, num_qubits)

    sim = Aer.get_backend('aer_simulator')
    qc.save_statevector()
    qc = transpile(qc, sim)
    result = sim.run(qc).result()
    decrypted_statevector = result.get_statevector(qc)

    phase_angles = np.angle(decrypted_statevector)
    flat_image = (phase_angles / (2 * np.pi)) * 255.0
    flat_image = flat_image[:np.prod(image_shape)]
    decrypted_image = flat_image.reshape(image_shape)

    return decrypted_image

# Image modification function
def modify_image(image):
    """Add a small amount of noise to the image."""
    noise = np.random.normal(0, 0.05, image.shape)  # Adding Gaussian noise
    noisy_image = np.clip(image + noise, 0, 1)  # Ensure values are between 0 and 1
    return noisy_image

def calc(imgs):
    model = load_model('model.keras')
    imgs = cv2.cvtColor(np.array(imgs), cv2.COLOR_RGB2BGR)
    imgs = cv2.resize(imgs, (256, 256))
    imgs = imgs / 255.0
    imgs = np.expand_dims(imgs, axis=0)
    yhat = model.predict(imgs)
    print("Raw prediction:", yhat)  # Debugging line
    return yhat[0][0] > 0.5

def calculator(img):
    model = load_model('model.keras')
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = resize_image(Image.fromarray(img), target_size=(256, 256))
    img = np.array(img) / 255.0

    encrypted_image = encrypt_image(img)
    decrypted_image = decrypt_image(encrypted_image, img.shape)

    modified_img = modify_image(img)
    encrypted_modified_image = encrypt_image(modified_img)
    decrypted_modified_image = decrypt_image(encrypted_modified_image, img.shape)

    img = np.expand_dims(decrypted_image, axis=0)
    return img, encrypted_image, decrypted_image, modified_img, encrypted_modified_image, decrypted_modified_image

def resize_image(image, target_size=(256, 256)):
    image = image.resize(target_size, Image.LANCZOS)
    return image

st.title('Covid Detection using CNN')
uploaded_file = st.file_uploader("Choose an image file")

if uploaded_file is not None:
    byte_io = io.BytesIO(uploaded_file.getvalue())
    image = Image.open(byte_io)
    image.save('image.png')
    original_img, encrypted_img, decrypted_img, modified_img, encrypted_mod_img, decrypted_mod_img = calculator(image)

    ans = calc(image)
    st.write('Non Covid' if ans else 'Covid')
    st.image('image.png', caption='Original Image')

    original_img = original_img.squeeze()
    decrypted_img = decrypted_img.squeeze()
    modified_img = modified_img.squeeze()
    decrypted_mod_img = decrypted_mod_img.squeeze()

    encrypted_img_magnitude = np.abs(encrypted_img)
    encrypted_img_magnitude = encrypted_img_magnitude[:256*256]
    encrypted_img_magnitude = encrypted_img_magnitude.reshape((256, 256))

    encrypted_mod_img_magnitude = np.abs(encrypted_mod_img)
    encrypted_mod_img_magnitude = encrypted_mod_img_magnitude[:256*256]
    encrypted_mod_img_magnitude = encrypted_mod_img_magnitude.reshape((256, 256))

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax[0, 0].imshow(original_img, cmap='gray')
    ax[0, 0].title.set_text('Original Image')

    ax[0, 1].imshow(encrypted_img_magnitude, cmap='gray')
    ax[0, 1].title.set_text('Encrypted Original Image')

    ax[0, 2].imshow(decrypted_img, cmap='gray')
    ax[0, 2].title.set_text('Decrypted Original Image')

    ax[1, 0].imshow(modified_img, cmap='gray')
    ax[1, 0].title.set_text('Modified Image')

    ax[1, 1].imshow(encrypted_mod_img_magnitude, cmap='gray')
    ax[1, 1].title.set_text('Encrypted Modified Image')

    ax[1, 2].imshow(decrypted_mod_img, cmap='gray')
    ax[1, 2].title.set_text('Decrypted Modified Image')

    st.pyplot(fig)

    mse = np.mean((original_img - decrypted_img) ** 2)
    mse_mod = np.mean((modified_img - decrypted_mod_img) ** 2)

    st.write(f"Mean Squared Error (MSE) - Original: {mse:.4f}")
    st.write(f"Mean Squared Error (MSE) - Modified: {mse_mod:.4f}")