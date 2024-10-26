import cv2
import numpy as np
import os
from glob import glob
import random

# Crear directorios de salida para imágenes y ground truths (máscaras)
output_dir_images = 'output_images/images/'
output_dir_masks = 'output_images/masks/'

if not os.path.exists(output_dir_images):
    os.makedirs(output_dir_images)

if not os.path.exists(output_dir_masks):
    os.makedirs(output_dir_masks)

# Cargar las imágenes y las máscaras
images = sorted(glob('retina/train/images/*.jpg'))  # Ruta a tus imágenes originales
masks = sorted(glob('retina/train/groundtruths/*.jpg'))    # Ruta a tus máscaras originales

# Función para ajustar brillo y contraste
def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# Función para aplicar ecualización de histograma
def apply_histogram_equalization(img):
    if len(img.shape) == 3:  # Imagen a color
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    else:  # Imagen en escala de grises
        return cv2.equalizeHist(img)

# Función para aplicar detección de bordes Canny
def apply_canny_edge_detection(img):
    return cv2.Canny(img, 100, 200)

# Función para aplicar filtro Sobel
def apply_sobel_edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(grad_x, grad_y)
    return np.uint8(sobel)

# Función para añadir ruido gaussiano
def add_gaussian_noise(img):
    mean = 0
    sigma = 25
    noise = np.random.normal(mean, sigma, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

# Función para realizar rotaciones
def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

# Generar nuevas imágenes y máscaras con las transformaciones
total_images = 0
target_images = 980  # Total de imágenes que quieres generar
augmentations_per_image = 50  # Aumentamos el número de augmentaciones por imagen

for idx, (img_path, mask_path) in enumerate(zip(images, masks)):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Guardar la imagen original y su máscara
    cv2.imwrite(f'{output_dir_images}/img_{idx:03d}_original.png', img)
    cv2.imwrite(f'{output_dir_masks}/mask_{idx:03d}_original.png', mask)
    total_images += 1

    # Aplicar augmentations
    for aug in range(augmentations_per_image):
        if total_images >= target_images:
            break

        # Selección aleatoria de transformaciones para cada imagen
        transformed_img = img.copy()
        transformed_mask = mask.copy()

        # Prefijo para la imagen generada
        prefix = ""

        # Aplicar rotación aleatoria
        angle = random.choice([0, 90, 180, 270])  # Añadir la rotación de 0 grados
        transformed_img = rotate_image(transformed_img, angle)
        transformed_mask = rotate_image(transformed_mask, angle)
        prefix += f"rot_{angle}_"

        # Aplicar ajuste de brillo y contraste aleatorio
        alpha = random.uniform(0.5, 1.5)  # Contraste
        beta = random.randint(-50, 50)    # Brillo
        transformed_img = adjust_brightness_contrast(transformed_img, alpha=alpha, beta=beta)
        prefix += f"bright_contrast_"

        # Aplicar ecualización de histograma
        transformed_img = apply_histogram_equalization(transformed_img)
        prefix += "hist_eq_"

        # Aplicar Sobel con alta probabilidad
        transformed_img = apply_sobel_edge_detection(transformed_img)
        prefix += "sobel_"

        # Aplicar filtro de Canny
        transformed_img = apply_canny_edge_detection(transformed_img)
        prefix += "canny_"

        # Aplicar ruido gaussiano
        transformed_img = add_gaussian_noise(transformed_img)
        prefix += "gaussian_noise_"

        # Guardar las imágenes transformadas y sus máscaras
        cv2.imwrite(f'{output_dir_images}/{prefix}img_{idx:03d}_aug_{aug}.png', transformed_img)
        cv2.imwrite(f'{output_dir_masks}/{prefix}mask_{idx:03d}_aug_{aug}.png', transformed_mask)
        total_images += 1

    if total_images >= target_images:
        break

print(f"Generación completa. Total de imágenes generadas: {total_images}")
