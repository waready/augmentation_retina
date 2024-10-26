import cv2
import numpy as np
import os
from glob import glob

# Crear directorios de salida para imágenes y máscaras
output_dir_images = 'augmentacion-retina/images/'
output_dir_masks = 'augmentacion-retina/groundtruths/'

if not os.path.exists(output_dir_images):
    os.makedirs(output_dir_images)

if not os.path.exists(output_dir_masks):
    os.makedirs(output_dir_masks)

# Cargar las imágenes y las máscaras
images = sorted(glob('retina/train/imagen/*.jpg'))
masks = sorted(glob('retina/train/groundtruths/*.jpg'))

# Verificar que hay imágenes y máscaras
print(f"Imágenes encontradas: {len(images)}")
print(f"Máscaras encontradas: {len(masks)}")

if not images or not masks:
    print("No se encontraron imágenes o máscaras en las rutas especificadas.")
    exit()

# Función para ajustar brillo
def adjust_brightness(img, beta):
    return cv2.convertScaleAbs(img, beta=beta)

# Función para rotar imagen
def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

# Generar imágenes y máscaras
total_images = 0
brightness_levels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Niveles de brillo más extensos
rotation_angles = list(range(0, 360, 30))  # Rotaciones cada 30 grados

for idx, (img_path, mask_path) in enumerate(zip(images, masks)):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    for brightness in brightness_levels:
        transformed_img = adjust_brightness(img, beta=brightness)
        for angle in rotation_angles:
            rotated_img = rotate_image(transformed_img, angle)
            rotated_mask = rotate_image(mask, angle)

            # Guardar imagen y máscara
            cv2.imwrite(f'{output_dir_images}/img_{idx:03d}_brightness_{brightness}_rot_{angle}.jpg', rotated_img)
            cv2.imwrite(f'{output_dir_masks}/mask_{idx:03d}_brightness_{brightness}_rot_{angle}.jpg', rotated_mask)
            total_images += 1
            print(f'Generando imagen {total_images}: brillo {brightness}, rotación {angle}')

            if total_images >= 1000:
                break
        if total_images >= 1000:
            break
    if total_images >= 1000:
        break

print(f"Generación completa. Total de imágenes generadas: {total_images}")
