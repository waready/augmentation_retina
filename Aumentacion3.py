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
images = sorted(glob('retina/train/images/*.jpg'))
masks = sorted(glob('retina/train/groundtruths/*.jpg'))

# Verificar que hay imágenes y máscaras
print(f"Imágenes encontradas: {len(images)}")
print(f"Máscaras encontradas: {len(masks)}")

if not images or not masks:
    print("No se encontraron imágenes o máscaras en las rutas especificadas.")
    exit()

# Funciones de transformación
def adjust_brightness(img, beta):
    return cv2.convertScaleAbs(img, beta=beta)

def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def flip_image(img, flip_code):
    return cv2.flip(img, flip_code)

def zoom_image(img, zoom_factor):
    height, width = img.shape[:2]
    new_width, new_height = int(width * zoom_factor), int(height * zoom_factor)
    resized_img = cv2.resize(img, (new_width, new_height))
    crop_x1 = (new_width - width) // 2
    crop_y1 = (new_height - height) // 2
    return resized_img[crop_y1:crop_y1 + height, crop_x1:crop_x1 + width]

def translate_image(img, x_shift, y_shift):
    (h, w) = img.shape[:2]
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    return cv2.warpAffine(img, M, (w, h))

# Parámetros de aumento
brightness_levels = [10, 30, 50, 70, 90]
rotation_angles = [0, 90, 180, 270]
flip_codes = [0, 1, -1]  # Horizontal, vertical, ambos
zoom_factors = [1.0, 1.1, 1.2]  # Zoom sin cambiar dimensiones
translate_shifts = [(-10, 0), (10, 0), (0, -10), (0, 10)]  # Translación leve

total_images = 0
target_images = 1000  # Número total deseado de imágenes

for idx, (img_path, mask_path) in enumerate(zip(images, masks)):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    for brightness in brightness_levels:
        bright_img = adjust_brightness(img, beta=brightness)
        for angle in rotation_angles:
            rotated_img = rotate_image(bright_img, angle)
            rotated_mask = rotate_image(mask, angle)
            for flip_code in flip_codes:
                flipped_img = flip_image(rotated_img, flip_code)
                flipped_mask = flip_image(rotated_mask, flip_code)
                for zoom_factor in zoom_factors:
                    zoomed_img = zoom_image(flipped_img, zoom_factor)
                    zoomed_mask = zoom_image(flipped_mask, zoom_factor)
                    for x_shift, y_shift in translate_shifts:
                        translated_img = translate_image(zoomed_img, x_shift, y_shift)
                        translated_mask = translate_image(zoomed_mask, x_shift, y_shift)

                        # Guardar la imagen y la máscara generada
                        img_filename = f'img_{idx:03d}_b{brightness}_r{angle}_f{flip_code}_z{int(zoom_factor*100)}_t{x_shift}_{y_shift}.jpg'
                        mask_filename = f'mask_{idx:03d}_b{brightness}_r{angle}_f{flip_code}_z{int(zoom_factor*100)}_t{x_shift}_{y_shift}.jpg'

                        cv2.imwrite(f'{output_dir_images}/{img_filename}', translated_img)
                        cv2.imwrite(f'{output_dir_masks}/{mask_filename}', translated_mask)
                        total_images += 1

                        if total_images >= target_images:
                            break
                    if total_images >= target_images:
                        break
                if total_images >= target_images:
                    break
            if total_images >= target_images:
                break
        if total_images >= target_images:
            break
    if total_images >= target_images:
        break

print(f"Generación completa. Total de imágenes generadas: {total_images}")
