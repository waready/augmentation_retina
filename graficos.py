import pandas as pd
import matplotlib.pyplot as plt

# Cargar los archivos CSV
test_metrics = pd.read_csv("test_metrics3.csv")
epoch_log = pd.read_csv("Epoch_Log3.csv")

### --- Gráficos de Resultados de Prueba (Dice vs IOU) ---

# Extraer datos de las métricas de prueba
dice_values = test_metrics['dice']
iou_values = test_metrics['iou']

# Encontrar los valores máximos y sus índices
max_dice_idx = dice_values.idxmax()
max_iou_idx = iou_values.idxmax()

# Crear los gráficos
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Dispersión (Scatter) de Dice vs IOU
ax[0].scatter(dice_values, iou_values, color='b', alpha=0.7)
ax[0].set_xlabel('Dice Coefficient')
ax[0].set_ylabel('IOU')
ax[0].set_title('Scatter plot: Dice vs IOU')
ax[0].annotate(f'Max Dice: {dice_values[max_dice_idx]:.4f}', 
               (dice_values[max_dice_idx], iou_values[max_dice_idx]), 
               xytext=(10, -10), textcoords='offset points', 
               arrowprops=dict(arrowstyle='->', color='black'))

# Gráfico 2: Evolución del rendimiento por muestras
ax[1].plot(dice_values, label='Dice Coefficient', marker='o', linestyle='-', color='g')
ax[1].plot(iou_values, label='IOU', marker='x', linestyle='--', color='r')
ax[1].annotate(f'Max IOU: {iou_values[max_iou_idx]:.4f}', 
               (max_iou_idx, iou_values[max_iou_idx]), 
               xytext=(10, -10), textcoords='offset points', 
               arrowprops=dict(arrowstyle='->', color='black'))
ax[1].set_xlabel('Sample Index')
ax[1].set_ylabel('Score')
ax[1].set_title('Performance over Samples')
ax[1].legend()

# Mostrar los gráficos
plt.tight_layout()
plt.show()

### --- Gráficos de Entrenamiento por Épocas ---

# Extraer datos de entrenamiento
epochs = epoch_log['epoch']
train_dice = epoch_log['dice_coef']
val_dice = epoch_log['val_dice_coef']
train_iou = epoch_log['iou']
val_iou = epoch_log['val_iou']
loss = epoch_log['loss']
val_loss = epoch_log['val_loss']

# Encontrar los valores óptimos
max_train_dice_idx = train_dice.idxmax()
max_val_dice_idx = val_dice.idxmax()
min_val_loss_idx = val_loss.idxmin()

# Crear los gráficos
fig, ax = plt.subplots(2, 2, figsize=(16, 10))

# Gráfico 1: Coeficiente Dice (Entrenamiento vs Validación)
ax[0, 0].plot(epochs, train_dice, label='Train Dice', marker='o', linestyle='-', color='b')
ax[0, 0].plot(epochs, val_dice, label='Val Dice', marker='x', linestyle='--', color='r')
ax[0, 0].annotate(f'Max Val Dice: {val_dice[max_val_dice_idx]:.4f}', 
                  (max_val_dice_idx, val_dice[max_val_dice_idx]), 
                  xytext=(10, -10), textcoords='offset points', 
                  arrowprops=dict(arrowstyle='->', color='black'))
ax[0, 0].set_xlabel('Epoch')
ax[0, 0].set_ylabel('Dice Coefficient')
ax[0, 0].set_title('Dice Coefficient over Epochs')
ax[0, 0].legend()

# Gráfico 2: IOU (Entrenamiento vs Validación)
ax[0, 1].plot(epochs, train_iou, label='Train IOU', marker='o', linestyle='-', color='g')
ax[0, 1].plot(epochs, val_iou, label='Val IOU', marker='x', linestyle='--', color='m')
ax[0, 1].set_xlabel('Epoch')
ax[0, 1].set_ylabel('IOU')
ax[0, 1].set_title('IOU over Epochs')
ax[0, 1].legend()

# Gráfico 3: Pérdida (Entrenamiento vs Validación)
ax[1, 0].plot(epochs, loss, label='Train Loss', marker='o', linestyle='-', color='c')
ax[1, 0].plot(epochs, val_loss, label='Val Loss', marker='x', linestyle='--', color='y')
ax[1, 0].annotate(f'Min Val Loss: {val_loss[min_val_loss_idx]:.4f}', 
                  (min_val_loss_idx, val_loss[min_val_loss_idx]), 
                  xytext=(10, 10), textcoords='offset points', 
                  arrowprops=dict(arrowstyle='->', color='black'))
ax[1, 0].set_xlabel('Epoch')
ax[1, 0].set_ylabel('Loss')
ax[1, 0].set_title('Loss over Epochs')
ax[1, 0].legend()

# Gráfico 4: Resumen de métricas de entrenamiento
ax[1, 1].plot(epochs, train_dice, label='Train Dice', marker='o', linestyle='-', color='b')
ax[1, 1].plot(epochs, train_iou, label='Train IOU', marker='x', linestyle='--', color='g')
ax[1, 1].plot(epochs, loss, label='Train Loss', marker='^', linestyle='-.', color='c')
ax[1, 1].set_xlabel('Epoch')
ax[1, 1].set_ylabel('Metrics')
ax[1, 1].set_title('Train Metrics Overview')
ax[1, 1].legend()

# Mostrar los gráficos
plt.tight_layout()
plt.show()
