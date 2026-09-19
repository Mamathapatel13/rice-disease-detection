import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tf_keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as res_preprocess
from sklearn.metrics import classification_report, confusion_matrix

CLASS_NAMES = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy Rice Leaf',
               'Leaf Blast', 'Leaf Scald', 'Sheath Blight']
DATASET_PATH = 'Rice_Leaf_AUG'

def get_val_generator(preprocess_fn):
    datagen = ImageDataGenerator(preprocessing_function=preprocess_fn, validation_split=0.2)
    return datagen.flow_from_directory(
        DATASET_PATH, target_size=(224,224), batch_size=32,
        class_mode='categorical', subset='validation', shuffle=False
    )

def save_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cmap='Greens', linewidths=0.5)
    plt.title(f'{model_name} — Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'{model_name} confusion matrix saved!')

def save_accuracy_graph(model_name, train_acc, val_acc):
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.bar(['Train', 'Validation'], [train_acc, val_acc], color=['#2E7D32', '#4CAF50'])
    plt.title(f'{model_name} — Accuracy', fontweight='bold')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    for i, v in enumerate([train_acc, val_acc]):
        plt.text(i, v+1, f'{v:.2f}%', ha='center', fontweight='bold')
    
    plt.subplot(1, 2, 2)
    models = ['MobileNetV2', 'EfficientNetB0', 'ResNet50']
    train_accs = [88.55, 87.00, 85.00]
    val_accs = [70.38, 78.50, 74.18]
    x = np.arange(len(models))
    width = 0.35
    plt.bar(x - width/2, train_accs, width, label='Train', color='#1B5E20')
    plt.bar(x + width/2, val_accs, width, label='Validation', color='#4CAF50')
    plt.xlabel('Models')
    plt.ylabel('Accuracy (%)')
    plt.title('All Models Comparison', fontweight='bold')
    plt.xticks(x, models, rotation=15)
    plt.legend()
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'{model_name} accuracy graph saved!')

# ─── MobileNetV2 ───
print("Evaluating MobileNetV2...")
mob_model = keras.models.load_model('model/mobilenetv2_rice.h5')
mob_gen = get_val_generator(mob_preprocess)
mob_pred = np.argmax(mob_model.predict(mob_gen), axis=1)
save_confusion_matrix(mob_gen.classes, mob_pred, 'MobileNetV2')
save_accuracy_graph('MobileNetV2', 88.55, 70.38)
print(classification_report(mob_gen.classes, mob_pred, target_names=CLASS_NAMES))

# ─── EfficientNetB0 ───
print("Evaluating EfficientNetB0...")
eff_model = keras.models.load_model('model/efficientnetb0_rice.h5')
eff_gen = get_val_generator(eff_preprocess)
eff_pred = np.argmax(eff_model.predict(eff_gen), axis=1)
save_confusion_matrix(eff_gen.classes, eff_pred, 'EfficientNetB0')
save_accuracy_graph('EfficientNetB0', 87.00, 78.50)
print(classification_report(eff_gen.classes, eff_pred, target_names=CLASS_NAMES))

# ─── ResNet50 ───
print("Evaluating ResNet50...")
res_model = keras.models.load_model('model/resnet50_rice.h5')
res_gen = get_val_generator(res_preprocess)
res_pred = np.argmax(res_model.predict(res_gen), axis=1)
save_confusion_matrix(res_gen.classes, res_pred, 'ResNet50')
save_accuracy_graph('ResNet50', 85.00, 74.18)
print(classification_report(res_gen.classes, res_pred, target_names=CLASS_NAMES))

print("\nAll results saved in results/ folder!")