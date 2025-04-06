import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import gc
import warnings
warnings.filterwarnings('ignore')

# ================================
# Custom Focal Loss Implementation
# ================================
class FocalLoss(tf.keras.losses.Loss):
    """Implementation of Focal Loss without tensorflow-addons dependency"""
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        
    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
            
        # Convert y_true to one-hot if needed
        if len(tf.shape(y_true)) == 1 or tf.shape(y_true)[-1] == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), tf.shape(y_pred)[-1])
            
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0))
        
        # Calculate focal loss
        p_t = tf.where(tf.equal(y_true, 1.0), y_pred, 1 - y_pred)
        alpha_factor = tf.where(tf.equal(y_true, 1.0), self.alpha, 1 - self.alpha)
        modulating_factor = tf.pow((1.0 - p_t), self.gamma)
        
        loss = alpha_factor * modulating_factor * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

# ================================
# Paths & Config
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "images")
CSV_FILE = os.path.join(BASE_DIR, "dataset", "HAM10000_metadata.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "skin_disease_model.h5")
SAVEDMODEL_PATH = os.path.join(BASE_DIR, "model", "saved_model")
LOG_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories if they don't exist
for directory in [os.path.dirname(MODEL_PATH), SAVEDMODEL_PATH, LOG_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configuration parameters
IMG_SIZE = (224, 224)  # Increased for transfer learning
BATCH_SIZE = 16  # Reduced batch size to help with gradient updates
EPOCHS = 100  # Increased epochs with early stopping
LEARNING_RATE = 1e-4  # Lower learning rate for transfer learning
WEIGHT_DECAY = 1e-5  # L2 regularization

# ================================
# Load & Prepare Data
# ================================
def load_data(csv_file, image_dir):
    """Load and prepare dataset with error handling"""
    try:
        print("Loading dataset...")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
            
        data = pd.read_csv(csv_file)
        data['image_path'] = data['image_id'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))
        
        # Check for image directory
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
        # Check for existing images
        valid_paths = data['image_path'].apply(os.path.exists)
        if not valid_paths.all():
            print(f"⚠️ {(~valid_paths).sum()} images not found. Filtering dataset.")
            data = data[valid_paths]
            
        if len(data) == 0:
            raise ValueError("No valid images found in the dataset")
        
        # Display class distribution
        class_dist = data['dx'].value_counts()
        print("\nClass distribution:")
        for cls, count in class_dist.items():
            print(f"  {cls}: {count} ({count/len(data)*100:.1f}%)")
        
        return data
    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

try:
    data = load_data(CSV_FILE, DATASET_DIR)
except Exception as e:
    print(f"Failed to load data: {e}")
    import sys
    sys.exit(1)

# ================================
# Data Preprocessing
# ================================
def prepare_dataset(data, test_size=0.15, val_size=0.15):
    """Prepare train/val/test splits with stratification"""
    try:
        # Encode labels
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(data['dx'].values)
        num_classes = len(np.unique(labels_encoded))
        
        # First split: separate test set
        train_val_idx, test_idx = train_test_split(
            np.arange(len(data)),
            test_size=test_size,
            random_state=42,
            stratify=labels_encoded
        )
        
        # Second split: separate validation set from training set
        adjusted_val_size = val_size / (1 - test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=adjusted_val_size,
            random_state=42,
            stratify=labels_encoded[train_val_idx]
        )
        
        return {
            'train': data.iloc[train_idx].reset_index(drop=True),
            'val': data.iloc[val_idx].reset_index(drop=True),
            'test': data.iloc[test_idx].reset_index(drop=True),
            'label_encoder': label_encoder,
            'num_classes': num_classes
        }
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        raise

try:
    dataset = prepare_dataset(data)
    print(f"\nDataset splits: Train={len(dataset['train'])}, Val={len(dataset['val'])}, Test={len(dataset['test'])}")
except Exception as e:
    print(f"Failed to prepare dataset: {e}")
    import sys
    sys.exit(1)

# ================================
# Data Generators with Augmentation
# ================================
def create_data_generators(dataset, img_size):
    """Create data generators with appropriate augmentation"""
    try:
        # Strong augmentation for training to improve generalization
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,  # Skin lesions can be viewed from any angle
            brightness_range=[0.8, 1.2],
            fill_mode='reflect'
        )
        
        # Minimal processing for validation and test sets
        valid_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Prepare the data generators
        train_generator = train_datagen.flow_from_dataframe(
            dataset['train'],
            x_col='image_path',
            y_col='dx',
            target_size=img_size,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = valid_test_datagen.flow_from_dataframe(
            dataset['val'],
            x_col='image_path',
            y_col='dx',
            target_size=img_size,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        test_generator = valid_test_datagen.flow_from_dataframe(
            dataset['test'],
            x_col='image_path',
            y_col='dx',
            target_size=img_size,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, val_generator, test_generator
    
    except Exception as e:
        print(f"Error creating data generators: {e}")
        raise

try:
    train_gen, val_gen, test_gen = create_data_generators(dataset, IMG_SIZE)
except Exception as e:
    print(f"Failed to create data generators: {e}")
    import sys
    sys.exit(1)

# Calculate class weights for imbalanced data
try:
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(dataset['train']['dx']),
        y=dataset['train']['dx']
    )
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    print("\nClass weights:", {dataset['label_encoder'].classes_[i]: f"{w:.2f}" for i, w in class_weights_dict.items()})
except Exception as e:
    print(f"Warning: Could not compute class weights: {e}")
    class_weights_dict = None

# ================================
# Model Architecture with Transfer Learning
# ================================
def create_model(input_shape, num_classes):
    """Create a model based on EfficientNet with customizations"""
    try:
        # Load pre-trained model with frozen layers
        base_model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze base model 
        for layer in base_model.layers:
            layer.trainable = False
            
        # Custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(WEIGHT_DECAY))(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(WEIGHT_DECAY))(x)
        x = Dropout(0.3)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=outputs)
        
        # Use custom focal loss
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss=focal_loss,
            metrics=['accuracy', tf.keras.metrics.AUC(multi_label=True)]
        )
        
        return model, base_model
    
    except Exception as e:
        print(f"Error creating model: {e}")
        raise

try:
    # Create model with transfer learning
    model, base_model = create_model((IMG_SIZE[0], IMG_SIZE[1], 3), dataset['num_classes'])
    model.summary()
except Exception as e:
    print(f"Failed to create model: {e}")
    import sys
    sys.exit(1)

# ================================
# Training Callbacks
# ================================
callbacks = [
    ModelCheckpoint(
        MODEL_PATH,
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_auc',
        patience=15,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)
]

# ================================
# Training Strategy
# ================================
def train_model(model, train_gen, val_gen, epochs, callbacks, class_weights):
    """Two-phase training with gradual unfreezing"""
    try:
        # Phase 1: Train only the top layers
        print("\n--- Phase 1: Training classification head ---")
        history_1 = model.fit(
            train_gen,
            epochs=int(epochs/2),
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weights,
            workers=os.cpu_count() // 2 if os.cpu_count() else 1,  # Use half of available CPU cores
            use_multiprocessing=True
        )
        
        # Phase 2: Fine-tune layers from the top
        print("\n--- Phase 2: Fine-tuning with unfrozen layers ---")
        
        # Unfreeze top 30% of the base model
        for layer in base_model.layers[-int(len(base_model.layers) * 0.3):]:
            layer.trainable = True
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE/10),
            loss=FocalLoss(alpha=0.25, gamma=2.0),
            metrics=['accuracy', tf.keras.metrics.AUC(multi_label=True)]
        )
        
        history_2 = model.fit(
            train_gen,
            epochs=epochs,
            initial_epoch=int(epochs/2),
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weights,
            workers=os.cpu_count() // 2 if os.cpu_count() else 1,  # Use half of available CPU cores
            use_multiprocessing=True
        )
        
        # Combine histories
        combined_history = {}
        for key in history_1.history:
            combined_history[key] = history_1.history[key] + history_2.history[key]
        
        return combined_history
    
    except Exception as e:
        print(f"Error during training: {e}")
        raise

try:
    # Check if a model exists already, otherwise train
    if os.path.exists(MODEL_PATH):
        print(f"\nFound existing model at {MODEL_PATH}. Loading weights...")
        model.load_weights(MODEL_PATH)
        history = None
    else:
        print("\nTraining new model...")
        history = train_model(model, train_gen, val_gen, EPOCHS, callbacks, class_weights_dict)
except Exception as e:
    print(f"Training failed: {e}")
    import sys
    sys.exit(1)

# ================================
# Evaluation 
# ================================
def evaluate_model(model, test_gen, label_encoder):
    """Comprehensive model evaluation"""
    try:
        # Load best model weights if they exist
        if os.path.exists(MODEL_PATH):
            model.load_weights(MODEL_PATH)
        
        # Get predictions
        y_pred_probs = model.predict(test_gen, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = test_gen.classes
        
        # Performance metrics
        test_loss, test_acc, test_auc = model.evaluate(test_gen, verbose=1)
        print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")
        print(f"✅ Test AUC: {test_auc:.4f}")
        
        # Calculate per-class AUC
        try:
            auc_per_class = roc_auc_score(
                tf.keras.utils.to_categorical(y_true, num_classes=len(label_encoder.classes_)),
                y_pred_probs,
                average=None
            )
            print("\nAUC per class:")
            for i, cls in enumerate(label_encoder.classes_):
                print(f"  {cls}: {auc_per_class[i]:.4f}")
        except Exception as e:
            print(f"Could not calculate per-class AUC: {e}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix for better visualization
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_probs': y_pred_probs,
            'conf_matrix': conf_matrix,
            'conf_matrix_norm': conf_matrix_norm,
            'accuracy': test_acc,
            'auc': test_auc
        }
    
    except Exception as e:
        print(f"Error evaluating model: {e}")
        raise

try:
    # Evaluate the model
    evaluation = evaluate_model(model, test_gen, dataset['label_encoder'])
except Exception as e:
    print(f"Evaluation failed: {e}")
    import sys
    sys.exit(1)

# ================================
# Improved Visualization
# ================================
def plot_confusion_matrices(evaluation, label_encoder, save_dir):
    """Plot both raw and normalized confusion matrices"""
    try:
        class_names = label_encoder.classes_
        
        # Function to plot a single confusion matrix
        def plot_cm(cm, title, filename, fmt, cmap):
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt=fmt, 
                cmap=cmap,
                xticklabels=class_names,
                yticklabels=class_names,
                cbar=True
            )
            plt.xlabel('Predicted Labels', fontsize=12)
            plt.ylabel('True Labels', fontsize=12)
            plt.title(title, fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
            plt.show()
        
        # Plot raw confusion matrix
        plot_cm(
            evaluation['conf_matrix'], 
            'Confusion Matrix (Raw Counts)', 
            'confusion_matrix_raw.png', 
            'd', 
            'Blues'
        )
        
        # Plot normalized confusion matrix
        plot_cm(
            evaluation['conf_matrix_norm'], 
            'Confusion Matrix (Normalized)', 
            'confusion_matrix_norm.png', 
            '.2f', 
            'viridis'
        )
    
    except Exception as e:
        print(f"Could not plot confusion matrices: {e}")

# Plot training history
def plot_training_history(history, save_dir):
    """Plot training and validation metrics"""
    try:
        if not history:
            print("No training history to plot.")
            return
            
        metrics = ['loss', 'accuracy', 'auc']
        fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
        
        for i, metric in enumerate(metrics):
            axes[i].plot(history[metric], label=f'Training {metric}')
            axes[i].plot(history[f'val_{metric}'], label=f'Validation {metric}')
            axes[i].set_title(f'Model {metric.capitalize()}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].legend()
            axes[i].grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    except Exception as e:
        print(f"Could not plot training history: {e}")

# Plot misclassified examples
def plot_misclassified(evaluation, dataset, label_encoder, save_dir, num_examples=10):
    """Plot examples of misclassified images"""
    try:
        misclassified = np.where(evaluation['y_pred'] != evaluation['y_true'])[0]
        
        if len(misclassified) == 0:
            print("No misclassified examples found.")
            return
        
        # If more than requested examples, sample randomly
        if len(misclassified) > num_examples:
            indices = np.random.choice(misclassified, num_examples, replace=False)
        else:
            indices = misclassified
        
        # Calculate number of rows and columns for the plot
        n_cols = min(5, len(indices))
        n_rows = (len(indices) - 1) // n_cols + 1
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        test_df = dataset['test']
        
        for i, idx in enumerate(indices):
            # Get image path
            img_path = test_df.iloc[idx]['image_path']
            
            # Load and display the image
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            
            true_label = label_encoder.classes_[evaluation['y_true'][idx]]
            pred_label = label_encoder.classes_[evaluation['y_pred'][idx]]
            
            axes[i].imshow(img_array)
            axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=10)
            axes[i].axis('off')
        
        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'misclassified_examples.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    except Exception as e:
        print(f"Could not plot misclassified examples: {e}")

# ================================
# Error Analysis & Recommendations
# ================================
def analyze_error_patterns(evaluation, label_encoder):
    """Analyze error patterns to identify improvement areas"""
    try:
        conf_matrix = evaluation['conf_matrix']
        class_names = label_encoder.classes_
        
        # Calculate error rates per class
        true_counts = conf_matrix.sum(axis=1)
        correct_counts = np.diag(conf_matrix)
        error_rates = 1 - (correct_counts / true_counts)
        
        print("\nError analysis:")
        print("-" * 40)
        
        # Most problematic classes
        problem_classes = np.argsort(error_rates)[::-1]
        print("Classes with highest error rates:")
        for i in problem_classes[:3]:
            print(f"  {class_names[i]}: {error_rates[i]*100:.1f}% error rate")
        
        # Most common misclassifications
        np.fill_diagonal(conf_matrix, 0)  # Zero out the diagonal to focus on errors
        error_pairs = []
        
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and conf_matrix[i, j] > 0:
                    error_pairs.append((i, j, conf_matrix[i, j]))
        
        error_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print("\nMost common misclassifications:")
        for true_idx, pred_idx, count in error_pairs[:5]:
            print(f"  {class_names[true_idx]} → {class_names[pred_idx]}: {count} instances " +
                  f"({count/true_counts[true_idx]*100:.1f}% of {class_names[true_idx]})")
        
        print("\nRecommendations for improvement:")
        print("-" * 40)
        print("1. Consider collecting more data for the most problematic classes")
        print("2. Implement specialized augmentation techniques for commonly confused classes")
        print("3. Consider a hierarchical classification approach for similar classes")
        print("4. Explore ensemble methods combining multiple model architectures")
        print("5. Implement regular clinical validation to ensure accuracy on new cases")
    
    except Exception as e:
        print(f"Error in analysis: {e}")

# ================================
# Production-Ready Model Export
# ================================
def export_production_model(model, save_path, label_encoder):
    """Export the model for production use in TF SavedModel format"""
    try:
        print("\nExporting model for production...")
        
        # Create a preprocessing function
        @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.uint8)])
        def preprocess_and_predict(image):
            # Resize
            image = tf.image.resize(image, IMG_SIZE)
            # Normalize
            image = tf.cast(image, tf.float32) / 255.0
            # Predict
            predictions = model(image, training=False)
            return {
                'class_probabilities': predictions,
                'predicted_class_idx': tf.argmax(predictions, axis=1),
                'predicted_class': tf.gather(
                    tf.constant(label_encoder.classes_, dtype=tf.string),
                    tf.argmax(predictions, axis=1)
                )
            }
        
        # Create a serving model
        serving_model = tf.keras.Sequential([model])
        
        # Create signatures
        signatures = {
            'serving_default': preprocess_and_predict
        }
        
        # Save the model
        tf.saved_model.save(
            serving_model,
            save_path,
            signatures=signatures
        )
        
        # Save label encoder classes
        np.save(os.path.join(os.path.dirname(save_path), 'label_classes.npy'), label_encoder.classes_)
        
        print(f"✅ Model successfully exported to {save_path}")
        print(f"Label classes saved to {os.path.join(os.path.dirname(save_path), 'label_classes.npy')}")
        
        return True
    
    except Exception as e:
        print(f"Error exporting model: {e}")
        return False

# ================================
# Create a prediction function
# ================================
def create_prediction_function(model_path):
    """Create a standalone prediction function for single images"""
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        label_classes = np.load(os.path.join(os.path.dirname(model_path), 'label_classes.npy'))
        
        def predict_image(image_path):
            """Predict skin condition from an image file"""
            try:
                # Load and preprocess image
                img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                
                # Predict
                predictions = model.predict(img_array)
                predicted_class_idx = np.argmax(predictions[0])
                predicted_class = label_classes[predicted_class_idx]
                confidence = float(predictions[0][predicted_class_idx])
                
                # Get top 3 predictions
                top_indices = np.argsort(predictions[0])[::-1][:3]
                top_3_predictions = [
                    {
                        'class': label_classes[idx],
                        'confidence': float(predictions[0][idx])
                    }
                    for idx in top_indices
                ]
                
                return {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'top_3_predictions': top_3_predictions
                }
            
            except Exception as e:
                return {'error': str(e)}
        
        return predict_image
    
    except Exception as e:
        print(f"Error creating prediction function: {e}")
        return None

# ================================
# Main execution
# ================================
if __name__ == "__main__":
    try:
        # Create visualizations if available
        if evaluation:
            plot_confusion_matrices(evaluation, dataset['label_encoder'], RESULTS_DIR)
            if history:
                plot_training_history(history, RESULTS_DIR)
            plot_misclassified(evaluation, dataset, dataset['label_encoder'], RESULTS_DIR)
            analyze_error_patterns(evaluation, dataset['label_encoder'])
        
        # Export model for production
        export_success = export_production_model(model, SAVEDMODEL_PATH, dataset['label_encoder'])
        
        if export_success:
            # Test production model with a sample image
            predict_fn = create_prediction_function(SAVEDMODEL_PATH)
            if predict_fn and len(dataset['test']) > 0:
                sample_image = dataset['test'].iloc[0]['image_path']
                print("\nTesting production model with a sample image:")
                result = predict_fn(sample_image)
                print(f"Prediction: {result}")
        
        print("\nModel training and evaluation complete! Results saved to:", RESULTS_DIR)
        print("Production model saved to:", SAVEDMODEL_PATH)
    
    except Exception as e:
        print(f"An error occurred during execution: {e}")