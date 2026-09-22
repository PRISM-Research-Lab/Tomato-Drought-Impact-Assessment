import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


############################################
# 1. Load Dataset
############################################
def load_dataset(file_path):

    df = pd.read_csv(file_path)

    print("Dataset Shape:", df.shape)
    print(df.head())

    return df


############################################
# 2. Dataset Summary
############################################
def dataset_summary(df):

    print("\nDataset Info")
    print(df.info())

    print("\nClass Distribution")
    print(df['y'].value_counts())


############################################
# 3. Class Distribution Plot
############################################
def plot_class_distribution(df):

    plt.figure(figsize=(6,4))
    sns.countplot(x='y', data=df)

    plt.title("Class Distribution")
    plt.xlabel("Class (0 = Normal, 1 = Drought)")
    plt.ylabel("Number of Samples")

    plt.show()


############################################
# 4. Mean Spectral Curve
############################################
def plot_mean_spectrum(df):

    import numpy as np
    import matplotlib.pyplot as plt

    # spectral columns
    spectral_cols = df.columns[:-1]

    # wavelength axis
    wavelengths = np.linspace(348, 1052, len(spectral_cols))

    # separate classes
    normal = df[df['y'] == 0][spectral_cols]
    drought = df[df['y'] == 1][spectral_cols]

    # mean spectrum
    mean_normal = normal.mean()
    mean_drought = drought.mean()

    plt.figure(figsize=(10,5))

    plt.plot(wavelengths, mean_normal, label="Normal")
    plt.plot(wavelengths, mean_drought, label="Drought")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title("Mean Spectral Signature")
    plt.legend()

    plt.show()

############################################
# 5. Spectral Heatmap
############################################
def plot_heatmap(df):

    X = df.drop(columns=['y'])

    plt.figure(figsize=(10,6))
    sns.heatmap(X, cmap="viridis")

    plt.title("Spectral Data Heatmap")
    plt.xlabel("Spectral Bands")
    plt.ylabel("Samples")

    plt.show()

def plot_mean_spectrum_with_std(df):

    import numpy as np
    import matplotlib.pyplot as plt

    # Separate spectral features
    spectral_cols = df.columns[:-1]

    # Wavelength axis (211 bands from 348–1052 nm)
    wavelengths = np.linspace(348, 1052, len(spectral_cols))

    # Split classes
    normal = df[df['y'] == 0][spectral_cols]
    drought = df[df['y'] == 1][spectral_cols]

    # Mean spectra
    mean_normal = normal.mean()
    mean_drought = drought.mean()

    # Standard deviation
    std_normal = normal.std()
    std_drought = drought.std()

    plt.figure(figsize=(10,5))

    # Plot mean curves
    plt.plot(wavelengths, mean_normal, label="Normal", color="brown")
    plt.plot(wavelengths, mean_drought, label="Drought", color="green")

    # Add shaded variance
    plt.fill_between(wavelengths,
                     mean_normal - std_normal,
                     mean_normal + std_normal,
                     color="brown", alpha=0.2)

    plt.fill_between(wavelengths,
                     mean_drought - std_drought,
                     mean_drought + std_drought,
                     color="green", alpha=0.2)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title("Mean Vis–NIR Spectral Reflectance of Tomato Seedlings")
    plt.legend()
    plt.grid(True)

    plt.show()

############################################
# 6. PCA Visualization
############################################
def pca_visualization(df):

    X = df.drop(columns=['y'])
    y = df['y']

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(7,5))

    plt.scatter(
        X_pca[y==0,0],
        X_pca[y==0,1],
        label="Normal"
    )

    plt.scatter(
        X_pca[y==1,0],
        X_pca[y==1,1],
        label="Drought"
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Visualization")

    plt.legend()
    plt.show()


############################################
# 7. Random Forest Feature Importance
############################################
def feature_importance(df):

    X = df.drop(columns=['y'])
    y = df['y']

    rf = RandomForestClassifier(n_estimators=200)

    rf.fit(X, y)

    importance = rf.feature_importances_

    indices = np.argsort(importance)[::-1]

    top_features = indices[:15]

    plt.figure(figsize=(8,5))

    plt.bar(range(len(top_features)), importance[top_features])

    plt.xticks(range(len(top_features)), top_features)

    plt.title("Top Feature Importance (Random Forest)")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")

    plt.show()

    return indices

def plot_top_wavelengths_with_classes(df, indices):
    import numpy as np
    import matplotlib.pyplot as plt

    # Spectral columns
    spectral_cols = df.drop(columns=['y']).columns
    wavelengths = np.linspace(348, 1052, len(spectral_cols))

    # Split classes
    normal = df[df['y'] == 0][spectral_cols]
    drought = df[df['y'] == 1][spectral_cols]

    # Mean spectra
    mean_normal = normal.mean()
    mean_drought = drought.mean()

    # Top features indices
    top = indices[:10]

    plt.figure(figsize=(12,5))

    # Plot mean spectra
    plt.plot(wavelengths, mean_normal, label="Normal", color="brown")
    plt.plot(wavelengths, mean_drought, label="Drought", color="green")

    # Highlight top wavelengths
    for i in top:
        plt.axvline(wavelengths[i], color='red', linestyle='--', alpha=0.7)

    plt.title("Mean Spectra with Top Random Forest Wavelengths")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.grid(True)
    plt.show()

############################################
# 8. Highlight Important Wavelengths
############################################
def plot_top_wavelengths(df, indices):
    spectral_cols = df.drop(columns=['y']).columns
    wavelengths = np.linspace(348, 1052, len(spectral_cols))

    top = indices[:10]

    plt.figure(figsize=(10,4))
    plt.plot(wavelengths, df.drop(columns=['y']).mean(), label="Mean Spectrum")

    for i in top:
        plt.axvline(wavelengths[i], color='red', linestyle='--')

    plt.title("Important Spectral Wavelengths")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.show()


############################################
# 9. Train Simple Model + Confusion Matrix
############################################
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score, f1_score

def model_evaluation_with_metrics(df):
    X = df.drop(columns=['y'])
    y = df['y']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    # Predict
    y_pred = rf.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Drought"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # Precision, Recall, F1-score (macro average)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-score (macro): {f1:.4f}\n")

    # Full classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Drought"]))


def cnn_1d_model_evaluation(df, epochs=100, batch_size=16, validation_split=0.2):

    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, \
        accuracy_score, precision_score, recall_score, f1_score

    # --------------------------
    # 1. Prepare Data
    # --------------------------
    X = df.drop(columns=['y']).values
    y = df['y'].values

    X = X[..., np.newaxis]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    input_shape = (X.shape[1], 1)

    # --------------------------
    # 2. Hybrid CNN Model
    # --------------------------
    inputs = tf.keras.Input(shape=input_shape)

    # ---- CNN branch ----
    x1 = tf.keras.layers.Conv1D(32, 5, activation='relu')(inputs)
    x1 = tf.keras.layers.MaxPooling1D(2)(x1)
    x1 = tf.keras.layers.Conv1D(64, 5, activation='relu')(x1)
    x1 = tf.keras.layers.MaxPooling1D(2)(x1)
    x1 = tf.keras.layers.Flatten()(x1)

    # ---- Simple dense branch (raw spectral info) ----
    x2 = tf.keras.layers.Flatten()(inputs)
    x2 = tf.keras.layers.Dense(32, activation='relu')(x2)

    # ---- Fusion ----
    x = tf.keras.layers.Concatenate()([x1, x2])

    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # --------------------------
    # 3. Train
    # --------------------------
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )

    # --------------------------
    # 4. Plot
    # --------------------------
    # plt.figure(figsize=(10, 4))
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(history.history['accuracy'], label='Train Acc')
    # plt.plot(history.history['val_accuracy'], label='Val Acc')
    # # plt.title('Accuracy', fontsize=18)
    # plt.legend(fontsize=18)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # plt.grid()
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(history.history['loss'], label='Train Loss')
    # plt.plot(history.history['val_loss'], label='Val Loss')
    # # plt.title('Loss', fontsize=18)
    # plt.legend(fontsize=18)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # plt.grid()
    # plt.tight_layout()
    # plt.show()
    plt.figure(figsize=(10, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid()

    plt.tight_layout()
    plt.show()
    # --------------------------
    # 5. Evaluate
    # --------------------------
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='macro'):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred, average='macro'):.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Normal", "Drought"]))

    # cm = confusion_matrix(y_test, y_pred)
    # ConfusionMatrixDisplay(cm,
    #                        display_labels=["Normal", "Drought"]).plot(cmap="Blues")
    # plt.title("Confusion Matrix")
    # plt.show()

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm,
                                  display_labels=["Normal", "Drought"])
    disp.plot(cmap="Blues")

    # plt.title("Confusion Matrix", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # increase cell values
    for text in disp.text_.ravel():
        text.set_fontsize(18)

    # increase axis label fonts
    disp.ax_.set_xlabel("Predicted label", fontsize=18)
    disp.ax_.set_ylabel("True label", fontsize=18)

    plt.show()

    return model, y_pred_prob,y_pred,y_test

def cnn_1d_resnet_model_evaluation(df, epochs=100, batch_size=16, validation_split=0.2):
    """
    Train and evaluate a 1D-ResNet on hyperspectral tomato data.

    Parameters:
    - df: pandas DataFrame with spectral features and 'y' as label (0=Normal, 1=Drought)
    - epochs: number of training epochs
    - batch_size: batch size for training
    - validation_split: fraction of training data for validation
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, \
        precision_score, recall_score, f1_score

    # --------------------------
    # 1. Prepare Data
    # --------------------------
    X = df.drop(columns=['y']).values
    y = df['y'].values
    X = X[..., np.newaxis]  # shape (num_samples, num_bands, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --------------------------
    # 2. Build 1D ResNet-style Model
    # --------------------------
    inputs = tf.keras.Input(shape=(X.shape[1], 1))

    # First conv layer
    x = tf.keras.layers.Conv1D(32, kernel_size=5, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Residual block 1
    shortcut = x
    x = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(32, 5, padding='same')(x)
    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Residual block 2
    shortcut = x
    x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(64, 5, padding='same')(x)
    # if shapes differ, project shortcut
    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = tf.keras.layers.Conv1D(64, 1, padding='same')(shortcut)
    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Flatten + dense
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # --------------------------
    # 3. Train Model
    # --------------------------
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )

    # --------------------------
    # 4. Plot Training History
    # --------------------------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --------------------------
    # 5. Evaluate on Test Set
    # --------------------------
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-score (macro): {f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Drought"]))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Drought"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    return model

def cnn_1d_resnet_deeper(df, epochs=100, batch_size=16, validation_split=0.2):
    """
    Train and evaluate a deeper 1D-ResNet on hyperspectral tomato data.
    Uses all features, no feature selection.

    Parameters:
    - df: pandas DataFrame with spectral features and 'y' as label (0=Normal, 1=Drought)
    - epochs: number of training epochs
    - batch_size: batch size for training
    - validation_split: fraction of training data for validation
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, \
        precision_score, recall_score, f1_score

    # --------------------------
    # 1. Prepare Data
    # --------------------------
    X = df.drop(columns=['y']).values
    y = df['y'].values
    X = X[..., np.newaxis]  # shape (samples, features, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --------------------------
    # 2. Build Deeper 1D ResNet
    # --------------------------
    inputs = tf.keras.Input(shape=(X.shape[1], 1))

    # First conv layer
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Residual block 1
    shortcut = x
    x = tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(32, 3, padding='same')(x)
    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Residual block 2
    shortcut = x
    x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(64, 3, padding='same')(x)
    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = tf.keras.layers.Conv1D(64, 1, padding='same')(shortcut)
    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Residual block 3 (deeper)
    shortcut = x
    x = tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(128, 3, padding='same')(x)
    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = tf.keras.layers.Conv1D(128, 1, padding='same')(shortcut)
    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Flatten + dense
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # --------------------------
    # 3. Train Model
    # --------------------------
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )

    # --------------------------
    # 4. Plot Training History
    # --------------------------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --------------------------
    # 5. Evaluate on Test Set
    # --------------------------
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-score (macro): {f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Drought"]))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Drought"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    return model
def cnn_1d_resnet_with_manual_features(df, manual_features_df, epochs=100, batch_size=16, validation_split=0.2):
    """
    Train and evaluate a 1D-ResNet on hyperspectral tomato data with extra manual features.

    Parameters:
    - df: pandas DataFrame with spectral features and 'y' as label (0=Normal, 1=Drought)
    - manual_features_df: pandas DataFrame with additional engineered features (must match df rows)
    - epochs: number of training epochs
    - batch_size: batch size
    - validation_split: fraction of training data for validation
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, \
        precision_score, recall_score, f1_score

    # --------------------------
    # 1. Prepare Data
    # --------------------------
    X_spectral = df.drop(columns=['y']).values
    X_manual = manual_features_df.values  # shape (num_samples, num_manual_features)
    y = df['y'].values

    # reshape spectral for 1D-CNN: (samples, timesteps, features)
    X_spectral = X_spectral[..., np.newaxis]  # (num_samples, num_bands, 1)

    # train/test split (stratified)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(X_spectral, y):
        X_spectral_train, X_spectral_test = X_spectral[train_idx], X_spectral[test_idx]
        X_manual_train, X_manual_test = X_manual[train_idx], X_manual[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    # --------------------------
    # 2. Build 1D ResNet + Manual Features Model
    # --------------------------
    spectral_input = tf.keras.Input(shape=(X_spectral.shape[1], 1), name='spectral_input')
    manual_input = tf.keras.Input(shape=(X_manual.shape[1],), name='manual_input')

    # First conv layer
    x = tf.keras.layers.Conv1D(32, kernel_size=5, padding='same', activation='relu')(spectral_input)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Residual block 1
    shortcut = x
    x = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(32, 5, padding='same')(x)
    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Residual block 2
    shortcut = x
    x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(64, 5, padding='same')(x)
    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = tf.keras.layers.Conv1D(64, 1, padding='same')(shortcut)
    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Flatten spectral features
    x = tf.keras.layers.Flatten()(x)

    # Concatenate manual features
    x = tf.keras.layers.Concatenate()([x, manual_input])

    # Dense layers
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[spectral_input, manual_input], outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # --------------------------
    # 3. Train Model
    # --------------------------
    history = model.fit(
        {'spectral_input': X_spectral_train, 'manual_input': X_manual_train},
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )

    # --------------------------
    # 4. Plot Training History
    # --------------------------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --------------------------
    # 5. Evaluate on Test Set
    # --------------------------
    y_pred_prob = model.predict({'spectral_input': X_spectral_test, 'manual_input': X_manual_test})
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-score (macro): {f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Drought"]))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Drought"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    return model

def cnn_1d_resnet_with_selected_features(df, top_n_features=30, epochs=100, batch_size=16, validation_split=0.2):
    """
    Train and evaluate a 1D-ResNet using only the top N important spectral features.

    Parameters:
    - df: pandas DataFrame with spectral features and 'y' as label (0=Normal, 1=Drought)
    - top_n_features: number of top features to select based on Random Forest importance
    - epochs: number of training epochs
    - batch_size: batch size for training
    - validation_split: fraction of training data for validation
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, \
        precision_score, recall_score, f1_score

    # --------------------------
    # 1. Select Top Features
    # --------------------------
    X_all = df.drop(columns=['y'])
    y = df['y'].values

    # Random Forest to estimate feature importance
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_all, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n_features]

    print(f"Selected top {top_n_features} features based on importance indices: {indices}")

    X_selected = X_all.iloc[:, indices].values
    X_selected = X_selected[..., np.newaxis]  # shape (samples, features, 1)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )

    # --------------------------
    # 2. Build 1D ResNet-style Model
    # --------------------------
    inputs = tf.keras.Input(shape=(X_selected.shape[1], 1))

    # First conv layer
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Residual block 1
    shortcut = x
    x = tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(32, 3, padding='same')(x)
    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Residual block 2
    shortcut = x
    x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(64, 3, padding='same')(x)
    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = tf.keras.layers.Conv1D(64, 1, padding='same')(shortcut)
    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Flatten + Dense
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # --------------------------
    # 3. Train Model
    # --------------------------
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )

    # --------------------------
    # 4. Plot Training History
    # --------------------------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --------------------------
    # 5. Evaluate on Test Set
    # --------------------------
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-score (macro): {f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Drought"]))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Drought"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    return model

    # --------------------------
    # 6. Evaluate on Test Set
    # --------------------------
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-score (macro): {f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Drought"]))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Drought"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    print(f"Best model saved to: {save_path}")
    return model


def generate_manual_features(df):
    """
    Generate manual features from 211 Vis-NIR spectral bands.
    Assumes df has spectral columns first and 'y' as the last column.

    Returns:
        manual_df: DataFrame with manual features
    """
    spectral_cols = df.columns[:-1]  # all spectral bands
    X = df[spectral_cols]

    manual_features = pd.DataFrame(index=df.index)

    # 1. Statistical features
    manual_features['mean_reflectance'] = X.mean(axis=1)
    manual_features['std_reflectance'] = X.std(axis=1)
    manual_features['max_reflectance'] = X.max(axis=1)
    manual_features['min_reflectance'] = X.min(axis=1)
    manual_features['range_reflectance'] = X.max(axis=1) - X.min(axis=1)
    manual_features['skew_reflectance'] = X.skew(axis=1)
    manual_features['kurtosis_reflectance'] = X.kurtosis(axis=1)

    # 2. Simple vegetation indices (example)
    # NDVI-like: (NIR - Red) / (NIR + Red)
    # Assuming first band ~348 nm (Red), last band ~1052 nm (NIR)
    manual_features['nir_red_ratio'] = (X.iloc[:, -1] - X.iloc[:, 0]) / (X.iloc[:, -1] + X.iloc[:, 0] + 1e-6)

    # 3. Spectral slopes
    # Slope between first and middle band
    mid_idx = X.shape[1] // 2
    manual_features['slope_first_mid'] = (X.iloc[:, mid_idx] - X.iloc[:, 0]) / (mid_idx)

    # Slope between middle and last band
    manual_features['slope_mid_last'] = (X.iloc[:, -1] - X.iloc[:, mid_idx]) / (X.shape[1] - mid_idx - 1)

    # 4. Area under curve
    manual_features['auc'] = X.sum(axis=1)

    return manual_features

def cnn_1d_resnet_balanced(df, epochs=100, batch_size=16, validation_split=0.2):
    """
    Train and evaluate a 1D-ResNet on hyperspectral tomato data,
    balancing classes automatically using undersampling.

    Parameters:
    - df: pandas DataFrame with spectral features and 'y' as label (0=Normal, 1=Drought)
    - epochs: number of training epochs
    - batch_size: batch size for training
    - validation_split: fraction of training data for validation
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, \
        precision_score, recall_score, f1_score
    import pandas as pd

    # --------------------------
    # 1. Balance the dataset
    # --------------------------
    df_minority = df[df['y'] == 1]
    df_majority = df[df['y'] == 0]

    df_majority_downsampled = df_majority.sample(len(df_minority), random_state=42)

    # Use pd.concat instead of append
    df_balanced = pd.concat([df_minority, df_majority_downsampled], axis=0).sample(frac=1, random_state=42)

    # --------------------------
    # 2. Prepare Data
    # --------------------------
    X = df_balanced.drop(columns=['y']).values
    y = df_balanced['y'].values
    X = X[..., np.newaxis]  # shape (num_samples, num_bands, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --------------------------
    # 3. Build 1D ResNet-style Model
    # --------------------------
    inputs = tf.keras.Input(shape=(X.shape[1], 1))
    x = tf.keras.layers.Conv1D(32, kernel_size=5, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Residual block 1
    shortcut = x
    x = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(32, 5, padding='same')(x)
    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Residual block 2
    shortcut = x
    x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(64, 5, padding='same')(x)
    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = tf.keras.layers.Conv1D(64, 1, padding='same')(shortcut)
    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Flatten + Dense
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # --------------------------
    # 4. Train Model
    # --------------------------
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )

    # --------------------------
    # 5. Plot Training History
    # --------------------------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --------------------------
    # 6. Evaluate on Test Set
    # --------------------------
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-score (macro): {f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Drought"]))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Drought"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    return model

def cnn_1d_resnet_hybrid(df, manual_features, epochs=100, batch_size=16, validation_split=0.2):
    """
    Train and evaluate a hybrid model: 1D-ResNet (original) + manual features.
    Original 1D-ResNet architecture is preserved.

    Parameters:
    - df: pandas DataFrame with spectral features and 'y' as label (0=Normal, 1=Drought)
    - manual_features: pandas DataFrame with manual features (same rows as df)
    - epochs: number of training epochs
    - batch_size: batch size
    - validation_split: fraction for validation
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, \
        precision_score, recall_score, f1_score

    # --------------------------
    # 1. Prepare Data
    # --------------------------
    X_spectral = df.drop(columns=['y']).values
    X_spectral = X_spectral[..., np.newaxis]  # shape (samples, bands, 1)
    X_manual = manual_features.values  # manual features shape (samples, features)
    y = df['y'].values

    # Train/test split
    from sklearn.model_selection import train_test_split
    Xs_train, Xs_test, Xm_train, Xm_test, y_train, y_test = train_test_split(
        X_spectral, X_manual, y, test_size=0.2, random_state=42
    )

    # --------------------------
    # 2. Build Original 1D-ResNet branch
    # --------------------------
    spectral_inputs = tf.keras.Input(shape=(X_spectral.shape[1], 1), name='spectral_input')

    # Original 1D-ResNet architecture
    x = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu')(spectral_inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    shortcut = x
    x = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(32, 5, padding='same')(x)
    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    shortcut = x
    x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(64, 5, padding='same')(x)
    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = tf.keras.layers.Conv1D(64, 1, padding='same')(shortcut)
    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    spectral_output = x

    # --------------------------
    # 3. Manual Features branch
    # --------------------------
    manual_inputs = tf.keras.Input(shape=(X_manual.shape[1],), name='manual_input')
    m = tf.keras.layers.Dense(32, activation='relu')(manual_inputs)
    m = tf.keras.layers.Dense(16, activation='relu')(m)
    manual_output = m

    # --------------------------
    # 4. Combine both branches
    # --------------------------
    combined = tf.keras.layers.Concatenate()([spectral_output, manual_output])
    z = tf.keras.layers.Dense(64, activation='relu')(combined)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(z)

    model = tf.keras.Model(inputs=[spectral_inputs, manual_inputs], outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # --------------------------
    # 5. Train the hybrid model
    # --------------------------
    history = model.fit(
        [Xs_train, Xm_train], y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )

    # --------------------------
    # 6. Evaluate
    # --------------------------
    y_pred_prob = model.predict([Xs_test, Xm_test])
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-score (macro): {f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Drought"]))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Drought"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    return model

def cnn_1d_resnet_hybrid_best(df, manual_features, epochs=100, batch_size=16, validation_split=0.2):
    """
    Train and evaluate a hybrid model: 1D-ResNet (original) + manual features.
    Original 1D-ResNet architecture is preserved.
    Best validation weights are automatically restored.

    Parameters:
    - df: pandas DataFrame with spectral features and 'y' as label (0=Normal, 1=Drought)
    - manual_features: pandas DataFrame with manual features (same rows as df)
    - epochs: number of training epochs
    - batch_size: batch size
    - validation_split: fraction for validation
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, \
        precision_score, recall_score, f1_score

    # --------------------------
    # 1. Prepare Data
    # --------------------------
    X_spectral = df.drop(columns=['y']).values
    X_spectral = X_spectral[..., np.newaxis]  # shape (samples, bands, 1)
    X_manual = manual_features.values
    y = df['y'].values

    Xs_train, Xs_test, Xm_train, Xm_test, y_train, y_test = train_test_split(
        X_spectral, X_manual, y, test_size=0.2, random_state=42
    )

    # --------------------------
    # 2. Build 1D-ResNet branch
    # --------------------------
    spectral_inputs = tf.keras.Input(shape=(X_spectral.shape[1], 1), name='spectral_input')

    # Original 1D-ResNet architecture
    x = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu')(spectral_inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    shortcut = x
    x = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(32, 5, padding='same')(x)
    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    shortcut = x
    x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(64, 5, padding='same')(x)
    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = tf.keras.layers.Conv1D(64, 1, padding='same')(shortcut)
    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    spectral_output = x

    # --------------------------
    # 3. Manual Features branch
    # --------------------------
    manual_inputs = tf.keras.Input(shape=(X_manual.shape[1],), name='manual_input')
    m = tf.keras.layers.Dense(32, activation='relu')(manual_inputs)
    m = tf.keras.layers.Dense(16, activation='relu')(m)
    manual_output = m

    # --------------------------
    # 4. Combine both branches
    # --------------------------
    combined = tf.keras.layers.Concatenate()([spectral_output, manual_output])
    z = tf.keras.layers.Dense(64, activation='relu')(combined)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(z)

    model = tf.keras.Model(inputs=[spectral_inputs, manual_inputs], outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # --------------------------
    # 5. Callbacks: EarlyStopping (best val weights restored)
    # --------------------------
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )

    # --------------------------
    # 6. Train Model
    # --------------------------
    history = model.fit(
        [Xs_train, Xm_train], y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
        callbacks=[early_stop]
    )

    # --------------------------
    # 7. Plot Training History
    # --------------------------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --------------------------
    # 8. Evaluate on Test Set
    # --------------------------
    y_pred_prob = model.predict([Xs_test, Xm_test])
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-score (macro): {f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Drought"]))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Drought"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    return model

def cnn_1d_resnet_hybrid_best2(df, manual_features, selected_wavelengths=None,
                              epochs=100, batch_size=16, validation_split=0.2):
    """
    Train and evaluate a hybrid model: 1D-ResNet (original) + manual features.
    Original 1D-ResNet architecture is preserved.
    Best validation weights are automatically restored.

    Parameters:
    - df: pandas DataFrame with spectral features and 'y' as label (0=Normal, 1=Drought)
    - manual_features: pandas DataFrame with manual features (same rows as df)
    - selected_wavelengths: list of column names or indices to use; if None, use all spectral features
    - epochs: number of training epochs
    - batch_size: batch size
    - validation_split: fraction for validation
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, classification_report,
                                 accuracy_score, precision_score, recall_score, f1_score)

    # --------------------------
    # 1. Prepare Data
    # --------------------------
    if selected_wavelengths is not None:
        X_spectral = df[selected_wavelengths].values
    else:
        X_spectral = df.drop(columns=['y']).values

    X_spectral = X_spectral[..., np.newaxis]  # shape (samples, bands, 1)
    X_manual = manual_features.values
    y = df['y'].values

    Xs_train, Xs_test, Xm_train, Xm_test, y_train, y_test = train_test_split(
        X_spectral, X_manual, y, test_size=0.2, random_state=42
    )

    # --------------------------
    # 2. Build 1D-ResNet branch
    # --------------------------
    spectral_inputs = tf.keras.Input(shape=(X_spectral.shape[1], 1), name='spectral_input')

    # Original 1D-ResNet architecture
    x = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu')(spectral_inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    shortcut = x
    x = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(32, 5, padding='same')(x)
    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    shortcut = x
    x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(64, 5, padding='same')(x)
    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = tf.keras.layers.Conv1D(64, 1, padding='same')(shortcut)
    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    spectral_output = x

    # --------------------------
    # 3. Manual Features branch
    # --------------------------
    manual_inputs = tf.keras.Input(shape=(X_manual.shape[1],), name='manual_input')
    m = tf.keras.layers.Dense(32, activation='relu')(manual_inputs)
    m = tf.keras.layers.Dense(16, activation='relu')(m)
    manual_output = m

    # --------------------------
    # 4. Combine both branches
    # --------------------------
    combined = tf.keras.layers.Concatenate()([spectral_output, manual_output])
    z = tf.keras.layers.Dense(64, activation='relu')(combined)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(z)

    model = tf.keras.Model(inputs=[spectral_inputs, manual_inputs], outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # --------------------------
    # 5. Callbacks: EarlyStopping (best val weights restored)
    # --------------------------
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )

    # --------------------------
    # 6. Train Model
    # --------------------------
    history = model.fit(
        [Xs_train, Xm_train], y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
        callbacks=[early_stop]
    )

    # --------------------------
    # 7. Plot Training History
    # --------------------------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --------------------------
    # 8. Evaluate on Test Set
    # --------------------------
    y_pred_prob = model.predict([Xs_test, Xm_test])
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)  # Recall = Sensitivity
    specificity = tn / (tn + fp)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"F1-score:    {f1:.4f}\n")

    # Optional: per-class metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Drought"]))

    # Confusion matrix plot
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Drought"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    return model

def cnn_1d_resnet_hybrid_best_balanced(df, manual_features, epochs=100, batch_size=16, validation_split=0.2):
    """
    Train and evaluate a hybrid 1D-ResNet + manual features model on imbalanced data.
    Downsamples the majority class to match the minority class.
    Best validation weights are automatically restored.

    Parameters:
    - df: pandas DataFrame with spectral features and 'y' as label (0=Normal, 1=Drought)
    - manual_features: pandas DataFrame with manual features (same rows as df)
    - epochs: number of training epochs
    - batch_size: batch size
    - validation_split: fraction for validation
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, \
        precision_score, recall_score, f1_score
    from sklearn.utils import resample
    import pandas as pd

    # --------------------------
    # 1. Balance the dataset
    # --------------------------
    df_combined = df.copy()
    df_combined['manual_idx'] = manual_features.index  # temporary index for manual features

    # Identify minority and majority classes
    class_counts = df_combined['y'].value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    n_minority = class_counts.min()

    df_minority = df_combined[df_combined['y'] == minority_class]
    df_majority = df_combined[df_combined['y'] == majority_class]

    # Downsample majority class
    df_majority_downsampled = resample(df_majority,
                                       replace=False,
                                       n_samples=n_minority,
                                       random_state=42)
    df_balanced = pd.concat([df_minority, df_majority_downsampled]).sample(frac=1, random_state=42)  # shuffle

    # Split back manual features
    X_spectral = df_balanced.drop(columns=['y', 'manual_idx']).values
    manual_indices = df_balanced['manual_idx'].values
    X_manual = manual_features.loc[manual_indices].values
    y = df_balanced['y'].values

    X_spectral = X_spectral[..., np.newaxis]  # shape (samples, bands, 1)

    # --------------------------
    # 2. Train/test split
    # --------------------------
    Xs_train, Xs_test, Xm_train, Xm_test, y_train, y_test = train_test_split(
        X_spectral, X_manual, y, test_size=0.2, random_state=42
    )

    # --------------------------
    # 3. Build 1D-ResNet branch
    # --------------------------
    spectral_inputs = tf.keras.Input(shape=(X_spectral.shape[1], 1), name='spectral_input')

    # Original 1D-ResNet architecture
    x = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu')(spectral_inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    shortcut = x
    x = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(32, 5, padding='same')(x)
    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    shortcut = x
    x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(64, 5, padding='same')(x)
    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = tf.keras.layers.Conv1D(64, 1, padding='same')(shortcut)
    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    spectral_output = x

    # --------------------------
    # 4. Manual Features branch
    # --------------------------
    manual_inputs = tf.keras.Input(shape=(X_manual.shape[1],), name='manual_input')
    m = tf.keras.layers.Dense(32, activation='relu')(manual_inputs)
    m = tf.keras.layers.Dense(16, activation='relu')(m)
    manual_output = m

    # --------------------------
    # 5. Combine branches
    # --------------------------
    combined = tf.keras.layers.Concatenate()([spectral_output, manual_output])
    z = tf.keras.layers.Dense(64, activation='relu')(combined)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(z)

    model = tf.keras.Model(inputs=[spectral_inputs, manual_inputs], outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # --------------------------
    # 6. Callbacks
    # --------------------------
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )

    # --------------------------
    # 7. Train
    # --------------------------
    history = model.fit(
        [Xs_train, Xm_train], y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
        callbacks=[early_stop]
    )

    # --------------------------
    # 8. Plot History
    # --------------------------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --------------------------
    # 9. Evaluate
    # --------------------------
    y_pred_prob = model.predict([Xs_test, Xm_test])
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-score (macro): {f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Drought"]))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Drought"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    return model

def cnn_1d_resnet_evaluation_protocol(df, epochs=100, batch_size=16, n_runs=10):
    """
    Train and evaluate a 1D-ResNet on hyperspectral tomato data
    following the 80/20 split + 10-fold CV + repeated runs protocol.

    Parameters:
    - df: pandas DataFrame with spectral features and 'y' as label (0=Normal, 1=Drought)
    - epochs: number of training epochs
    - batch_size: batch size
    - n_runs: number of repeated experiments (for mean ± std)
    """
    import numpy as np
    import tensorflow as tf
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    X = df.drop(columns=['y']).values
    y = df['y'].values
    X = X[..., np.newaxis]  # shape (samples, bands, 1)

    # Store metrics for all runs
    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    for run in range(n_runs):
        print(f"\n=== Run {run+1}/{n_runs} ===")

        # 1️⃣ Stratified 80/20 split
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=run
        )

        # 2️⃣ 10-fold CV on training data
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=run)
        val_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
            X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
            y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

            # Build model
            inputs = tf.keras.Input(shape=(X.shape[1], 1))
            x = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu')(inputs)
            x = tf.keras.layers.MaxPooling1D(2)(x)

            # Residual block 1
            shortcut = x
            x = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu')(x)
            x = tf.keras.layers.Conv1D(32, 5, padding='same')(x)
            x = tf.keras.layers.Add()([shortcut, x])
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.MaxPooling1D(2)(x)

            # Residual block 2
            shortcut = x
            x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
            x = tf.keras.layers.Conv1D(64, 5, padding='same')(x)
            if shortcut.shape[-1] != x.shape[-1]:
                shortcut = tf.keras.layers.Conv1D(64, 1, padding='same')(shortcut)
            x = tf.keras.layers.Add()([shortcut, x])
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.MaxPooling1D(2)(x)

            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # Train fold
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )

            # Record validation accuracy
            val_acc = model.evaluate(X_val, y_val, verbose=0)[1]
            val_scores.append(val_acc)

        # Optionally, you could choose best hyperparameters here using val_scores
        # For simplicity, we just proceed to train on full training set

        # 3️⃣ Train final model on full 80% training set
        inputs = tf.keras.Input(shape=(X.shape[1], 1))
        x = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling1D(2)(x)

        shortcut = x
        x = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv1D(32, 5, padding='same')(x)
        x = tf.keras.layers.Add()([shortcut, x])
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)

        shortcut = x
        x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv1D(64, 5, padding='same')(x)
        if shortcut.shape[-1] != x.shape[-1]:
            shortcut = tf.keras.layers.Conv1D(64, 1, padding='same')(shortcut)
        x = tf.keras.layers.Add()([shortcut, x])
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        final_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        final_model.fit(
            X_train_full, y_train_full,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        # 4️⃣ Evaluate on test set
        y_pred_prob = final_model.predict(X_test)
        y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"Run {run+1} Test Acc: {acc:.4f}, F1: {f1:.4f}")
        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

    # 5️⃣ Report mean ± std
    print("\n=== Final Results (Mean ± Std) ===")
    print(f"Accuracy: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"Precision: {np.mean(prec_list):.4f} ± {np.std(prec_list):.4f}")
    print(f"Recall: {np.mean(rec_list):.4f} ± {np.std(rec_list):.4f}")
    print(f"F1-score: {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")

    return final_model


import numpy as np
import matplotlib.pyplot as plt


def plot_spectral_profiles(df, label_col='y', wavelength_min=348, wavelength_max=1052, highlight_range=None):
    """
    Plot spectral profiles for a dataset with labels.

    Parameters:
    - df: pandas DataFrame, last column is assumed to be label if label_col provided
    - label_col: column name of the class label
    - wavelength_min: minimum wavelength (nm)
    - wavelength_max: maximum wavelength (nm)
    - highlight_range: tuple (start_nm, end_nm) to highlight a specific wavelength range
    """
    X = df.drop(columns=[label_col]).values
    y = df[label_col].values
    num_features = X.shape[1]
    wavelengths = np.linspace(wavelength_min, wavelength_max, num_features)

    # Plot mean spectra per class
    plt.figure(figsize=(12, 6))

    for class_label, color, name in zip([0, 1], ['blue', 'red'], ['Normal', 'Drought']):
        class_data = X[y == class_label]
        mean_spectrum = class_data.mean(axis=0)
        std_spectrum = class_data.std(axis=0)

        plt.plot(wavelengths, mean_spectrum, color=color, label=name)
        plt.fill_between(wavelengths,
                         mean_spectrum - std_spectrum,
                         mean_spectrum + std_spectrum,
                         color=color, alpha=0.2)

    if highlight_range:
        plt.axvspan(highlight_range[0], highlight_range[1], color='yellow', alpha=0.3, label='Selected Range')

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Spectral value / reflectance")
    plt.title("Mean Spectral Profiles with ±1 Std")
    plt.legend()
    plt.show()

    return wavelengths


def cnn_1d_resnet_model_evaluation(df, selected_wavelengths=None, epochs=100, batch_size=16, validation_split=0.2):
    """
    Train and evaluate a 1D-ResNet on hyperspectral tomato data using selected wavelength ranges.

    Parameters:
    - df: pandas DataFrame with spectral features and 'y' as label (0=Normal, 1=Drought)
    - selected_wavelengths: list of column names or indices to use; if None, use all wavelengths
    - epochs: number of training epochs
    - batch_size: batch size for training
    - validation_split: fraction of training data for validation
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, \
        precision_score, recall_score, f1_score

    # --------------------------
    # 1. Prepare Data
    # --------------------------
    if selected_wavelengths is not None:
        X = df[selected_wavelengths].values
    else:
        X = df.drop(columns=['y']).values

    y = df['y'].values
    X = X[..., np.newaxis]  # shape (num_samples, num_bands, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --------------------------
    # 2. Build 1D ResNet-style Model
    # --------------------------
    inputs = tf.keras.Input(shape=(X.shape[1], 1))

    # First conv layer
    x = tf.keras.layers.Conv1D(32, kernel_size=5, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Residual block 1
    shortcut = x
    x = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(32, 5, padding='same')(x)
    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Residual block 2
    shortcut = x
    x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(64, 5, padding='same')(x)
    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = tf.keras.layers.Conv1D(64, 1, padding='same')(shortcut)
    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Flatten + dense
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # --------------------------
    # 3. Train Model
    # --------------------------
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )

    # --------------------------
    # 4. Plot Training History
    # --------------------------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --------------------------
    # 5. Evaluate on Test Set
    # --------------------------
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()  # binary classification

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)  # Sensitivity
    specificity = tn / (tn + fp)  # True Negative Rate
    f1 = f1_score(y_test, y_pred)

    # Print metrics
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Sensitivity: {recall:.4f}")  # Recall
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"F1-score:    {f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Drought"]))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Drought"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    return model

def cnn_1d_spnet_model_evaluation(df, selected_wavelengths=None, epochs=100, batch_size=16, validation_split=0.2):
    """
    Train and evaluate a 1D-SP-Net on hyperspectral tomato data.

    Includes:
    - Residual–GC blocks
    - Channel attention
    - Swish activation
    - Dropout (0.5)
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, \
        accuracy_score, precision_score, recall_score, f1_score

    # --------------------------
    # 1. Prepare Data
    # --------------------------
    if selected_wavelengths is not None:
        X = df[selected_wavelengths].values
    else:
        X = df.drop(columns=['y']).values

    y = df['y'].values
    X = X[..., np.newaxis]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Swish activation
    def swish(x):
        return tf.keras.activations.swish(x)

    # --------------------------
    # Residual Module
    # --------------------------
    def residual_module(x):
        shortcut = x

        x = tf.keras.layers.Conv1D(32, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(swish)(x)

        x = tf.keras.layers.Conv1D(32, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Add()([shortcut, x])
        x = tf.keras.layers.Activation(swish)(x)

        return x

    # --------------------------
    # Global Context (GC) Module
    # --------------------------
    def gc_module(x):
        filters = x.shape[-1]

        context = tf.keras.layers.GlobalAveragePooling1D()(x)
        context = tf.keras.layers.Dense(filters // 2, activation=swish)(context)
        context = tf.keras.layers.Dense(filters, activation='sigmoid')(context)

        context = tf.keras.layers.Reshape((1, filters))(context)
        x = tf.keras.layers.Multiply()([x, context])

        return x

    # --------------------------
    # Residual–GC Block
    # --------------------------
    def residual_gc_block(x):
        x = residual_module(x)
        x = gc_module(x)
        return x

    # --------------------------
    # Channel Attention (extra)
    # --------------------------
    def channel_attention(x):
        filters = x.shape[-1]

        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
        max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)

        attention = tf.keras.layers.Multiply()([avg_pool, max_pool])
        attention = tf.keras.layers.Dense(filters, activation='sigmoid')(attention)

        attention = tf.keras.layers.Reshape((1, filters))(attention)
        x = tf.keras.layers.Multiply()([x, attention])

        return x

    # --------------------------
    # 2. Build 1D-SP-Net Model
    # --------------------------
    inputs = tf.keras.Input(shape=(X.shape[1], 1))

    # Initial layer
    x = tf.keras.layers.Conv1D(32, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(swish)(x)

    # Residual–GC blocks
    x = residual_gc_block(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = residual_gc_block(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Channel attention
    x = channel_attention(x)

    # Classification head
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation=swish)(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # --------------------------
    # 3. Train Model
    # --------------------------
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )

    # --------------------------
    # 4. Plot Training History
    # --------------------------
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # --------------------------
    # 5. Evaluation
    # --------------------------
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    specificity = tn / (tn + fp)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Sensitivity: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"F1-score:    {f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Drought"]))

    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Drought"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    return model

def cnn_1d_vgg_model_evaluation(df, selected_wavelengths=None, epochs=100, batch_size=16, validation_split=0.2):

    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, \
        accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

    # --------------------------
    # 1. Prepare Data
    # --------------------------
    if selected_wavelengths is not None:
        X = df[selected_wavelengths].values
    else:
        X = df.drop(columns=['y']).values

    y = df['y'].values
    X = X[..., np.newaxis]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --------------------------
    # 2. Build VGG-style Model
    # --------------------------
    inputs = tf.keras.Input(shape=(X.shape[1], 1))

    # Block 1
    x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    # Block 2
    x = tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    # Block 3
    x = tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    # Global pooling instead of flatten (better for small data)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Dense layers
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # --------------------------
    # 3. Class Imbalance Handling
    # --------------------------
    from collections import Counter
    counter = Counter(y_train)
    total = sum(counter.values())
    class_weight = {cls: total/(len(counter)*count) for cls, count in counter.items()}

    # --------------------------
    # 4. Train
    # --------------------------
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        class_weight=class_weight,
        verbose=1
    )

    # --------------------------
    # 5. Plot Training
    # --------------------------
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.legend()
    plt.title('Accuracy')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.legend()
    plt.title('Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # --------------------------
    # 6. Evaluation
    # --------------------------
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    specificity = tn / (tn + fp)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Sensitivity: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"F1-score:    {f1:.4f}")
    print(f"MCC:         {mcc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Drought"]))

    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Drought"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    return model

# y_pred_prob = [[4.61928913e-08]
#  [6.09781027e-01]
#  [1.24561528e-04]
#  [5.36574423e-03]
#  [4.72914577e-02]
#  [2.23877825e-04]
#  [4.32145207e-05]
#  [2.02324169e-04]
#  [9.99046147e-01]
#  [7.48030245e-02]
#  [9.87208962e-01]
#  [4.88908365e-02]
#  [8.70651364e-01]
#  [1.12466550e-05]
#  [4.35992748e-01]
#  [4.70397249e-03]
#  [4.96757866e-06]
#  [5.16987033e-02]
#  [6.36827119e-11]
#  [7.92847514e-01]
#  [3.46560217e-02]
#  [2.69031967e-04]
#  [2.56136334e-09]
#  [1.79300994e-01]
#  [4.73504188e-04]
#  [6.00341335e-02]
#  [3.72672430e-03]
#  [8.39867766e-08]
#  [9.98910606e-01]
#  [9.99929011e-01]
#  [6.53994918e-01]
#  [5.75362028e-05]
#  [9.99682367e-01]
#  [3.31153325e-03]
#  [9.29129064e-01]
#  [7.33210668e-02]
#  [1.81914568e-01]
#  [9.99999940e-01]
#  [1.13850692e-03]
#  [3.27539220e-02]
#  [4.73490625e-04]
#  [6.34521604e-01]
#  [1.06294602e-02]
#  [3.27401102e-01]
#  [7.32208556e-03]
#  [8.48739266e-01]
#  [1.10171875e-02]
#  [1.05555353e-06]
#  [7.03991827e-05]
#  [4.57058311e-01]
#  [1.49019703e-03]
#  [9.99441504e-01]
#  [3.47166479e-01]
#  [1.01531355e-03]
#  [8.22232962e-01]
#  [2.31598187e-02]
#  [7.70884156e-01]
#  [9.96235430e-01]
#  [1.52072869e-03]
#  [4.37153540e-05]
#  [6.47100201e-03]
#  [8.70180652e-02]
#  [1.20993927e-01]
#  [9.93003428e-01]
#  [3.26128534e-11]
#  [2.11200304e-03]
#  [8.11437905e-01]
#  [9.84270051e-02]
#  [9.99991417e-01]
#  [2.45724514e-01]
#  [2.60048453e-03]
#  [9.99036610e-01]
#  [2.73336220e-04]
#  [5.50572992e-08]
#  [6.37870326e-05]
#  [5.91670187e-06]]
def run_yield_loss_analysis(threshold=0.5, max_loss=40):
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter

    # --------------------------
    # Hardcoded probabilities
    # --------------------------

    y_pred_prob = np.array([
        4.61928913e-08, 6.09781027e-01, 1.24561528e-04, 5.36574423e-03,
        4.72914577e-02, 2.23877825e-04, 4.32145207e-05, 2.02324169e-04,
        9.99046147e-01, 7.48030245e-02, 9.87208962e-01, 4.88908365e-02,
        8.70651364e-01, 1.12466550e-05, 4.35992748e-01, 4.70397249e-03,
        4.96757866e-06, 5.16987033e-02, 6.36827119e-11, 7.92847514e-01,
        3.46560217e-02, 2.69031967e-04, 2.56136334e-09, 1.79300994e-01,
        4.73504188e-04, 6.00341335e-02, 3.72672430e-03, 8.39867766e-08,
        9.98910606e-01, 9.99929011e-01, 6.53994918e-01, 5.75362028e-05,
        9.99682367e-01, 3.31153325e-03, 9.29129064e-01, 7.33210668e-02,
        1.81914568e-01, 9.99999940e-01, 1.13850692e-03, 3.27539220e-02,
        4.73490625e-04, 6.34521604e-01, 1.06294602e-02, 3.27401102e-01,
        7.32208556e-03, 8.48739266e-01, 1.10171875e-02, 1.05555353e-06,
        7.03991827e-05, 4.57058311e-01, 1.49019703e-03, 9.99441504e-01,
        3.47166479e-01, 1.01531355e-03, 8.22232962e-01, 2.31598187e-02,
        7.70884156e-01, 9.96235430e-01, 1.52072869e-03, 4.37153540e-05,
        6.47100201e-03, 8.70180652e-02, 1.20993927e-01, 9.93003428e-01,
        3.26128534e-11, 2.11200304e-03, 8.11437905e-01, 9.84270051e-02,
        9.99991417e-01, 2.45724514e-01, 2.60048453e-03, 9.99036610e-01,
        2.73336220e-04, 5.50572992e-08, 6.37870326e-05, 5.91670187e-06
    ])

    # --------------------------
    # Yield loss computation
    # --------------------------
    yield_loss = []
    for p in y_pred_prob:
        if p < threshold:
            loss = 0.0
        else:
            loss = ((p - threshold) / (1 - threshold)) * max_loss
        yield_loss.append(loss)

    yield_loss = np.array(yield_loss)

    # --------------------------
    # Categorization
    # --------------------------
    categories = []
    for y in yield_loss:
        if y == 0:
            categories.append("No Loss")
        elif y <= 0.25 * max_loss:
            categories.append("Low")
        elif y <= 0.75 * max_loss:
            categories.append("Moderate")
        else:  # >0.75*max_loss
            categories.append("Severe")

    # --------------------------
    # Summary
    # --------------------------
    print("===== Yield Loss Summary =====")
    print(f"Min Loss: {yield_loss.min():.2f}%")
    print(f"Max Loss: {yield_loss.max():.2f}%")
    print(f"Average Loss: {yield_loss.mean():.2f}%")

    counts = Counter(categories)
    total = len(categories)

    print("\nCategory Distribution:")
    for k, v in counts.items():
        print(f"{k}: {v} ({(v/total)*100:.2f}%)")

    # --------------------------
    # Plots
    # --------------------------
    plt.figure()
    plt.hist(yield_loss, bins=20)
    plt.xlabel("Yield Loss (%)", fontsize=18)
    plt.ylabel("Frequency", fontsize=18)
    # plt.title("Yield Loss Distribution", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)
    plt.show()

    plt.figure()
    labels = list(counts.keys())
    values = list(counts.values())
    plt.bar(labels, values)
    plt.xlabel("Category", fontsize=18)
    plt.ylabel("Count", fontsize=18)
    # plt.title("Yield Loss Categories", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)
    plt.show()

    return y_pred_prob, yield_loss, categories


def generate_yield_report_from_prob(y_pred_prob, max_loss=60):
    """
    Generate drought yield loss report directly from predicted probabilities.

    Parameters:
        y_pred_prob (array-like): Model predicted probabilities
        max_loss (float): Maximum yield loss (%), default = 60

    Returns:
        str: Plain language report
    """

    yield_data = []

    # Step 1: Convert probability → yield loss + category
    for i, p in enumerate(y_pred_prob):
        if p < 0.5:
            loss = 0
        else:
            loss = ((p - 0.5) / 0.5) * max_loss

        # Categorization
        if loss == 0:
            cat = "No Loss"
        elif loss < 0.25*max_loss:
            cat = "Low"
        elif loss < 0.75*max_loss:
            cat = "Moderate"
        else:
            cat = "Severe"

        yield_data.append({
            "Sample": i + 1,
            "YieldLoss": round(loss, 2),
            "Category": cat
        })

    # Step 2: Statistics
    total_samples = len(yield_data)
    losses = [d["YieldLoss"] for d in yield_data]

    min_loss = min(losses)
    max_loss_val = max(losses)
    avg_loss = sum(losses) / total_samples

    # Category counts
    categories = {"No Loss": 0, "Low": 0, "Moderate": 0, "Severe": 0}
    for d in yield_data:
        categories[d["Category"]] += 1

    def pct(x):
        return (x / total_samples) * 100

    # Step 3: Report
    report = f"""
Tomato Yield Loss Assessment under Drought Stress
================================================

Overview:
Predicted drought stress probabilities were converted into estimated yield loss
using a maximum potential loss of {max_loss}% under severe drought conditions.

Dataset Summary (n = {total_samples}):

No Loss   : {categories['No Loss']} ({pct(categories['No Loss']):.2f}%)
Low       : {categories['Low']} ({pct(categories['Low']):.2f}%)
Moderate  : {categories['Moderate']} ({pct(categories['Moderate']):.2f}%)
Severe    : {categories['Severe']} ({pct(categories['Severe']):.2f}%)

Yield Loss Statistics:

Minimum Loss : {min_loss:.2f}%
Maximum Loss : {max_loss_val:.2f}%
Average Loss : {avg_loss:.2f}%

Interpretation:
Most samples fall under the "No Loss" category, indicating generally healthy plants.
However, moderate and severe cases suggest potential yield reduction under drought stress.
On average, the predicted yield loss is {avg_loss:.2f}%.

Recommendations:
- Monitor moderate and severe plants closely
- Apply irrigation to reduce stress impact
- Use continuous spectral monitoring for early detection
"""

    return report


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def plot_roc_curve(y_test, y_pred_prob):
    # Flatten in case shape is (n,1)
    y_pred_prob = y_pred_prob.ravel()

    # Compute ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(7, 6))

    plt.plot(fpr, tpr, linewidth=3, label=f"AUC = {roc_auc * 100:.2f}%")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=2)

    # Large text
    plt.xlabel("False Positive Rate", fontsize=18)
    plt.ylabel("True Positive Rate", fontsize=18)
    # plt.title(f"ROC Curve for {model_name}", fontsize=18)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return roc_auc

############################################
# MAIN WORKFLOW
############################################
if __name__ == "__main__":
    import os

    file_path = "C:/WVSU_Documents/Research/Tomato_Drougth_Stress_Data/Datasets_Drought_Stress/prospe_IR2.csv"
    print("File exists:", os.path.exists(file_path))
    df = load_dataset(file_path)
    maximum_yeild_loss=60 # 60%
    # Train and evaluate 1D-CNN: model, y_pred_prob,y_pred
    cnn_model,y_pred_prob,y_pred,y_test = cnn_1d_model_evaluation(df, epochs=100, batch_size=16)
    # print("Prediction results:", y_pred)
    # np.savetxt("y_pred_prob.txt", y_pred_prob)
    # np.savetxt("y_test.txt", y_test)
    # np.savetxt("y_pred.txt", y_pred)
    # plot_roc_curve(y_test, y_pred_prob)
    y_pred_prob, yield_loss, categories= run_yield_loss_analysis(threshold=0.5, max_loss=maximum_yeild_loss)
    # report=generate_yield_report_from_prob(y_pred_prob, max_loss=maximum_yeild_loss)
    print(yield_loss)
