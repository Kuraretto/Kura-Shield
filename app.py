import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import os
import joblib
import tensorflow as tf
import tempfile
import librosa
import librosa.display
import soundfile as sf
import seaborn as sns
from PIL import Image
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(page_title="Deepfake Forensics Lab", layout="wide")

# ==========================================
# CONFIG & SETUP
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output') 

# Create directories if they don't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

AUDIO_CONF = {
    'sr': 16000,
    'duration': 3,
    'n_mels': 128,
    'n_fft': 2048,
    'hop_length': 512,
    'fmax': 8000
}

# GPU Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"[GPU Setup Error] {e}")

# ==========================================
# HELPER FUNCTIONS (FILE MANAGEMENT)
# ==========================================

def get_output_path(media_type, input_filename):
    """
    Creates directory structure: Output/<media_type>/<input_filename_no_ext>/
    """
    filename_no_ext = os.path.splitext(input_filename)[0]
    safe_foldername = "".join([c for c in filename_no_ext if c.isalnum() or c in (' ', '-', '_')]).strip()
    
    save_dir = os.path.join(OUTPUT_DIR, media_type, safe_foldername)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    return save_dir

# ==========================================
# CORE CLASSES (DETECTOR)
# ==========================================

class DeepfakeDetector:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.models = {}
        self.feature_extractors = {}
        self.scaler = StandardScaler()
        self.xgb = XGBClassifier() 
        
    def create_fine_tuned_model(self, model_name, num_classes=2):
        if model_name == 'DenseNet201':
            base_model = tf.keras.applications.DenseNet201(include_top=False, weights=None, input_shape=self.input_shape, pooling='avg')
        elif model_name == 'EfficientNetB5':
            base_model = tf.keras.applications.EfficientNetB5(include_top=False, weights=None, input_shape=self.input_shape, pooling='avg')
        elif model_name == 'Xception':
            base_model = tf.keras.applications.Xception(include_top=False, weights=None, input_shape=self.input_shape, pooling='avg')
        elif model_name == 'ConvNeXtSmall':
            base_model = tf.keras.applications.ConvNeXtSmall(include_top=False, weights=None, input_shape=self.input_shape, pooling='avg')
        
        inputs = base_model.input
        if model_name == 'ConvNeXtSmall':
            x = base_model.layers[-2].output
        else:
            x = base_model.output
            
        x = tf.keras.layers.Dense(512, activation='relu', name=f'{model_name}_dense_512')(x)
        x = tf.keras.layers.Dropout(0.5, name=f'{model_name}_dropout')(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name=f'{model_name}_output')(x)
        
        model = tf.keras.Model(inputs, outputs, name=f'{model_name}_finetuned')
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'AUC'])
        return model

    def create_feature_extractor(self, model_name):
        fine_tuned_model = self.models[model_name]
        feature_layer = fine_tuned_model.layers[-4].output
        return tf.keras.Model(inputs=fine_tuned_model.input, outputs=feature_layer)

    def extract_features(self, images):
        model_names = ['DenseNet201', 'EfficientNetB5', 'Xception', 'ConvNeXtSmall']
        all_features = {name: [] for name in model_names}

        batch_images = []
        for img in images:
            img_resized = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
            batch_images.append(img_resized)
        
        batch_images = np.array(batch_images)

        for model_name in model_names:
            model_input = batch_images.copy()
            model_name_lower = model_name.lower()
            
            if model_name_lower == "convnextsmall":
                model_input = tf.keras.applications.convnext.preprocess_input(model_input)
            elif model_name_lower == "efficientnetb5":
                model_input = tf.keras.applications.efficientnet.preprocess_input(model_input)
            elif model_name_lower == "densenet201":
                model_input = tf.keras.applications.densenet.preprocess_input(model_input)
            elif model_name_lower == "xception":
                model_input = tf.keras.applications.xception.preprocess_input(model_input)
            else:
                model_input = tf.keras.applications.imagenet_utils.preprocess_input(model_input, mode='tf')
                
            features = self.feature_extractors[model_name].predict(model_input, verbose=0)
            all_features[model_name].extend(features)
        
        return np.concatenate([np.array(all_features[name]) for name in model_names], axis=1)

    def predict(self, image, selected_features):
        features = self.extract_features([image])
        X_selected = features[:, selected_features]
        X_scaled = self.scaler.transform(X_selected)
        return self.xgb.predict(X_scaled), self.xgb.predict_proba(X_scaled)

    def predict_individual_models(self, image):
        model_names = ['DenseNet201', 'EfficientNetB5', 'Xception', 'ConvNeXtSmall']
        individual_preds = {}

        img_resized = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
        batch_images = np.array([img_resized])

        for model_name in model_names:
            model_input = batch_images.copy()
            model_name_lower = model_name.lower()
            if model_name_lower == "convnextsmall": model_input = tf.keras.applications.convnext.preprocess_input(model_input)
            elif model_name_lower == "efficientnetb5": model_input = tf.keras.applications.efficientnet.preprocess_input(model_input)
            elif model_name_lower == "densenet201": model_input = tf.keras.applications.densenet.preprocess_input(model_input)
            elif model_name_lower == "xception": model_input = tf.keras.applications.xception.preprocess_input(model_input)
            else: model_input = tf.keras.applications.imagenet_utils.preprocess_input(model_input, mode='tf')

            pred = self.models[model_name].predict(model_input, verbose=0)
            individual_preds[model_name] = pred[0] 

        return individual_preds

# ==========================================
# AUDIO UTILS
# ==========================================

@st.cache_resource
def load_audio_models():
    h5_path = os.path.join(MODEL_DIR, 'voiceshield_model.h5')
    tflite_path = os.path.join(MODEL_DIR, 'voiceshield_model.tflite')
    h5_model = None
    tflite_interpreter = None
    
    if os.path.exists(h5_path):
        try:
            h5_model = tf.keras.models.load_model(h5_path)
        except Exception as e:
            st.error(f"Error loading H5 Audio Model: {e}")
            
    if os.path.exists(tflite_path):
        try:
            tflite_interpreter = tf.lite.Interpreter(model_path=tflite_path)
            tflite_interpreter.allocate_tensors()
        except Exception as e:
            st.error(f"Error loading TFLite Audio Model: {e}")
    return h5_model, tflite_interpreter

def extract_audio_segments(file_path, overlap=0.5):
    try:
        audio, _ = librosa.load(file_path, sr=AUDIO_CONF['sr']) 
        sample_rate = AUDIO_CONF['sr']
        duration = AUDIO_CONF['duration']
        segment_samples = int(duration * sample_rate) 
        step_samples = int(segment_samples * (1 - overlap)) 
        segments = []
        timestamps = []
        
        if len(audio) < segment_samples:
             audio = np.pad(audio, (0, segment_samples - len(audio)))
        
        for start_idx in range(0, len(audio) - int(segment_samples*0.1), step_samples):
            end_idx = start_idx + segment_samples
            if end_idx > len(audio):
                segment_audio = np.pad(audio[start_idx:], (0, end_idx - len(audio)))
            else:
                segment_audio = audio[start_idx:end_idx]
            
            mel = librosa.feature.melspectrogram(y=segment_audio, sr=sample_rate, n_mels=AUDIO_CONF['n_mels'], n_fft=AUDIO_CONF['n_fft'], hop_length=AUDIO_CONF['hop_length'], fmax=AUDIO_CONF['fmax'])
            mel_db = librosa.power_to_db(mel, ref=np.max)
            norm_mel = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
            
            if norm_mel.shape[1] > 94: norm_mel = norm_mel[:, :94]
            elif norm_mel.shape[1] < 94: norm_mel = np.pad(norm_mel, ((0, 0), (0, 94 - norm_mel.shape[1])))
                
            segments.append(norm_mel)
            timestamps.append(start_idx / sample_rate)
            
        return audio, segments, timestamps
    except Exception as e:
        st.error(f"Error processing audio segments: {e}")
        return None, None, None

def predict_audio_batch(h5_model, tflite_interpreter, segments):
    h5_scores = []
    tflite_scores = []
    for seg in segments:
        input_data = seg[np.newaxis, ..., np.newaxis]
        input_data_f32 = input_data.astype(np.float32)
        if h5_model:
            pred = h5_model.predict(input_data, verbose=0)[0][0]
            h5_scores.append(pred)
        if tflite_interpreter:
            input_details = tflite_interpreter.get_input_details()
            output_details = tflite_interpreter.get_output_details()
            tflite_interpreter.set_tensor(input_details[0]['index'], input_data_f32)
            tflite_interpreter.invoke()
            pred_lite = tflite_interpreter.get_tensor(output_details[0]['index'])[0][0]
            tflite_scores.append(pred_lite)
    return h5_scores, tflite_scores

def generate_timeline_report(audio, timestamps, h5_scores, tflite_scores, filename):
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(14, 10))
    plt.suptitle(f"Deepfake Timeline Analysis: {filename}", fontsize=16, fontweight='bold')
    
    avg_h5 = np.mean(h5_scores) if h5_scores else 0
    avg_tflite = np.mean(tflite_scores) if tflite_scores else 0
    
    ax1 = plt.subplot(3, 1, 1)
    librosa.display.waveshow(audio, sr=AUDIO_CONF['sr'], color='#2c3e50', alpha=0.6)
    ax1.set_title("Full Audio Waveform", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, len(audio)/AUDIO_CONF['sr'])

    ax2 = plt.subplot(3, 1, 2)
    if h5_scores:
        ax2.plot(timestamps, h5_scores, label='Keras (.h5)', color='#e74c3c', linewidth=2, marker='o', markersize=4)
    if tflite_scores:
        ax2.plot(timestamps, tflite_scores, label='TFLite', color='#3498db', linewidth=2, linestyle='--')
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Decision Threshold')
    ax2.fill_between(timestamps, 0.5, 1.0, color='#e74c3c', alpha=0.1, label='Fake Zone')
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(0, len(audio)/AUDIO_CONF['sr'])
    ax2.set_title("Deepfake Probability Over Time", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Probability (1.0 = Fake)")
    ax2.legend(loc='upper right')

    ax3 = plt.subplot(3, 1, 3)
    labels = ['H5 Average', 'TFLite Average']
    values = [avg_h5 * 100, avg_tflite * 100]
    colors = ['#c0392b' if v > 50 else '#27ae60' for v in values]
    bars = ax3.barh(labels, values, color=colors, edgecolor='black', height=0.5)
    ax3.set_xlim(0, 100)
    ax3.axvline(50, color='gray', linestyle='--')
    ax3.set_title("Global Confidence Score", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Probability (%)")
    for bar in bars:
        width = bar.get_width()
        ax3.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center', fontweight='bold')

    plt.tight_layout()
    return fig, avg_h5

# ==========================================
# GRAD-CAM & UTILS
# ==========================================

def calculate_heatmap_spread(heatmap):
    medium_activation_count = np.sum((heatmap > 0.3) & (heatmap < 0.7))
    total_activation_count = np.sum(heatmap > 0.1)
    if total_activation_count == 0: return 0.0
    return medium_activation_count / total_activation_count

def find_target_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name
    raise ValueError("Could not find a suitable 4D layer for Grad-CAM.")

def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_model_visualization(model, img_rgb, model_name):
    try:
        img_resized = cv2.resize(img_rgb, (224, 224))
        model_input = np.expand_dims(img_resized, axis=0)
        
        model_name_lower = model_name.lower()
        if model_name_lower == "convnextsmall":
            model_input = tf.keras.applications.convnext.preprocess_input(model_input)
        elif model_name_lower == "efficientnetb5":
            model_input = tf.keras.applications.efficientnet.preprocess_input(model_input)
        elif model_name_lower == "densenet201":
            model_input = tf.keras.applications.densenet.preprocess_input(model_input)
        elif model_name_lower == "xception":
            model_input = tf.keras.applications.xception.preprocess_input(model_input)
        else:
            model_input = tf.keras.applications.imagenet_utils.preprocess_input(model_input, mode='tf')

        target_layer = find_target_layer(model)
        heatmap = get_gradcam_heatmap(model, model_input, target_layer, pred_index=1) 
        spread_score = calculate_heatmap_spread(heatmap)

        heatmap = np.uint8(255 * heatmap)
        jet = matplotlib.colormaps["jet"] 
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img_rgb.shape[1], img_rgb.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

        superimposed_img = jet_heatmap * 0.4 + img_rgb
        superimposed_img = np.uint8(superimposed_img)
        
        return superimposed_img, spread_score
    except Exception as e:
        print(f"Grad-CAM Error ({model_name}): {e}")
        return img_rgb, 0.0

# ==========================================
# APP LOADING 
# ==========================================

@st.cache_resource
def load_real_detector():
    detector = DeepfakeDetector()
    model_names = ['DenseNet201', 'EfficientNetB5', 'Xception', 'ConvNeXtSmall']
    missing_models = []
    for name in model_names:
        path = os.path.join(MODEL_DIR, f'{name}_best.h5')
        if os.path.exists(path):
            detector.models[name] = detector.create_fine_tuned_model(name)
            detector.models[name].load_weights(path)
            detector.feature_extractors[name] = detector.create_feature_extractor(name)
        else:
            missing_models.append(name)
            
    if missing_models:
        return None, None, f" Missing Image Models: {', '.join(missing_models)}. Please check 'models/' directory."

    try:
        if os.path.exists(os.path.join(MODEL_DIR, 'xgb_model.joblib')):
            detector.xgb = joblib.load(os.path.join(MODEL_DIR, 'xgb_model.joblib'))
            detector.scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
            features_list = np.load(os.path.join(MODEL_DIR, 'selected_features.npy'))
            return detector, features_list, None
        else:
             return None, None, "Missing XGBoost classifier files."
    except Exception as e:
        return None, None, f"Error loading classifier: {e}"

detector, selected_features, error_msg = load_real_detector()
audio_h5, audio_tflite = load_audio_models()

if error_msg:
    pass 
elif detector:
    if len(tf.config.list_physical_devices('GPU')) > 0:
        st.success("🚀 GPU Detected! Models running on CUDA.", icon="✅")
    else:
        st.success("⚠️ No GPU detected. Running on CPU.", icon="🐢")

# ==========================================
# PROCESSING PIPELINES
# ==========================================

def process_full_video_pipeline(video_file, detector, selected_features):
    st.subheader("2. Full Video Forensics")
    
    # --- PATHING: Output/video/<filename>/ ---
    save_dir = get_output_path("video", video_file.name)
    st.markdown(f"📂 **Output Folder:** `{save_dir}`")
    
    st.markdown("Processing EVERY frame through ALL 4 models. Generating 4 Grad-CAM videos.")

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    tfile.close() 
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_size = (224, 224) 
    model_names = ['DenseNet201', 'EfficientNetB5', 'Xception', 'ConvNeXtSmall']
    writers = {}
    output_filenames = {}
    
    # --- FIX: Use WebM (VP8) for Browser Playback ---
    # Browser compatibility: .webm with 'vp80' codec
    fourcc = cv2.VideoWriter_fourcc(*'vp80') 
    
    for name in model_names:
        # Change extension to .webm
        fname = os.path.join(save_dir, f"gradcam_{name}.webm")
        output_filenames[name] = fname
        writers[name] = cv2.VideoWriter(fname, fourcc, fps, output_size)

    frame_predictions = [] 
    model_spreads = {name: [] for name in model_names} 
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            pred, prob = detector.predict(frame_rgb, selected_features)
            frame_predictions.append(prob[0][1]) 
        except:
            frame_predictions.append(0.0)

        for name in model_names:
            try:
                heatmap_img, spread = generate_model_visualization(detector.models[name], frame_rgb, name)
                model_spreads[name].append(spread)
                out_frame_resized = cv2.resize(heatmap_img, output_size)
                out_frame_bgr = cv2.cvtColor(out_frame_resized, cv2.COLOR_RGB2BGR)
                writers[name].write(out_frame_bgr)
            except Exception as e:
                pass

        frame_count += 1
        if frame_count % 5 == 0: 
            status_text.text(f"Processing Frame {frame_count}/{total_frames}")
            progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    for w in writers.values(): w.release()
    st.success(f"Processing Complete! Files saved to {save_dir}")
    
    avg_fake_prob = np.mean(frame_predictions)
    st.markdown("### Final Verdict (XGBoost Ensemble)")
    col1, col2 = st.columns(2)
    with col1:
        if avg_fake_prob > 0.5:
            st.error(f"## FAKE")
            st.metric("Avg Confidence", f"{avg_fake_prob*100:.2f}%")
        else:
            st.success(f"## REAL")
            st.metric("Avg Confidence", f"{(1-avg_fake_prob)*100:.2f}%")
            
    with col2:
        st.metric("Frames Analyzed", frame_count)
        st.metric("Avg Noise Level (Xception)", f"{np.mean(model_spreads['Xception']):.2f}")

    st.markdown("---")
    st.subheader("Analysis Graphs")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(frame_predictions, color='red', label='Ensemble Prediction')
    ax.axhline(y=0.5, color='blue', linestyle='--')
    ax.set_title('Frame-by-Frame Fake Probability')
    ax.legend()
    st.pyplot(fig)
    
    st.subheader(" Generated Grad-CAM Analysis & Frame Inspection")
    
    # --- NEW: Frame-by-Frame Slider Logic ---
    def display_video_with_slider(video_path, model_name):
        """Helper to display video and add a frame slider below it"""
        st.markdown(f"**{model_name}**")
        
        if os.path.exists(video_path):
            # 1. Display Video
            st.video(video_path)
            
            # 2. Frame Slider Logic
            cap_insp = cv2.VideoCapture(video_path)
            total_frames_insp = int(cap_insp.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Unique key for every slider based on model name
            frame_idx = st.slider(f"Inspect Frame ({model_name})", 0, max(0, total_frames_insp-1), 0, key=f"slider_{model_name}")
            
            # Seek to frame
            cap_insp.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret_insp, frame_insp = cap_insp.read()
            
            if ret_insp:
                frame_insp = cv2.cvtColor(frame_insp, cv2.COLOR_BGR2RGB)
                st.image(frame_insp, caption=f"Frame {frame_idx}", use_column_width=True)
            
            cap_insp.release()

            # 3. Download Button
            with open(video_path, 'rb') as f:
                st.download_button(f"Download {model_name}", f, file_name=os.path.basename(video_path), mime='video/webm')
        else:
            st.error(f"File not found: {video_path}")

    # Display logic in 2 columns
    cols = st.columns(2) 
    for i, name in enumerate(model_names):
        with cols[i % 2]:
            fname = output_filenames[name]
            display_video_with_slider(fname, name)

def crop_image_ui(image):
    st.subheader("1. Crop Image (Focus on the Face)")
    col_controls, col_preview = st.columns([1, 2])
    with col_controls:
        width, height = image.size
        left = st.slider("Left Crop", 0, width // 2, 0)
        right = st.slider("Right Crop", 0, width // 2, 0)
        top = st.slider("Top Crop", 0, height // 2, 0)
        bottom = st.slider("Bottom Crop", 0, height // 2, 0)
    with col_preview:
        crop_area = (left, top, width - right, height - bottom)
        cropped_img = image.crop(crop_area)
        st.image(cropped_img, caption="Preview", width=400)
    return cropped_img

def process_image_analysis(cropped_img_pil, filename):
    st.subheader("2. Ensemble Analysis Results")
    
    # --- PATHING: Output/image/<filename>/ ---
    save_dir = get_output_path("image", filename)
    st.markdown(f"📂 **Output Folder:** `{save_dir}`")
    
    # Save original crop
    cropped_img_pil.save(os.path.join(save_dir, "cropped_input.png"))
    
    img_array = np.array(cropped_img_pil.convert('RGB'))
    
    with st.spinner("Running 4 CNNs + XGBoost Classifier..."):
        try:
            pred, prob = detector.predict(img_array, selected_features)
            cls = pred[0]
            conf = prob[0]
            final_confidence = conf[cls] * 100
            individual_probs = detector.predict_individual_models(img_array)

            if cls == 1:
                st.error(f"###  FINAL VERDICT: FAKE ({final_confidence:.2f}%)")
            else:
                st.success(f"###  FINAL VERDICT: REAL ({final_confidence:.2f}%)")

            st.markdown("#### Ensemble Model (XGBoost) Output")
            cols_ens = st.columns(2)
            with cols_ens[0]: st.metric("Probability of REAL", f"{conf[0]*100:.2f}%")
            with cols_ens[1]: st.metric("Probability of FAKE", f"{conf[1]*100:.2f}%")
            st.markdown("---")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            return

    st.subheader("3. Individual Model Insights")
    cols = st.columns(4)
    model_names = ['DenseNet201', 'EfficientNetB5', 'Xception', 'ConvNeXtSmall']
    
    for i, name in enumerate(model_names):
        with cols[i]:
            st.markdown(f"#### {name}")
            fake_prob = individual_probs[name][1] * 100
            heatmap_img, spread_score = generate_model_visualization(detector.models[name], img_array, name)
            
            # Save Heatmaps
            heatmap_save_path = os.path.join(save_dir, f"gradcam_{name}.png")
            cv2.imwrite(heatmap_save_path, cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR))
            
            st.metric(label="Fake Confidence", value=f"{fake_prob:.1f}%")
            st.metric(label="Attention Spread", value=f"{spread_score:.2f}")
            st.image(heatmap_img, caption=f"Grad-CAM: {name}", use_column_width=True)

# --- AUDIO PIPELINE ---
def process_audio_pipeline(audio_file):
    st.subheader("Audio Forensics Analysis")
    
    # --- PATHING: Output/audio/<filename>/ ---
    save_dir = get_output_path("audio", audio_file.name)
    st.markdown(f"📂 **Output Folder:** `{save_dir}`")
    
    file_ext = os.path.splitext(audio_file.name)[1].lower()
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
    tfile.write(audio_file.read())
    tfile.close() 
    temp_path = tfile.name
    
    st.info(f"Processing media: {audio_file.name}")

    with st.spinner("Segmenting Audio & Running Inference..."):
        # 1. Extract segments
        audio, segments, timestamps = extract_audio_segments(temp_path, overlap=0.5)
        
        if segments:
            # Save processed WAV
            wav_save_path = os.path.join(save_dir, "processed_audio.wav")
            sf.write(wav_save_path, audio, AUDIO_CONF['sr'])
            
            # 2. Predict
            h5_scores, tflite_scores = predict_audio_batch(audio_h5, audio_tflite, segments)
            
            # 3. Global Average Score
            avg_score = np.mean(h5_scores) if h5_scores else 0.0
            
            if avg_score > 0.5:
                st.error(f"###  FINAL VERDICT: FAKE (Average Confidence: {(avg_score*100):.2f}%)")
            else:
                st.success(f"###  FINAL VERDICT: REAL (Average Confidence: {(1-avg_score)*100:.2f}%)")
            
            # 4. Generate Timeline Report
            fig, _ = generate_timeline_report(audio, timestamps, h5_scores, tflite_scores, audio_file.name)
            
            # Save & Display Graph
            graph_save_path = os.path.join(save_dir, "analysis_report.png")
            fig.savefig(graph_save_path)
            
            st.pyplot(fig) # Display graph
            st.audio(wav_save_path) # Display audio player from SAVED file

        else:
            st.error("Failed to process media file. If this is a video, ensure ffmpeg is installed")

# ==========================================
# MAIN UI
# ==========================================

def main():
    st.title(" Deepfake Forensics Lab")
    
    if error_msg:
        st.error(" System Startup Failed")
        st.warning(error_msg)
        st.info(" Please check the 'models/' directory and ensure all 4 CNN models + Audio models + XGBoost files are present.")
        return

    # Sidebar
    mode = st.sidebar.radio("Select Media Type:", ["Image", "Video", "Audio"])

    if mode == "Image":
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if uploaded_file:
            original_image = Image.open(uploaded_file)
            cropped_image = crop_image_ui(original_image)
            if st.button("Analyze Cropped Area"):
                process_image_analysis(cropped_image, uploaded_file.name)
    
    elif mode == "Video":
        uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        if uploaded_video:
            st.video(uploaded_video)
            if st.button("Start Full Forensics Analysis"):
                process_full_video_pipeline(uploaded_video, detector, selected_features)
                
    elif mode == "Audio":
        st.header(" Voice Deepfake Detection")
        if not audio_h5 and not audio_tflite:
            st.error(" Audio models not found! Check 'models/' directory.")
        else:
            uploaded_audio = st.file_uploader("Upload Audio or Video File", type=['wav', 'mp3', 'flac', 'ogg', 'm4a', 'mp4', 'avi', 'mov', 'mkv'])
            if uploaded_audio:
                if st.button("Analyze Audio"):
                    process_audio_pipeline(uploaded_audio)

if __name__ == '__main__':
    main()