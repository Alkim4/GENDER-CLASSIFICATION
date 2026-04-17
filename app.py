import streamlit as st
import numpy as np
import pandas as pd
import cv2
import joblib
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.measure import label, regionprops
from scipy.stats import skew, kurtosis
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Gender Classifier",
    page_icon="🔬",
    layout="centered",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}

.main {
    background-color: #0d0d0d;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

.stApp {
    background-color: #0d0d0d;
    color: #e8e8e8;
}

/* Title block */
.title-block {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}

.title-block h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    letter-spacing: -0.03em;
    color: #f0f0f0;
    margin-bottom: 0.3rem;
}

.title-block p {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #555;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    border: 1px dashed #333 !important;
    border-radius: 8px !important;
    background: #111 !important;
    padding: 1rem !important;
}

/* Result card */
.result-card {
    background: #111;
    border: 1px solid #222;
    border-radius: 10px;
    padding: 1.8rem 2rem;
    margin-top: 1rem;
}

.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #555;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.result-value-female {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.2rem;
    color: #f472b6;
    letter-spacing: -0.02em;
}

.result-value-male {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.2rem;
    color: #60a5fa;
    letter-spacing: -0.02em;
}

.meta-row {
    display: flex;
    gap: 1.5rem;
    margin-top: 1.2rem;
    padding-top: 1.2rem;
    border-top: 1px solid #1e1e1e;
    flex-wrap: wrap;
}

.meta-item {
    flex: 1;
    min-width: 130px;
}

.meta-item-label {
    font-size: 0.65rem;
    color: #444;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.2rem;
}

.meta-item-value {
    font-size: 0.85rem;
    color: #aaa;
}

.divider {
    border: none;
    border-top: 1px solid #1e1e1e;
    margin: 1.5rem 0;
}

/* Spinner text */
[data-testid="stSpinner"] p {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #555;
}

/* Image */
[data-testid="stImage"] {
    border-radius: 8px;
    overflow: hidden;
}

/* File uploader label */
[data-testid="stFileUploader"] label {
    font-family: 'DM Mono', monospace !important;
    color: #666 !important;
    font-size: 0.8rem !important;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 1rem;
    max-width: 780px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load models (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.join(os.path.dirname(__file__), "models")
    model        = joblib.load(os.path.join(base, "svm_model.pkl"))
    scaler       = joblib.load(os.path.join(base, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(base, "feature_names.pkl"))
    le           = joblib.load(os.path.join(base, "label_encoder.pkl"))
    return model, scaler, feature_names, le


# ─────────────────────────────────────────────
# Feature extraction  (exact copy from Part 1)
# ─────────────────────────────────────────────
FIXED_SIZE = (256, 256)

def safe_stat(func, arr, default=0.0):
    arr = np.asarray(arr).astype(np.float32).ravel()
    if arr.size == 0:
        return float(default)
    try:
        val = func(arr)
        if np.isnan(val) or np.isinf(val):
            return float(default)
        return float(val)
    except:
        return float(default)


def create_green_leaf_mask(img_rgb):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower = np.array([25, 20, 20], dtype=np.uint8)
    upper = np.array([100, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(img_hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest_idx, 255, 0).astype(np.uint8)
    return mask


def create_lesion_mask(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)
    cond1 = ((h >= 5) & (h <= 40) & (s >= 40) & (v >= 40))
    cond2 = (b >= 140)
    cond3 = (a >= 128)
    lesion = np.where((cond1 & cond2) | (cond1 & cond3), 255, 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, kernel)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_CLOSE, kernel)
    return lesion


def extract_color_features(img_rgb, img_gray, img_hsv, img_lab):
    feats = {}
    for i, ch_name in enumerate(['r', 'g', 'b']):
        ch = img_rgb[:, :, i]
        feats[f'rgb_mean_{ch_name}'] = float(np.mean(ch))
        feats[f'rgb_std_{ch_name}']  = float(np.std(ch))
        feats[f'rgb_skew_{ch_name}'] = safe_stat(skew, ch)
    for i, ch_name in enumerate(['h', 's', 'v']):
        ch = img_hsv[:, :, i]
        feats[f'hsv_mean_{ch_name}'] = float(np.mean(ch))
        feats[f'hsv_std_{ch_name}']  = float(np.std(ch))
    for i, ch_name in enumerate(['l', 'a', 'b']):
        ch = img_lab[:, :, i]
        feats[f'lab_mean_{ch_name}'] = float(np.mean(ch))
        feats[f'lab_std_{ch_name}']  = float(np.std(ch))
    feats['gray_mean']     = float(np.mean(img_gray))
    feats['gray_std']      = float(np.std(img_gray))
    feats['gray_skew']     = safe_stat(skew, img_gray)
    feats['gray_kurtosis'] = safe_stat(kurtosis, img_gray)
    hist_density, _ = np.histogram(img_gray, bins=256, range=(0, 256), density=True)
    feats['gray_entropy'] = float(-np.sum(hist_density * np.log2(hist_density + 1e-12)))
    for i, ch_name in enumerate(['r', 'g', 'b']):
        hist = cv2.calcHist([img_rgb], [i], None, [8], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-12)
        for j, val in enumerate(hist):
            feats[f'rgb_hist_{ch_name}_{j}'] = float(val)
    return feats


def extract_glcm_features(img_rgb, img_gray, img_hsv, img_lab):
    feats = {}
    glcm = graycomatrix(
        img_gray,
        distances=[1, 2],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256, symmetric=True, normed=True,
    )
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
        values = graycoprops(glcm, prop).flatten()
        feats[f'glcm_{prop}_mean'] = float(np.mean(values))
        feats[f'glcm_{prop}_std']  = float(np.std(values))
    return feats


def extract_lbp_features(img_rgb, img_gray, img_hsv, img_lab, radius=1, n_points=8):
    feats = {}
    lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    for i, val in enumerate(hist):
        feats[f'lbp_hist_{i}'] = float(val)
    feats['lbp_mean'] = float(np.mean(lbp))
    feats['lbp_std']  = float(np.std(lbp))
    return feats


def extract_shape_features(img_rgb, img_gray, img_hsv, img_lab):
    feats = {}
    mask = create_green_leaf_mask(img_rgb)
    labeled = label(mask > 0)
    props   = regionprops(labeled)
    if len(props) == 0:
        for k in ['face_area','face_perimeter','face_bbox_w','face_bbox_h',
                  'face_aspect_ratio','face_extent','face_solidity',
                  'face_equiv_diameter','face_eccentricity']:
            feats[k] = 0.0
        return feats
    region = max(props, key=lambda x: x.area)
    minr, minc, maxr, maxc = region.bbox
    bbox_h = maxr - minr
    bbox_w = maxc - minc
    feats['face_area']          = float(region.area)
    feats['face_perimeter']     = float(region.perimeter)
    feats['face_bbox_w']        = float(bbox_w)
    feats['face_bbox_h']        = float(bbox_h)
    feats['face_aspect_ratio']  = float(bbox_w / (bbox_h + 1e-12))
    feats['face_extent']        = float(region.extent)
    feats['face_solidity']      = float(region.solidity)
    feats['face_equiv_diameter']= float(region.equivalent_diameter_area)
    feats['face_eccentricity']  = float(region.eccentricity)
    return feats


def extract_hog_features(img_rgb, img_gray, img_hsv, img_lab):
    feats = {}
    hog_vec = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(32, 32),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True,
    )
    for i, val in enumerate(hog_vec):
        feats[f'hog_{i}'] = float(val)
    grad_x   = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y   = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    feats['grad_mean'] = float(np.mean(grad_mag))
    feats['grad_std']  = float(np.std(grad_mag))
    return feats


def extract_lesion_features(img_rgb, img_gray, img_hsv, img_lab):
    feats = {}
    lesion_mask = create_lesion_mask(img_rgb)
    leaf_mask   = create_green_leaf_mask(img_rgb)
    lesion_mask = cv2.bitwise_and(lesion_mask, lesion_mask, mask=leaf_mask)
    labeled  = label(lesion_mask > 0)
    props    = regionprops(labeled)
    leaf_area   = np.sum(leaf_mask   > 0)
    lesion_area = np.sum(lesion_mask > 0)
    feats['lesion_area']   = float(lesion_area)
    feats['lesion_ratio']  = float(lesion_area / (leaf_area + 1e-12))
    feats['lesion_count']  = float(len(props))
    if len(props) == 0:
        feats['lesion_mean_area']     = 0.0
        feats['lesion_largest_area']  = 0.0
        feats['lesion_perimeter_sum'] = 0.0
    else:
        areas      = [p.area      for p in props]
        perimeters = [p.perimeter for p in props]
        feats['lesion_mean_area']     = float(np.mean(areas))
        feats['lesion_largest_area']  = float(np.max(areas))
        feats['lesion_perimeter_sum'] = float(np.sum(perimeters))
    return feats


def extract_all_features(img_rgb, img_gray, img_hsv, img_lab):
    all_feats = {}
    all_feats.update(extract_color_features(img_rgb, img_gray, img_hsv, img_lab))
    all_feats.update(extract_glcm_features(img_rgb, img_gray, img_hsv, img_lab))
    all_feats.update(extract_lbp_features(img_rgb, img_gray, img_hsv, img_lab))
    all_feats.update(extract_shape_features(img_rgb, img_gray, img_hsv, img_lab))
    all_feats.update(extract_hog_features(img_rgb, img_gray, img_hsv, img_lab))
    all_feats.update(extract_lesion_features(img_rgb, img_gray, img_hsv, img_lab))
    return all_feats


def read_and_resize(img_array):
    img_bgr  = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_bgr  = cv2.resize(img_bgr, FIXED_SIZE)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_hsv  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    return img_rgb, img_gray, img_hsv, img_lab


def predict_gender(img_array, model, scaler, feature_names, le):
    img_rgb, img_gray, img_hsv, img_lab = read_and_resize(img_array)
    all_feats = extract_all_features(img_rgb, img_gray, img_hsv, img_lab)

    # Build full-width DataFrame in the exact column order the scaler expects
    all_cols = list(scaler.feature_names_in_)
    feat_df  = pd.DataFrame([all_feats])
    for col in all_cols:
        if col not in feat_df.columns:
            feat_df[col] = 0.0
    feat_df = feat_df[all_cols]

    scaled    = scaler.transform(feat_df)
    scaled_df = pd.DataFrame(scaled, columns=all_cols)
    selected  = scaled_df[feature_names].values

    pred      = model.predict(selected)
    label_out = le.inverse_transform(pred)[0]

    # Compute real confidence score from decision function or predict_proba
    try:
        if hasattr(model, "predict_proba"):
            proba      = model.predict_proba(selected)[0]
            confidence = float(np.max(proba)) * 100
        else:
            dec        = model.decision_function(selected)[0]
            # Convert decision score to a 0-100 confidence using sigmoid
            confidence = float(1 / (1 + np.exp(-abs(dec)))) * 100
    except Exception:
        confidence = None

    return label_out, confidence


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>Gender Classifier</h1>
    <p>SVM · Handcrafted Features · HOG / GLCM / LBP</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

model, scaler, feature_names, le = load_models()

uploaded_file = st.file_uploader(
    "Drop an image here or click to browse",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
    label_visibility="visible",
)

if uploaded_file is not None:
    image     = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        with st.spinner("Analysing..."):
            result, confidence = predict_gender(img_array, model, scaler, feature_names, le)

        icon  = "🚺" if result == "Female" else "🚹"
        color_class = "result-value-female" if result == "Female" else "result-value-male"
        confidence_str = f"{confidence:.2f} %" if confidence is not None else "N/A"

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Prediction</div>
            <div class="{color_class}">{icon} {result}</div>
            <div class="meta-row">
                <div class="meta-item">
                    <div class="meta-item-label">Model</div>
                    <div class="meta-item-value">SVM (RBF)</div>
                </div>
                <div class="meta-item">
                    <div class="meta-item-label">Features</div>
                    <div class="meta-item-value">100 selected</div>
                </div>
                <div class="meta-item">
                    <div class="meta-item-label">Confidence</div>
                    <div class="meta-item-value">{confidence_str}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-item-label">Feature Groups</div>
                    <div class="meta-item-value">HOG · GLCM · LBP</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0; color: #333;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">⬆</div>
        <div style="font-family: 'DM Mono', monospace; font-size: 0.8rem; color: #444;">
            Upload a JPG, PNG, BMP, or TIFF image to begin
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; font-size:0.7rem; color:#333; font-family:'DM Mono',monospace;">
    ITIM 82 · Lab Activity 5 · Part 3 — Fortuna
</div>
""", unsafe_allow_html=True)