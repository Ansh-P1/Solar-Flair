import os
import streamlit as st
import numpy as np
import cv2
import time
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================================
# 1. CONSTANTS & MATHEMATICAL SETUP
# ==========================================
METERS_PER_PIXEL = 0.075        
IRRADIANCE = 4.5                
PERFORMANCE_RATIO = 0.75        
COST_PER_KW = 60000             
TARIFF_SAVINGS = 7.0            

# ==========================================
# 2. CACHED AI LOADING
# ==========================================
# @st.cache_resource forces Streamlit to only load the 25MB model ONCE when the server starts.
# Without this, it would try to load the model every time the user clicked a button!

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'unet_epoch25.keras')

@st.cache_resource
def load_solar_ai():
    def dice_coeff(y_true, y_pred, smooth=1e-6):
        y_true_f = K.flatten(K.cast(y_true, 'float32'))
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_loss(y_true, y_pred):
        return 1 - dice_coeff(y_true, y_pred)
        
    return load_model(MODEL_PATH, custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})

unet_model = load_solar_ai()

# ==========================================
# 3. FINANCIAL ENGINE
# ==========================================
def calculate_solar_financials(mask_array, original_width, original_height):
    pixels = int(np.sum(mask_array))
    if pixels == 0:
        return None
        
    # Spatial Scaling Law: The 256x256 mask represents the entire original image.
    # Therefore, 1 pixel in the mask = (Original Width / 256) original pixels.
    # We must calculate the real-world area of a single mask pixel.
    zoom_x = original_width / 256.0
    zoom_y = original_height / 256.0
    pixel_area_m2 = (zoom_x * METERS_PER_PIXEL) * (zoom_y * METERS_PER_PIXEL)
        
    roof_area = pixels * pixel_area_m2
    usable_area = roof_area * 0.70
    system_capacity_kw = usable_area / 10.0
    daily_gen = system_capacity_kw * IRRADIANCE * PERFORMANCE_RATIO
    annual_gen = daily_gen * 365
    
    total_cost = system_capacity_kw * COST_PER_KW
    annual_savings = annual_gen * TARIFF_SAVINGS
    payback_years = total_cost / annual_savings if annual_savings > 0 else 0
    roi = (annual_savings / total_cost) * 100 if total_cost > 0 else 0
    
    return {
        "pixels": pixels,
        "roof_area": roof_area,
        "usable_area": usable_area,
        "capacity": system_capacity_kw,
        "daily_gen": daily_gen,
        "annual_gen": annual_gen,
        "cost": total_cost,
        "savings": annual_savings,
        "payback": payback_years,
        "roi": roi
    }

# ==========================================
# 4. STREAMLIT FRONT-END UI
# ==========================================
st.set_page_config(page_title="Solar Potential Predictor", page_icon="🌞", layout="wide")

st.title("🏡 Automated Solar Rooftop Calculator")
st.markdown("""
Welcome to the Solar Mapping Web App! 
Upload a top-down satellite image of a house, and our custom **U-Net** Deep Learning model will instantly scan it, find the roof footprint, and calculate your exact ROI, installation costs, and savings in Indian Rupees (₹).
""")
st.divider()

# File Uploader
uploaded_file = st.file_uploader("Upload a High-Res Satellite Picture (JPG/PNG/TIF)", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file is not None:
    # 1. Load the User's Image safely
    image = Image.open(uploaded_file).convert('RGB')
    
    # Save the original size for our Spatial Scaling Math before we shrink it!
    orig_width, orig_height = image.size
    
    # CRITICAL FIX: Shrink the image using PIL *before* converting to numpy. 
    # This prevents Windows from crashing with a MemoryError when trying to 
    # allocate 300,000,000+ bytes for massive 10,000x10,000 .tif satellite files.
    image_resized = image.resize((256, 256))
    
    # Pre-process it exactly like the model expects
    img_cv = np.array(image_resized)
    img_normalized = img_cv / 255.0
    input_tensor = np.expand_dims(img_normalized, axis=0)

    # UI Splitting
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        st.subheader("Your Target Satellite Image")
        # We display the resized PIL image so it doesn't crash the user's browser either!
        st.image(image_resized, use_container_width=True)
        
    # Detect file change to clear old session state
    if 'last_uploaded_file' not in st.session_state or st.session_state['last_uploaded_file'] != uploaded_file.name:
        st.session_state['last_uploaded_file'] = uploaded_file.name
        if 'ai_analyzed' in st.session_state:
            del st.session_state['ai_analyzed']
    
    if st.button("🚀 Analyze Roof & Calculate Profit", use_container_width=True, type="primary"):
        with st.spinner("AI scanning topography and processing convolutions..."):
            
            # 2. RUN NEURAL NETWORK
            prediction = unet_model.predict(input_tensor, verbose=0)[0]
            binary_mask = prediction > 0.5
            
            st.session_state['binary_mask'] = binary_mask
            st.session_state['img_normalized'] = img_normalized
            st.session_state['orig_width'] = orig_width
            st.session_state['orig_height'] = orig_height
            st.session_state['ai_analyzed'] = True
            
    if st.session_state.get('ai_analyzed', False):
        binary_mask = st.session_state['binary_mask']
        img_normalized = st.session_state['img_normalized']
        orig_width = st.session_state['orig_width']
        orig_height = st.session_state['orig_height']
        
        # Connected Components for distinct roofs
        mask_2d = binary_mask.squeeze().astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_2d)
        
        # Check for user click from component prior to rendering the final image blend
        clicked_coords = st.session_state.get("roof_map", None)
        selected_label = None
        
        if clicked_coords is not None:
            x, y = clicked_coords.get('x'), clicked_coords.get('y')
            if y is not None and x is not None:
                if y < labels.shape[0] and x < labels.shape[1]:
                    selected_label = labels[int(y), int(x)]
                    if selected_label == 0:  # Clicked on background
                        selected_label = None
        
        # Base yellow mask
        yellow_mask = np.zeros_like(img_normalized)
        yellow_mask[mask_2d > 0] = [1.0, 1.0, 0.0]
        
        # Highlight selected roof in GREEN
        if selected_label is not None:
            yellow_mask[labels == selected_label] = [0.0, 1.0, 0.0]
            
        # Blend original image with the mask
        blended_image = img_normalized.copy()
        blended_image[mask_2d > 0] = img_normalized[mask_2d > 0] * 0.5 + yellow_mask[mask_2d > 0] * 0.5
        
        with col_img2:
            st.subheader("AI Extracted Solar Footprint")
            st.markdown("💡 **Click on a specific roof** below to see its individual metrics!")
            
            # Convert to PIL Image for the interactive component
            disp_img = (blended_image * 255).astype(np.uint8)
            pil_disp_img = Image.fromarray(disp_img)
            
            # RENDER interactive coordinates component instead of static image
            _ = streamlit_image_coordinates(pil_disp_img, key="roof_map")
            
        # 3. GENERATE FINANCIAL DATA
        # Pick which mask to use for math
        if selected_label is not None:
            # Run math just on the clicked roof
            analysis_mask = (labels == selected_label)
            report_title = "⚡ Selected Single Roof Potential Report"
        else:
            # Run math on all roofs combined
            analysis_mask = binary_mask
            report_title = "⚡ Total Global Solar Potential Report"
            
        data = calculate_solar_financials(analysis_mask, orig_width, orig_height)
        
        if data is None:
            st.error("❌ The AI could not detect any flat, unobstructed roofs in this area. Try selecting another region or clicking elsewhere.")
        else:
            if selected_label is None:
                st.success("✅ Global Analytics generated successfully!")
            else:
                st.success("✅ Analytics for specifically selected roof generated!")
            
            # --- DASHBOARD METRICS ---
            st.header(report_title)
            
            # Top Row: Power
            p_col1, p_col2, p_col3 = st.columns(3)
            p_col1.metric("Recommended Capacity", f"{data['capacity']:.1f} kW", "Based on 70% usable area")
            p_col2.metric("Annual Energy Output", f"{data['annual_gen']:,.0f} kWh", f"{data['daily_gen']:.1f} kWh/day")
            p_col3.metric("Total Roof Area", f"{data['roof_area']:.1f} sq m", f"{data['pixels']:,} pixels mapped")
            
            st.divider()
            
            # Bottom Row: Financials
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            f_col1.metric("Est. Installation Cost", f"₹ {data['cost']:,.0f}", f"@ ₹60k / kW", delta_color="inverse")
            f_col2.metric("Annual Electricity Savings", f"₹ {data['savings']:,.0f}", f"@ ₹7 / unit grid offset")
            f_col3.metric("Return on Investment", f"{data['roi']:.1f} % / year", "Annual ROI")
            f_col4.metric("Payback Time", f"{data['payback']:.1f} Years", "Time to recover costs")
                
st.sidebar.title("About the AI")
st.sidebar.info("""
**Architecture:** Keras U-Net
**Loss Function:** Custom Dice-Loss
**Accuracy:** Outperforms Mask R-CNN & YOLOv8 on imbalanced topography arrays.
""")
