# <div align="center">

# 🌞 Solar Flair
### *Next-Gen AI-Powered Solar Potential Analysis*

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

![Solar Flair Dashboard Mockup](/Users/sarvyagyaprakash/.gemini/antigravity/brain/a3213a05-6037-4a96-ac7a-75b536f08cd0/solar_flair_dashboard_mockup_1778491399187.png)

</div>

## 🚀 Overview

**Solar Flair** is a state-of-the-art computer vision application designed to democratize solar energy adoption. By leveraging a custom-trained **U-Net Neural Network**, the platform automatically segments rooftops from satellite imagery and provides high-precision financial ROI projections, energy yield estimations, and installation cost breakdowns.

Whether you are a homeowner exploring solar options or a commercial developer conducting large-scale feasibility studies, Solar Flair provides the insights needed to make data-driven energy decisions.

---

## ✨ Key Features

-   **🎯 AI Rooftop Segmentation**: Uses a deep-learning U-Net model to identify flat, unobstructed roof areas with pixel-perfect accuracy.
-   **🖱️ Interactive Selection**: Click on specific roof segments to see individual metrics or view global analytics for the entire area.
-   **📊 Real-time Financial Engine**: Instantly calculates installation costs, annual savings, payback periods, and ROI in local currency (₹).
-   **📈 Energy Yield Prediction**: Estimates annual kWh output based on spatial scaling laws and regional irradiance constants.
-   **🛰️ Multi-Format Support**: Processes high-resolution satellite imagery in JPG, PNG, and professional GeoTIFF formats.
-   **💎 Premium UI**: A sleek, dark-mode glassmorphism interface built for maximum clarity and user engagement.

---

## 🛠️ Tech Stack

-   **Frontend**: [Streamlit](https://streamlit.io/) (Interactive Dashboard)
-   **Deep Learning**: [TensorFlow](https://tensorflow.org/) / [Keras](https://keras.io/) (U-Net Architecture)
-   **Image Processing**: [OpenCV](https://opencv.org/) & [Pillow](https://python-pillow.org/)
-   **Interactive Components**: [Streamlit Image Coordinates](https://github.com/vivien000/streamlit-image-coordinates)
-   **Analytics**: [NumPy](https://numpy.org/) (Spatial scaling & financial modeling)

---

## 📐 How It Works

### 1. Neural Network Architecture
The core of Solar Flair is a **U-Net** trained specifically on satellite topography datasets. It utilizes a custom **Dice Loss** function to handle class imbalance, ensuring that even small or complex rooftops are accurately captured.

### 2. Spatial Scaling Law
The application converts pixel-level data into real-world square meters using a precise scaling factor:
$$ \text{Area (m}^2\text{)} = \sum \text{Pixels} \times \left( \frac{\text{Width}_{\text{orig}}}{\text{Width}_{\text{mask}}} \times \text{Meters Per Pixel} \right)^2 $$

### 3. Financial Modeling
The ROI engine uses the following parameters (configurable in `app.py`):
-   **Tariff Savings**: ₹7.0 / unit grid offset
-   **Installation Cost**: ₹60,000 / kW
-   **Performance Ratio**: 75% efficiency coefficient

---

## ⚙️ Installation

To run Solar Flair locally, ensure you have Python 3.9+ installed.

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/Solar-Flair.git
   cd Solar-Flair
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Application**
   ```bash
   streamlit run app.py
   ```

---

## 📂 Project Structure

```text
.
├── app.py                # Main Streamlit application & Financial Engine
├── unet_epoch25.keras    # Pre-trained U-Net weights
├── requirements.txt      # Project dependencies
└── README.md             # Aesthetic documentation (You are here!)
```

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">

**Developed with ❤️ for a Greener Future.**

[⭐ Star this repo](https://github.com/your-username/Solar-Flair) | [🌐 Visit Website](https://solar-flair.streamlit.app/)

</div>
