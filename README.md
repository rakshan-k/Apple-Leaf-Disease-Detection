Apple Disease Detection

Overview

This project focuses on the identification and classification of apple diseases using advanced image processing and machine learning techniques. The approach involves diseased region segmentation, feature extraction, and classification while considering real-world challenges such as variations in texture, color, noise, and backgrounds.

Disease Types

Apple Scab: Leaf spots turn dark brown to black, expand, and merge together.

Black Rot: Also known as "frog-eye leaf spot," characterized by circular spots with reddish edges and light tan interiors.

Cedar Apple Rust: Identified by bright orange-yellow spots on leaves.

Challenges

Variability in textures, colors, and diseased spot appearances.

Background noise affecting segmentation accuracy.

Disease patterns that closely resemble non-diseased regions.

Methodology

Diseased Region Segmentation: Extracting affected areas using region-based techniques.

Feature Extraction: Identifying key characteristics of diseases using color and texture analysis.

Classification: Applying machine learning algorithms to categorize diseases accurately.

Image Processing Techniques

Color Space Conversion: RGB to LAB for better human perception and CMYK for printing purposes.

Augmentation: Scaling (40%) and clockwise rotation to enhance training data diversity.

Color Marker Creation:

Utilized roipoly function for region selection.

Generated a binary mask of size 256x256.

Combined LAB image with the mask to create feature vectors.

Feature Vector Representation

First vector: a component corresponding to non-zero fitted values.

Energy-based features:

e1 = 1 / p(a)

e2 = 1 / p(b)

Histogram Analysis

Examines variations in intensity and color composition for robust classification.

Discrete Wavelet Transform (DWT)

Extracts frequency content from images to enhance disease differentiation.

Repository Structure

|-- dataset/             # Image dataset for training and validation
|-- segmentation/        # Code for diseased region segmentation
|-- feature_extraction/  # Feature extraction techniques applied
|-- classification/      # Classification models and training scripts
|-- results/             # Output images and performance evaluation
|-- README.md            # Project documentation

Requirements

Ensure the following dependencies are installed:

pip install numpy opencv-python matplotlib scikit-learn tensorflow

Usage

Preprocess Images: Run segmentation scripts to extract diseased regions.

Feature Extraction: Convert images to LAB space and apply vectorization.

Train Model: Use extracted features to train a classifier.

Evaluate Performance: Compare classification results using accuracy metrics.

Contact

For questions or contributions, feel free to reach out via GitHub Issues.
