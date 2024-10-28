# Fundamental of Data Science in Sapienza - Homework Repository

## Overview
This repository contains assignments for the **Fundamental of Data Science** (FDS) course. Each homework assignment covers the key topics we'll see during our lectures. This README will be updated as new assignments are added.

## Table of Contents
1. [Assignments](#assignments)
   - [Homework 1](#homework-1-image-filtering-and-object-identification)
   - [Homework 2](#homework-2)
   - [Future Assignments](#future-assignments)
2. [References](#references)

## Assignments

### Homework 1: Image Filtering and Object Identification

| **1.** | **Image Filtering** | It's a fundamental process in image processing used to enhance features, suppress noise, or extract information from an image. It serves as a basis for more complex image analysis tasks. |
|-------|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Number** | **Subsection** | **Description** |
| 1.0 | **Warm-Up** | Introduction to convolution and its properties, helping to set the theorical groundwork of filtering techniques. |
| 1.1 | **1D Filters** | Basics of one-dimensional filtering, including Gaussian filters for smoothing and Laplacian filters for edge detection. |
| 1.2 | **2D Filters** | Explore two-dimensional filtering for direct application to images, using techniques like Gaussian and Laplacian filtering to enhance, smooth, or emphasize specific image features. |

| **2.** | **Multi-Scale Image Representations** | Tecniques used to analyze images at different levels of detail to capture important features more effectively. |
|-------|----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Number** | **Subsection** | **Description** |
| 2.1 | **Prewitt Operator** | Introduces the Prewitt operator, a gradient-based method used for edge detection|
| 2.2 | **Canny Edge Detector & Template Matching** | Apply the Canny edge detection method, a multi-step process for identifying edges while minimizing noise. |
| 2.3 | **Harris Corner Detector** | It identifies key points (corners) in an image, making it useful for recognizing distinctive features. |
| 2.4 | **Gaussian Pyramid & Aliasing** | Gaussian pyramids are used to analyze images at various resolutions. This section also discusses aliasing, which can distort images if high-frequency information is not properly handled during downscaling. |
| 2.5 | **Multi-Scale Template Matching** | Combining template matching with Gaussian pyramids, to identify objects across scales (varying image resolutions). |

| **3.** | **Object Identification** | Techniques to identify and differentiate objects within images using histogram analysis and similarity metrics. |
|-------|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Number** | **Subsection** | **Description** |
| 3.1 | **3D Joint Color Histogram** | Foundation for color-based object identification. |
| 3.2 | **Types of Histograms** | To compare images based on intensity, color, or other feature distributions. |
| 3.3 | **Histogram Metrics** | Calculate similarity between histograms to assess how closely two images match based on their histogram features. |
| 3.4 | **Image Retrieval** | To identify and return similar images from a dataset based on key features. |
| 3.5 | **Report** | Summarize the findings, discussing each step and technique applied, and analyzing the performance and accuracy of each approach. |

| **4.** | **Performance Evaluation** | Evaluating the performance of the implemented techniques and metrics to assess their effectiveness in object detection and identification. |
|-------|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Number** | **Subsection** | **Description** |
| 4.1 | **Evaluation** | Analyze standardized evaluation metrics. |
| 4.2 | **Nearest Neighbors** | It compares a target image to others in a dataset, ranking them by similarity. |
| 4.3 | **Retrieval Metrics** | Measure retrieval success using metrics like precision, recall, and F1 score, evaluating the effectiveness of image retrieval methods. |
| 4.4 | **Analysis & Report** | Provide a final report on the evaluation metrics and results, interpreting how well each method performed and identifying areas for improvement. |

**Location**: [HW1_v3.ipynb](HW1_v3.ipynb)

### Homework 2

### Future Assignments

### References
