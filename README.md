# mocap_motion_Analysis
📝 Project Description
This project is a high-performance gait analysis tool that uses computer vision to create professional-quality 3D skeleton visualizations from standard 2D video footage. The application processes video frame-by-frame, identifies key body landmarks, and renders a smooth, calibrated 3D skeleton in three anatomical planes.

The goal is to provide a low-cost, accessible alternative to expensive motion capture hardware like Vicon and Qualisys, making advanced movement analysis available to researchers, physical therapists, and athletes with only a simple video camera.

✨ Features
Multi-Plane Visualization: Renders the skeleton simultaneously in the Coronal, Sagittal, and Transverse planes, providing a complete view of movement.

Advanced Landmark Smoothing: Implements an Exponential Moving Average (EMA) filter to remove jitter and produce fluid, professional-quality animation.

Robust Pseudo-Calibration: Accurately scales the skeleton's height and depth using an average-based calibration system, providing a stable and consistent sense of perspective.

Dynamic Graphics: The skeleton's lines and joints are rendered with dynamic thickness and a subtle "glow" effect, enhancing the visual fidelity based on perspective.

Distinct Markers: Key joints like the head and feet are highlighted with unique colors and sizes to draw attention to critical points of analysis.
