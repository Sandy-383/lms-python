**Project Requirements Summary**
1. **System**: Windows 10/11 with Python 3.8+ (UTF-8 configured).
2. **Deep Learning**: PyTorch (`torch`, `torchvision`) with CUDA support for GPU training.
3. **Computer Vision**: OpenCV (`opencv-python`) and PIL (`pillow`) for image processing.
4. **Data Handling**: NumPy (`numpy`) for arrays and Pandas (`pandas`) for metadata CSVs.
5. **Application**: Streamlit (`streamlit`) for the real-time web interface.
6. **Utilities**: Matplotlib (`matplotlib`) for plotting and TQDM (`tqdm`) for progress bars.
7. **Hardware**: NVIDIA RTX GPU (recommended for C3D) and a standard Webcam.
8. **Installation**: `pip install torch torchvision numpy pandas opencv-python streamlit matplotlib tqdm "protobuf<5.0.0" mediapipe`
   *(Note: The `"protobuf<5.0.0"` is CRITICAL to prevent crashes found in newer versions)*
