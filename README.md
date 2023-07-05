# EasyGaze3D: Towards Effective and Flexible 3D Gaze Estimation from a Single RGB Camera 
## Usage

To use your own webcam for real-time gaze estimation:

1. Build NMS, Sim3DR and mesh render in 3DDFA_V2 (Thanks to this amazing work for 3D face model reconstruction!)

   ---> cd ./TDDFA_V2/
   
   ---> sh ./build.sh

2. Calibrate subject-specific features

   ---> Set cfg.EasyCali.subj_idx
   
   ---> Set cfg.EasyCali.to_save to True (Recommend to False before getting ready to start)

   ---> Run calibration/capture_images.py (Should fixate at the camera lens center, then press "s" on keyboard to capture 50 images continuously)

   ---> Set camera intrinsic parameters in camera_intrinsic_params.py (Calibrate the webcam before)

   ---> Run calibration/camera_intrinsic_params.py

   ---> Run calibrate_features.py (Set cfg.EasyCali.visualize to True to check the gaze lines)

3. Real-time gaze estimation

   ---> Set cfg.eye_mask_type to simple for higher fps if needed

   ---> Set cfg.render to True for facial shape visualization if needed

   ---> Run demo_webcam.py

## Acknowledgement
The 3D face model reconstruction in this work is modified from [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2).


