The project has been successfully tested on:

    --Cmake 3.29
    --CUDA 11.8
    --cudnn 8.9.7
    --TensorRT 8.6.1
    --Opencv 4.9.0
    --ubuntu 20.04
    --ros-noetic
    --Realsense D455

Before running this project, you are supposed to install:

* [librealsense](https://github.com/IntelRealSense/realsense-ros)

* [realsense-ros](https://github.com/IntelRealSense/librealsense)

Method to get **.onnx** model:

    conda activate pytorch
    python3
    from ultralytics import YOLO
    model = YOLO('yolo11x.pt')
    model.export(format='onnx',imgsz=(480,864))

Method to get **.engine** model:

* for FP32:

      ~/TensorRT/bin/trtexec --onnx=yolo11x.onnx --saveEngine=yolo11x.engine

* for FP16:

      ~/TensorRT/bin/trtexec --onnx=yolo11x.onnx --saveEngine=yolo11x.engine --fp16

* for INT8:

      ~/TensorRT/bin/trtexec --onnx=yolo11x.onnx --saveEngine=yolo11x.engine --int8


Method to compile:

    mkdir ~/catkin_ws
    cd ~/catkin_ws
    mkdir src && cd src
    git clone https://github.com/cll-git/YOLO11-TensorRT-ROS.git
    cd ..
    catkin_make

Method to run:

    1. Run 'source devel/setup.bash'

    2. Edit the 'params.yaml' file in the 'config' folder,
       and set the 'image_topic_in' to the topic of your camera. 

Edit the params **'kInputH'** and **'kInputW'** in **/include/config.h** to suit your camera.

Besides, set the **'model_path'** to the path of the model file.

    3. Edit the 'yolov11.launch' file in the 'launch' folder, 
       and set the path of the launch file correctly, to run the camera.

    4. Run 'roslaunch yolov11 yolov11.launch'

After successfully running the launch file, choose the topic '**/yolo/detect_result**' in rqt_image_view to view the segmentation results.
