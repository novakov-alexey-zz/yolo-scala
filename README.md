# Yolo v3 Inference in Scala

Example project to load ONNX [Yolov3 model](https://pjreddie.com/media/files/papers/YOLOv3.pdf) and infer some boxes in Scala.

![city-out-image](city-out.jpg)

## Get ONNX model for Yolov3

1. Clone this repository [https://github.com/novakov-alexey/yolo-keras](https://github.com/novakov-alexey/yolo-keras)

2. Go into the repository root folder

3. Create environment via `make` command (Makefile is provided):
```bash
make install
```
4. Download weights from here: [https://pjreddie.com/media/files/yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) and put them into `yolo-keras/models` directory

5. Run this script to create Keras model:
```bash
python create_model.py
```
6. Run the following command to create ONNX model:
```bash
make save-2-onnx
```
7. Copy ONNX model file to the `yolo-scala/models` directory of the `yolo-scala` repository