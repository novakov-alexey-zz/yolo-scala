import java.nio.file.Paths

object YoloModel:  
  val NetWidth = 416
  val NetHeight = 416
  val Channels = 3  
  val BoundingBox = 85
  val BoxHeader = 5
  val ClassCount = BoundingBox - BoxHeader
  val ObjectnessIndex = 4
  val Anchors = 3  
  val OnnxModelPath = Paths.get("models", "yolov3.onnx")
  val labels = Array("person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")

enum YoloGrid(val name: String, val size: Int, val anchors: Array[Int]):
  case Small   extends YoloGrid("conv_105", 52, Array(10, 13, 16, 30, 33, 23))
  case Medium  extends YoloGrid("conv_93", 26, Array(30, 61, 62, 45, 59, 119))
  case Large   extends YoloGrid("conv_81", 13, Array(116, 90, 156, 198, 373, 326))

def toYoloGrid(key: String) = 
  if key == YoloGrid.Small.name then YoloGrid.Small
  else if key == YoloGrid.Medium.name then YoloGrid.Medium
  else YoloGrid.Large

case class BoundBox(xmin: Int, ymin: Int, xmax: Int, ymax: Int, objectness: Float, classes: Array[Float])
