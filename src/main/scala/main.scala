import io.kjaer.compiletime.*
import org.emergentorder.compiletime.*
import org.emergentorder.onnx.Tensors.*
import org.emergentorder.onnx.backends.*

import org.bytedeco.opencv.global.opencv_imgcodecs.{imread, imwrite}
import org.bytedeco.opencv.global.opencv_imgproc.resize
import org.bytedeco.opencv.opencv_core.{Mat, Size, Scalar}
import org.bytedeco.javacpp.indexer.FloatRawIndexer
import org.bytedeco.opencv.global.opencv_core.{divide, CV_32FC4}
import org.bytedeco.opencv.global.opencv_imgproc.*

import ai.onnxruntime.{OrtEnvironment, OrtSession}
import ai.onnxruntime.OnnxTensor

import scala.jdk.CollectionConverters.*
import scala.util.Using
import scala.reflect.ClassTag

import java.nio.file.{Files, Paths, Path}
import java.nio.FloatBuffer

val TensorShapeDenotation = "Batch" ##: "Height" ##: "Width" ##: "Channel" ##: TSNil
val TensorDenotation: String & Singleton = "Image"
val ImageWidth = 416
val ImageHeight = 416
val Channels = 3
val GridSize: Dimension = 52
val BoundingBoxes = 85
val Anchors = 3
val BoundingBoxesDim: Dimension = (BoundingBoxes * Anchors).asInstanceOf[Dimension]
val ModelPath = Paths.get("models", "yolov3.onnx")

val env = OrtEnvironment.getEnvironment()
val session = env.createSession(ModelPath.toString, OrtSession.SessionOptions())

enum YoloGrid(val name: String, val size: Int, val anchors: Array[Int]):
  case Small   extends YoloGrid("conv_105", 52, Array(10, 13, 16, 30, 33, 23))
  case Medium  extends YoloGrid("conv_93", 26, Array(30, 61, 62, 45, 59, 119))
  case Large   extends YoloGrid("conv_81", 13, Array(116, 90, 156, 198, 373, 326))

case class BoundBox(xmin: Float, ymin: Float, xmax: Float, ymax: Float, objectness: Float, classes: Array[Float])

def shape(batch: Int) =
  batch #: ImageHeight #: ImageWidth #: Channels #: SNil

def getModel(path: Path = ModelPath) =
  val bytes = Files.readAllBytes(path)
  ORTModelBackend(bytes)

def predict(images: Array[Float], model: ORTModelBackend, batch: Dimension = 1) =
  val input = Tensor(images, TensorDenotation, TensorShapeDenotation, shape(batch))    
  model.fullModel[
    Float, 
    "ImageClassification", 
    "Batch" ##: "BoundingBoxes" ##: TSNil, 
    batch.type #: GridSize.type #: GridSize.type #: BoundingBoxesDim.type #: SNil
  ](Tuple(input))

def toYoloGrid(key: String) = 
  if key == YoloGrid.Small.name then YoloGrid.Small
  else if key == YoloGrid.Medium.name then YoloGrid.Medium
  else YoloGrid.Large

def predict(images: Array[Float], batch: Long): Map[YoloGrid, Array[Float]] =     
  var tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(images), Array(batch, ImageHeight, ImageWidth, Channels))
  val inputs = Map("input_1:0" -> tensor).asJava
  Using.resource(session.run(inputs)) { result =>
    result.iterator().asScala.toList
      .map{ r =>
        val key = toYoloGrid(r.getKey)
        key -> r.getValue.asInstanceOf[OnnxTensor].getFloatBuffer.array()
      }
      .toMap
  }

// TODO: clone initial image
def toModelInput(image: Mat): Array[Float] =  
  resize(image, image, Size(ImageHeight, ImageWidth))
  toArray(scale(image))

def scale(img: Mat): Mat =
  val out = Mat()
  img.assignTo(out, CV_32FC4)
  divide(255d, out, out)
  out

def toArray(mat: Mat): Array[Float] =
  val w = mat.cols
  val h = mat.rows
  val c = mat.channels
  val rowStride = w * c

  val result = new Array[Float](rowStride * h)
  val indexer = mat.createIndexer[FloatRawIndexer]()
  var off = 0
  var y = 0
  while (y < h)
    indexer.get(y, result, off, rowStride)
    off += rowStride
    y += 1

  result

def sigmoid(x: Float): Float = 
  1 / (1 + math.exp(-x)).toFloat

def getBoxes(
  prediction: Array[Array[Array[Array[Float]]]], 
  anchors: Array[Int], 
  gridSize: Int, 
  imageWidth: Int, 
  imageHeight: Int
): Array[BoundBox] = 
  val threshold = 0.6
  val boxes = for 
    (gridRow, row) <- prediction.zipWithIndex
    (boxes, col) <- gridRow.zipWithIndex
    (box, boxId) <- boxes.zipWithIndex 
  yield
    val objectness = sigmoid(box(4))
    if objectness > threshold then
      val (x, y, w, h) =  (sigmoid(box(0)), sigmoid(box(1)), box(2), box(3))
      val centerX = (row + x) / gridSize
      val centerY = (col + y) / gridSize
      val imageWidth = anchors(2 * boxId + 0) * math.exp(w).toFloat / gridSize
      val imageHeight = anchors(2 * boxId + 1) * math.exp(h).toFloat / gridSize
      val classes = box.slice(5, box.length).map(sigmoid)
      val xmin = centerX - imageWidth / 2 / imageWidth
      val xmax = centerX + imageWidth / 2 / imageWidth
      val ymin = centerY - imageHeight / 2 / imageHeight
      val ymax = centerY + imageHeight / 2 / imageHeight
      Some(BoundBox(xmin, ymin, xmax, ymax, objectness, classes))
    else None
  boxes.flatten

extension [T: ClassTag: Numeric](a: Array[T])(using o: Ordering[T])
  def argsort: Array[Int] =
    a.zipWithIndex.sortBy(_._1).map(_._2).toArray

def intervalOverlap(intervalA: (Float, Float), intervalB: (Float, Float)) =
  val (x1, x2) = intervalA
  val (x3, x4) = intervalB
  if x3 < x1 then 
    if x4 < x1 then 0
    else math.min(x2, x4) - x1
  else
    if x2 < x3 then 0
    else math.min(x2, x4) - x3

def bboxIou(b1: BoundBox, b2: BoundBox): Float =
  val intersect_w = intervalOverlap((b1.xmin, b1.xmax), (b2.xmin, b2.xmax))
  val intersect_h = intervalOverlap((b1.ymin, b1.ymax), (b2.ymin, b2.ymax))
  val intersect = intersect_w * intersect_h
  val (w1, h1) = (b1.xmax - b1.xmin, b1.ymax - b1.ymin)
  val (w2, h2) = (b2.xmax - b2.xmin, b2.ymax - b2.ymin)
  val union = w1 * h1 + w2 * h2 - intersect
  intersect / union

def nonMaximalSupression(boxes: Array[BoundBox], classCount: Int, nmsThresh: Float = 0.5): Unit =  
  for c <- (0 to classCount) do
    val sortedIndices = boxes.map(- _.classes(c)).argsort

    for i <- (0 to sortedIndices.length) do
      val indexI = sortedIndices(i)

      if boxes(indexI).classes(c) != 0 then
        for j <- (i + 1 to sortedIndices.length) do
          val indexJ = sortedIndices(j)
          
          if bboxIou(boxes(indexI), boxes(indexJ)) >= nmsThresh then 
            boxes(indexJ).classes(c) = 0

def gridPredictions(grid: YoloGrid, outputs: Map[YoloGrid, Array[Float]]) =
  outputs(grid)
    .grouped(BoundingBoxes).toArray
    .grouped(Anchors).toArray
    .grouped(grid.size).toArray


@main 
def main = 
  val file = Paths.get("zebra.jpg")  
  val image = imread(file.toString)
  val (imageWidth, imageHeight) = (image.size.width, image.size.height)
  val input = toModelInput(image)  
  val model = getModel()
  val outputs = predict(input, 1)
  
  val classCount = BoundingBoxes - 5

  val boxes = YoloGrid.values.map { yg =>
    val predictions = gridPredictions(yg, outputs)    
    getBoxes(predictions, yg.anchors, yg.size, imageWidth, imageHeight)
  }.flatten

  nonMaximalSupression(boxes, classCount)


  