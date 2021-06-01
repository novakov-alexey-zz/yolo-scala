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

import scala.jdk.CollectionConverters._
import scala.util.Using
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

enum YoloGrid(val name: String, val size: Int):
  case Small   extends YoloGrid("conv_105", 52)
  case Medium  extends YoloGrid("conv_93", 26)
  case Large   extends YoloGrid("conv_81", 13)  

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

def predict(images: Array[Float], batch: Long): Map[YoloGrid, Array[Float]] =     
  var tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(images), Array(batch, ImageHeight, ImageWidth, Channels))
  val inputs = Map("input_1:0" -> tensor).asJava
  Using.resource(session.run(inputs)) { result =>
    result.iterator().asScala.toList
      .map{ r => 
        val key =
          if r.getKey == YoloGrid.Small.name then YoloGrid.Small
          else if r.getKey == YoloGrid.Medium.name then YoloGrid.Medium
          else YoloGrid.Large

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

@main 
def main = 
  val file = Paths.get("zebra.jpg")  
  val image = imread(file.toString)
  val input = toModelInput(image)  
  val model = getModel()
  val outputs = predict(input, 1)
  
  val smallGrid = outputs(YoloGrid.Small)
    .grouped(BoundingBoxes).toArray
    .grouped(Anchors).toArray
    .grouped(YoloGrid.Small.size).toArray  