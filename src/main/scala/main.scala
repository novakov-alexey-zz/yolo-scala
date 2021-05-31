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

import java.nio.file.{Files, Paths, Path}

val tensorShapeDenotation = "Batch" ##: "Height" ##: "Width" ##: "Channel" ##: TSNil
val tensorDenotation: String & Singleton = "Image"
val ImageWidth = 416
val ImageHeight = 416
val Channels = 3
val gridSize: Dimension = 52
val lastDim: Dimension = 255

def shape(batch: Int) =
  batch #: ImageHeight #: ImageWidth #: Channels #: SNil

def getModel(path: Path = Paths.get("models", "yolov3.onnx")) =
  val bytes = Files.readAllBytes(path)
  ORTModelBackend(bytes)

def predict(images: Array[Float], model: ORTModelBackend, batch: Dimension = 1) =
  val input = Tensor(images, tensorDenotation, tensorShapeDenotation, shape(batch))    
  model.fullModel[
    Float, 
    "ImageClassification", 
    "Batch" ##: "Class" ##: TSNil, 
    batch.type #: gridSize.type #: gridSize.type #: lastDim.type #: SNil
  ](Tuple(input))

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
  val out = predict(input, model)
  println(out)
  println(out.data.length)