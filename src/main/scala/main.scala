import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}
import org.bytedeco.javacpp.indexer.FloatRawIndexer
import org.bytedeco.opencv.global.opencv_core.{CV_32FC3, divide}
import org.bytedeco.opencv.global.opencv_imgcodecs.{imread, imwrite}
import org.bytedeco.opencv.global.opencv_imgproc.{rectangle, putText, LINE_AA, FONT_HERSHEY_SIMPLEX}
import org.bytedeco.opencv.global.opencv_imgproc.resize
import org.bytedeco.opencv.opencv_core.{Mat, Point, Scalar, Size}

import java.nio.FloatBuffer
import scala.jdk.CollectionConverters.*
import scala.reflect.ClassTag
import scala.util.Using

import YoloModel.*
import YoloInference.*
import YoloPostprocessing.*

type Tensor4D = Array[Array[Array[Array[Float]]]]

extension [T: ClassTag: Numeric](a: Array[T])(using o: Ordering[T])
  def argsort: Array[Int] =
    a.zipWithIndex.sortBy(_._1).map(_._2).toArray

object YoloInference:
  def predict(images: Array[Float], batch: Long, session: OrtSession, env: OrtEnvironment): Map[YoloGrid, Array[Float]] =
    val shape = Array(batch, NetHeight, NetWidth, Channels)
    var tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(images), shape)
    val inputs = Map("input_1:0" -> tensor).asJava
    Using.resource(session.run(inputs)) { outputs =>
      outputs.iterator().asScala.toList
        .map { out =>
          val key = toYoloGrid(out.getKey)
          val array = out.getValue.asInstanceOf[OnnxTensor].getFloatBuffer.array()
          key -> array
        }
        .toMap
    }

  def toModelInput(image: Mat): Array[Float] =
    val out = Mat()
    image.assignTo(out, CV_32FC3)
    resize(out, out, Size(NetHeight, NetWidth))
    toArray(out).map(_ / 255f)

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

object YoloPostprocessing:
  def sigmoid(x: Float): Float =
    1 / (1 + math.exp(-x)).toFloat

  def getBoxes(
    prediction: Tensor4D,
    anchors: Array[Int],
    gridSize: Int,
    imageWidth: Int,
    imageHeight: Int,
    threshold: Float
  ): Array[BoundBox] =
    for
      (gridRow, row) <- prediction.zipWithIndex
      (boxes, col) <- gridRow.zipWithIndex
      (box, boxId) <- boxes.zipWithIndex
      objectness = sigmoid(box(ObjectnessIndex))
      if objectness > threshold
    yield
      val (x, y, w, h) =  (sigmoid(box(0)), sigmoid(box(1)), box(2), box(3))
      val centerX = (col + x) / gridSize
      val centerY = (row + y) / gridSize
      val width = anchors(2 * boxId + 0) * math.exp(w).toFloat / NetWidth
      val height = anchors(2 * boxId + 1) * math.exp(h).toFloat / NetHeight
      val classes = box.drop(BoxHeader).map(sigmoid)
      val xmin = (centerX - width / 2) * imageWidth
      val ymin = (centerY - height / 2) * imageHeight
      val xmax = (centerX + width / 2) * imageWidth
      val ymax = (centerY + height / 2) * imageHeight
      BoundBox(xmin.toInt, ymin.toInt, xmax.toInt, ymax.toInt, objectness, classes)

  def intervalOverlap(intervalA: (Int, Int), intervalB: (Int, Int)) =
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
    intersect.toFloat / union

  def nonMaximalSupression(boxes: Array[BoundBox], classCount: Int, nmsThresh: Float = 0.5): Unit =
    for c <- (0 until classCount) do
      val sortedIndices = boxes.map(- _.classes(c)).argsort

      for i <- (0 until sortedIndices.length) do
        val boxI = sortedIndices(i)

        if boxes(boxI).classes(c) != 0 then
          for j <- (i + 1 until sortedIndices.length) do
            val boxJ = sortedIndices(j)
            if bboxIou(boxes(boxI), boxes(boxJ)) >= nmsThresh then
              boxes(boxJ).classes(c) = 0

  def gridPredictions(grid: YoloGrid, outputs: Map[YoloGrid, Array[Float]]) =
    outputs(grid)
      .grouped(BoundingBox).toArray
      .grouped(Anchors).toArray
      .grouped(grid.size).toArray

  def getPredictedBoxes(boxes: Array[BoundBox], labels: Array[String], thresh: Float) =
    boxes.flatMap { box =>
      for i <- (0 until labels.length)
      if box.classes(i) > thresh
      yield (box, labels(i), box.classes(i) * 100)
    }

  def drawBoxes(frame: Mat, relevantBoxes: Array[(BoundBox, String, Float)]) =
    for (box, label, score) <- relevantBoxes do
        val (y1, x1, y2, x2) = (box.ymin, box.xmin, box.ymax, box.xmax)
        val text = f"$label ($score%1.3f)"
        val thickness = 1
        val fontScale = 0.5
        val font = FONT_HERSHEY_SIMPLEX
        val color = Scalar(
          255 * math.random + 100, 
          255 * math.random + 100, 
          255 * math.random + 100, 
          0)
        rectangle(
          frame,
          Point(x1.toInt, y1.toInt),
          Point(x2.toInt, y2.toInt),
          color,
          thickness,
          LINE_AA,
          0
        )        
        putText(
          frame,
          text,
          Point(x1.toInt - 10, y1.toInt - 10),
          font,
          fontScale,
          color,
          thickness,
          LINE_AA,
          false
        )
@main 
def main =
  val image = imread("city.jpg")
  val (imageWidth, imageHeight) = (image.size.width, image.size.height)
  val input = toModelInput(image)
  val env = OrtEnvironment.getEnvironment()
  val session = env.createSession(OnnxModelPath.toString, OrtSession.SessionOptions())
  val outputs = predict(input, 1, session, env)
  
  // discard all boxes with object probability less than threshold
  val threshold = 0.6f
  
  val boxes = YoloGrid.values.flatMap { yg =>
    val predictions = gridPredictions(yg, outputs)    
    getBoxes(predictions, yg.anchors, yg.size, imageWidth, imageHeight, threshold)
  }  
  println(s"Got ${boxes.length} boxes with objects")

  nonMaximalSupression(boxes, ClassCount)
  val relevantBoxes = getPredictedBoxes(boxes, labels, threshold)  
  println(s"Remaining boxes: ${relevantBoxes.length}")  
  drawBoxes(image, relevantBoxes)
  val outFile = "city-out.jpg"
  imwrite(outFile, image)
  println(s"Image saved at '$outFile'")

  env.close()
  session.close()