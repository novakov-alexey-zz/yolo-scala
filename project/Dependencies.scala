import sbt._

object Dependencies {
  lazy val onnxScala = "org.emergent-order" %% "onnx-scala-backends" % "0.15.0" withSources()
  lazy val onnxJava  = "com.microsoft.onnxruntime" % "onnxruntime" % "1.7.0"
  lazy val opencv = "org.bytedeco" % "javacv-platform" % "1.5.5"
}
