import Dependencies._

Global / onChangedBuildSource := ReloadOnSourceChanges

ThisBuild / scalaVersion     := "3.0.0"
ThisBuild / version          := "0.1.0-SNAPSHOT"

lazy val root = (project in file("."))
  .settings(
    name := "yolo-scala",
    libraryDependencies ++= Seq(
      onnxScala, opencv
    )
  )
