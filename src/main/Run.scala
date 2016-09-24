package main

import scala.collection.mutable
import scala.io.Source
import scala.util.Random

/**
  * Created by masoud on 9/12/16.
  */
object Run {

  def readInputs(filepath: String): List[List[String]] = {
    Source
      .fromFile(filepath)
      .getLines
      .map(line => line.split(",").toList)
      .toList
  }

  def prepareInputData(data: List[List[String]]): List[(List[Double], String)] = {
    val input =
      data
        .filter(_.size == 5)
        .map { case entry =>
          val (d, tag) = entry.splitAt(4)
          (d.map(_.toDouble), tag.head)
        }

    // normalize input
    val columnarInput = for {
      i <- 0 to input.head._1.size - 1
      column = input.map(_._1(i))
      min = column.min
      max = column.max
      normalizedColumn = column.map(e => (e - min) / (max  - min))
    } yield normalizedColumn

    val normalizedData: mutable.MutableList[List[Double]] = mutable.MutableList.fill(columnarInput.head.size)(List.empty)

    for {
      i <- 0 to columnarInput.head.size - 1
      j <- 0 to columnarInput.size - 1
    } (normalizedData.update(i, normalizedData(i) ++ List(columnarInput(j)(i))))

    normalizedData.toList zip input.map(_._2)
  }

  def splitToTestAndTrain(processedInput: List[(List[Double], String)]): (Map[String, List[List[Double]]], Map[String, List[List[Double]]]) = {
    var train = Map[String, List[List[Double]]]()
    var test = Map[String, List[List[Double]]]()

    processedInput
      .map { case (data, tag) => (tag, data) }
      .groupBy{ case (tag, data) => tag }
      .mapValues[List[List[Double]]] (it => it.map(e => e._2))
      .foreach { case (tag, data) =>
        val (d, t) = data.splitAt(40)
        train = train + (tag -> d)
        test = test + (tag -> t)
      }

    (train, test)
  }

  def createWeights(nodePerLayer: List[Int], b: Boolean): List[List[List[Double]]] = {
    val random = scala.util.Random

    // the first list contains the weights of each layer : size = # layers - 1
    // the second list contains the weights of each node in the next layer : size = # nodes in the next layer
    // the third list contains the weight of the connection to the node in the next layer : size = num nodes in the current level + 1 for bias
    val weights: mutable.MutableList[List[List[Double]]] = mutable.MutableList.fill(nodePerLayer.size - 1)(List.empty)

    def createWeightForLayer(from: Int, to: Int): List[List[Double]] = {
      val layerWeights = List.fill(to)(List.fill(from + 1)(1.0))
      layerWeights
        .map(l =>
          l.map( w =>
            if (b) 0 else 2 * random.nextDouble - 1
          )
        )
    }

    for {
      i <- 0 to nodePerLayer.size - 1 - 1
    } {
      weights.update(i, createWeightForLayer(nodePerLayer(i), nodePerLayer(i + 1)))
    }

    weights.toList
  }

  def addWeights(w1: List[List[List[Double]]], w2: List[List[List[Double]]]): List[List[List[Double]]] = {
    (w1 zip w2)
      .map { case (e1, e2) =>
        (e1 zip e2)
            .map { case (e11, e22) =>
              (e11 zip e22)
                .map {case (e111, e222) => e111 + e222}
            }
      }
  }

  def main(args: Array[String]) = {
    val irises: Map[String, List[Double]] = Map(
      "Iris-setosa" -> List(0, 0),
      "Iris-versicolor" -> List(1, 0),
      "Iris-virginica" -> List(1, 1))

    // input data config
    val input = readInputs("data/iris/iris.data")
    val processedInput = prepareInputData(input)
    val (train, test) = splitToTestAndTrain(processedInput)
    val testRun = false
    val trainingData: List[(String, List[Double])] =
      if (!testRun)
        Random.shuffle(train.toList
          .flatMap { case (name, dataPoints) =>
            dataPoints.map { case point => (name, point)}
          })
      else List(("Iris-versicolor", List(0.6666666666666666, 0.4583333333333333, 0.6271186440677966, 0.583333333333)))
    val testData: List[(String, List[Double])] =
      if (!testRun)
        test.toList
          .flatMap { case (name, dataPoints) =>
            dataPoints.map { case point => (name, point)}
          }
      else List(("Iris-versicolor",List(0.13888888888888887, 0.41666666666666663, 0.06779661016949151, 0.0)))

    println("trainingData" + trainingData)

    // algo config
    val learningFactor = 0.001
    val nodesPerLayer = List[Int](4, 10, 2)
    var weights = createWeights(nodesPerLayer, false)
    val TRAINING_ROUNDS = 5000

    var RoundError = 0.0

    println("******* beginning ******")
    for {(outputString, inputVector) <- testData} {
      val expectedOutput = irises.get(outputString).get
      val nn = new NeuralNetwork(nodesPerLayer, weights)
      val output = nn.runNetwork(inputVector)
      RoundError = RoundError + (output(1) zip expectedOutput).map { case (c, e) => Math.pow((c - e), 2)}.reduce(_ + _)
//      println("inputVector: " + inputVector)
//      println("beginning output: " + output)
//      println("expected output: " + expectedOutput)
    }
    println("RoundError: " + RoundError)

    var trainRoundCount = 0
    while (trainRoundCount < TRAINING_ROUNDS) {
      trainRoundCount += 1
      println(s"*** ROUND $trainRoundCount ***")

      var layerWeight = createWeights(nodesPerLayer, true)
      RoundError = 0.0

      for {(outputString, inputVector) <- trainingData} {
        val expectedOutput = irises.get(outputString).get
        val nn = new NeuralNetwork(nodesPerLayer, weights)
        val output = nn.runNetwork(inputVector)
        val error = nn.calcError(inputVector, expectedOutput, output)
        val weightUpdates = nn.backPropagate(inputVector, output, error, learningFactor)
        layerWeight = addWeights(weightUpdates, layerWeight)
        RoundError = RoundError + (output(1) zip expectedOutput).map { case (c, e) => Math.pow((c - e), 2)}.reduce(_ + _)

//        println("*** node ***")
//        println("weights: " + weights(1))
//        println("expectedOutput: " + expectedOutput)
//        println("output: " + output)
//        println("error: " + error)
//        println("weightUpdates: " + weightUpdates(1))

      }

      println("RoundError: " + RoundError)
      weights = addWeights(layerWeight, weights)
      layerWeight = createWeights(nodesPerLayer, true)
    }

    println("******* end ******")
    RoundError = 0
    for {(outputString, inputVector) <- testData} {
      val expectedOutput = irises.get(outputString).get
      val nn = new NeuralNetwork(nodesPerLayer, weights)
      val output = nn.runNetwork(inputVector)
      RoundError = RoundError + (output(1) zip expectedOutput).map { case (c, e) => Math.pow((c - e), 2)}.reduce(_ + _)
      println("inputVector: " + inputVector)
      println("end output: " + output(1))
      println("expected output: " + expectedOutput)
    }
    println("RoundError: " + RoundError)


//    val inputVector = List(1.0, 1.0, 1.0, 1.0)
//    val expectedOutput = List(1.0, 0.0)
//
//    val weights = createWeights(nodesPerLayer, false)
//    val nn = new NeuralNetwork(nodesPerLayer, weights)
//    println("expected output: " + expectedOutput)
//
//    val output = nn.runNetwork(inputVector)
//    println("output1: " + output)
//
//    val error = nn.calcError(inputVector, expectedOutput, output)
//    val weightUpdates = nn.backPropagate(inputVector, output, error, 0.1)
//    val newWeights = addWeights(weightUpdates, weights)
//
//    val nn2 = new NeuralNetwork(List[Int](4, 3, 2), newWeights)
//    val output2 = nn2.runNetwork(inputVector)
//    println("output2: " + output2)
  }


}
