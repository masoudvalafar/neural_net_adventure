package main

import scala.collection.mutable

/**
  * Created by masoud on 9/13/16.
  */
class NeuralNetwork(nodePerLayer: List[Int], weights: List[List[List[Double]]]) {


  //weights.foreach(a => println("layer: " + a))

  def runNetwork(input: List[Double]): List[List[Double]] = {
    assert(input.size == nodePerLayer(0))

    var completeOutput: List[List[Double]] = List.empty
    var layerInput: List[Double] = 1 :: input

//    println("inside runNetwork:")
    for {
      i <- 0 to nodePerLayer.size - 1 - 1
    } {
      var o = mutable.MutableList[Double]()
      val layerWeights = weights(i)
//      println("layerInput: " + layerInput)
//      println("layerWeights: " + layerWeights.size + " - " + layerWeights)
      layerWeights
        .map(nodeWeight => {
          val nodeInput =
            (nodeWeight zip layerInput)
              .map{ case (w, i) => w *i }
                .reduce(_ + _)
//            node
//              .map((w: Double) => w * layerInput(nodeNumber))
//              .reduce(_ + _)layerWeights: 10 - List(List(0.4343265314583844, -0.5747041663112318, -0.29144991017153443, -0.09048351016292067, -0.5352978200590228), List(0.2979496771469563, -0.45218067790512184, -0.5656367577735517, -0.8929898643687475, -0.5391326880875755), List(-0.41631368465857044, 0.40252375355607817, 0.36304253589881386, 0.4981618090073505, -0.2465028522837316), List(0.7334822193564972, -0.24773102338468456, -0.38414754520941297, 0.6047136763262293, -0.49593269882891033), List(0.03492133604786685, 0.038352767312893166, 0.3110976511242698, 0.454850083148971, -0.8616417159567369), List(0.14481367987544114, 0.4019708409135683, -0.15688241913861023, 0.7052274678405046, -0.3040317567179347), List(0.67873299534183, 0.33779373397598245, 0.43389623697636037, 0.4572036063180842, 0.07602241099546192), List(-0.611989091374902, 0.6960752713813985, -0.4588374366871768, -0.9293918253983338, 0.2958609305151916), List(-0.20648656991197112, -0.12674292464687986, 0.6493055138493908, -0.9148159998987699, -0.34490718230263595), List(0.32605233782359, 0.8179080932030551, 0.1820809930634646, 0.7776090549469816, 0.27355957243217266))



          val nodeOutput = 1.0 / (1 + Math.pow(Math.E, -1 * nodeInput))
          o += nodeOutput
        })

      // adding 1 for bias
      layerInput = 1 :: o.toList
      completeOutput = completeOutput ++ List(o.toList)
    }

    completeOutput
  }

  def calcError(input: List[Double], expectedOutput: List[Double], calculatedOutput: List[List[Double]]): List[List[Double]] = {
    assert(input.size == nodePerLayer(0))

    var error = List[List[Double]]()
    val unbiasedOutput: List[List[Double]] = List(input) ++ calculatedOutput
    val output = for { i <- 0 to unbiasedOutput.size - 1} yield { (if (i == unbiasedOutput.size - 1) List.empty else List(1.0)) ++ unbiasedOutput(i)}
//    println("inside calc error")
//    println("weights: " + weights)
//    println("expected: " + expectedOutput)
//    println("unbiased output: " + unbiasedOutput)
//    println("output: " + output)

    var modifier: List[Double] = (calculatedOutput(nodePerLayer.size - 2) zip expectedOutput).map { case (a, b) => a - b }
    for {
      i <- (0 to nodePerLayer.size - 2).reverse
    } {
//      println("**** " + i + " ****")
//      println("output(i + 1): " + output(i + 1))
//      println("modifier: " + modifier)
      val layerError =
        (output(i + 1) zip modifier)
          .map{ case (o, m) => o * (1 - o) * m }
//      println("layer error: " + layerError)
      error = error ++ List(if (i != nodePerLayer.size - 2) layerError.tail else layerError) // ignoring bias error
//      println("error: " + error)

      modifier = List.empty
//      println("output(i): " + output(i))
//      println("nodeperlayer(i): " + nodePerLayer(i))
//      println("weights(i): " + weights(i).size + " _ " + weights(i))
      for { j <- 0 to nodePerLayer(i)} {
        //weights from node j to nodes in the next layer
        val weightsOfJ: List[Double] = weights(i).map { case l => l(j) }
        modifier = modifier ++ List((layerError zip weightsOfJ).map { case (e, w) => e * w }.reduce(_ + _))
      }

//      println("modifier: " + modifier)
    }

    error.reverse
  }

  def backPropagate(input: List[Double], calculatedOutput: List[List[Double]], error: List[List[Double]], learningFactor: Double): List[List[List[Double]]] = {
    assert(input.size == nodePerLayer(0))

    val unbiasedOutput: List[List[Double]] = List(input) ++ calculatedOutput
    val output = for { i <- 0 to unbiasedOutput.size - 1} yield { (if (i == unbiasedOutput.size - 1) List.empty else List(1.0)) ++ unbiasedOutput(i)}
//    println("weights: " + weights)
//    println("unbiased output: " + unbiasedOutput)
//    println("output: " + output)
//    println("error: " + error)

    var weightAdjustments = List[List[List[Double]]]()

    for {
      i <- (0 to nodePerLayer.size - 2).reverse
    } {
      val layerWeightAdjustment =
        (List.fill(nodePerLayer(i + 1) + 1)(output(i)) zip error(i))
            .map {case (l, e) =>
              l.map{ case l1 => l1 * e * -1 * learningFactor }
            }
//      println("inside back propagate")
//      println("nodePerLayer(i + 1): " + nodePerLayer(i + 1))
//      println("output(i): " + output(i))
//      println("error(i): " + error(i))
//      println("layerWeightAdjustment: " + layerWeightAdjustment)
//      println("weight(i): " + weights(i))
      weightAdjustments = weightAdjustments ++ List(layerWeightAdjustment)
    }

    weightAdjustments.reverse
  }
}
