package main

import scala.collection.mutable

/**
  * Created by masoud on 9/13/16.
  */
class NeuralNetwork(nodePerLayer: List[Int], weights: List[List[List[Double]]]) {

  def runNetwork(input: List[Double]): List[List[Double]] = {
    assert(input.size == nodePerLayer(0))

    var completeOutput: List[List[Double]] = List.empty
    var layerInput: List[Double] = 1 :: input

    for {
      i <- 0 to nodePerLayer.size - 1 - 1
    } {
      var o = mutable.MutableList[Double]()
      val layerWeights = weights(i)
      layerWeights
        .map(nodeWeight => {
          val nodeInput =
            (nodeWeight zip layerInput)
              .map{ case (w, i) => w *i }
                .reduce(_ + _)

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
    val output =
      for { i <- 0 to unbiasedOutput.size - 1}
        yield { (if (i == unbiasedOutput.size - 1) List.empty else List(1.0)) ++ unbiasedOutput(i)}

    var modifier: List[Double] =
      (calculatedOutput(nodePerLayer.size - 2) zip expectedOutput)
        .map { case (a, b) => a - b }

    for {
      i <- (0 to nodePerLayer.size - 2).reverse
    } {
      val layerError =
        (output(i + 1) zip modifier)
          .map{ case (o, m) => o * (1 - o) * m }
      error = error ++ List(if (i != nodePerLayer.size - 2) layerError.tail else layerError) // ignoring bias error

      modifier = List.empty
      for { j <- 0 to nodePerLayer(i)} {
        val weightsOfJ: List[Double] = weights(i).map { case l => l(j) }
        modifier = modifier ++ List((layerError zip weightsOfJ).map { case (e, w) => e * w }.reduce(_ + _))
      }
    }

    error.reverse
  }

  def backPropagate(input: List[Double], calculatedOutput: List[List[Double]], error: List[List[Double]], learningFactor: Double): List[List[List[Double]]] = {
    assert(input.size == nodePerLayer(0))

    val unbiasedOutput: List[List[Double]] = List(input) ++ calculatedOutput
    val output = for { i <- 0 to unbiasedOutput.size - 1} yield { (if (i == unbiasedOutput.size - 1) List.empty else List(1.0)) ++ unbiasedOutput(i)}
    var weightAdjustments = List[List[List[Double]]]()

    for {
      i <- (0 to nodePerLayer.size - 2).reverse
    } {
      val layerWeightAdjustment =
        (List.fill(nodePerLayer(i + 1) + 1)(output(i)) zip error(i))
          .map {case (l, e) =>
            l.map{ case l1 => l1 * e * -1 * learningFactor }
          }
      weightAdjustments = weightAdjustments ++ List(layerWeightAdjustment)
    }

    weightAdjustments.reverse
  }
}
