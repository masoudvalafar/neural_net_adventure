package main

import scala.collection.mutable

/**
  * Neural Network training
  * Back-propagation algorithm steps:
  * 1- run the network and calculate the output for each layer
  * 2- calculate the error:
  *   - for node k in the output layer: delta_k = O_k (1 - O_k) (O_k - T_k)
  *   where O_k is the output of node k and T_k is the expected output
  *   - for each node j in other layers: delta_j = O_j (1 _ O_j) SIGMA (delata_k * Wjk)
  *   where Wjk is the weight from node j to node k in the next layer. Sigma is on every node k in the next layer
  * 3- weight update calculation: weight_update(i)(j)(k) = etta * delta_j * O_k
  * where etta is etta is the learning factor
  *
  * @param nodePerLayer a list of number of nodes for each layer. Starts with input and input is layer 0
  * @param weights weights used in the training.
  *                The first list contains the weights of each layer  weights(i) are weights from layer i to layer i + 1
  *                The second list contains the weights per nodes. weights(i)(j) is a list of weights fron all the nodes
  *                in layer i to node j of layer i + 1
  *                The third nested list is the list of weights: weights(i)(j)(k) is the weight from node k in layer i
  *                to node j in layer i + 1
  */
class NeuralNetwork(nodePerLayer: List[Int], weights: List[List[List[Double]]]) {

  /**
    * step 1 of the algorithm.
    * Feed the input into the network and calculate the output of the other layers. Adds a bias of 1 to each layer.
    *
    * @param input inputs to the first layer
    * @return for each layer, list of output of that layer
    */
  def calculateOutput(input: List[Double]): List[List[Double]] = {
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

  /**
    * Step 2 of the algorithm.
    *
    * @param input input to the network
    * @param expectedOutput expected output of the last layer
    * @param calculatedOutput output calculated by step 1
    * @return delta for each layer
    */
  def calcError(input: List[Double], expectedOutput: List[Double], calculatedOutput: List[List[Double]]): List[List[Double]] = {
    assert(input.size == nodePerLayer(0))

    var error = List[List[Double]]()
    val unbiasedOutput: List[List[Double]] = List(input) ++ calculatedOutput
    val output =
      for { i <- 0 to unbiasedOutput.size - 1}
        yield { (if (i == unbiasedOutput.size - 1) List.empty else List(1.0)) ++ unbiasedOutput(i)}

    // the part of the delta that depend on the expected output or nodes in the next layer
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

  /**
    * Step 3 of the algorithm.
    *
    * @param input input to the network
    * @param calculatedOutput output of step 1
    * @param error output of step 2
    * @param learningFactor coefficient for dampening update
    * @return updates for each weight given the input and expected output
    */
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
