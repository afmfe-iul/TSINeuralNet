package main;

import java.util.ArrayList;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class ANeuralNetwork {
	List<Layer> layers;

	public ANeuralNetwork(int[] layerConfig, int batchSize) {
		layers = new ArrayList<>();
		// Adds input layer
		layers.add(new Layer(new int[]{layerConfig[0], layerConfig[1]}, batchSize, true, false));
		
		// Adds hidden layers
		for(int i = 1; i < layerConfig.length - 1; i++){
			layers.add(new Layer(new int[]{layerConfig[i], layerConfig[i+1]}, batchSize, false, false));
		}
		
		// Adds output layer
		layers.add(new Layer(new int[]{layerConfig[layerConfig.length - 1]}, batchSize, false, true));
	}

	
	public void train(INDArray trainingData, INDArray trainingLabels, int epochs, int batchSize,
			int miniBatchSize, double learningRate) {
		INDArray trainingLabelsAsDigits =  Nd4j.argMax(trainingLabels, 1);
		System.out.println("Starting training of " + epochs + " epochs..."); // TODO save the training progress somewhere to show on UI
		int miniBatchStart = Integer.MAX_VALUE;
		for(int e = 0; e < epochs; e++){
			while(miniBatchStart >= batchSize - miniBatchSize) {
				miniBatchStart = (int) (Math.random() * batchSize);
			}
			System.out.println("Starting epoch " + e);
			INDArray yHat = forwardPropagation(trainingData.get(NDArrayIndex.interval(miniBatchStart, miniBatchStart + miniBatchSize), NDArrayIndex.all()));
			backPropagation(yHat, trainingLabels.get(NDArrayIndex.interval(miniBatchStart, miniBatchStart + miniBatchSize), NDArrayIndex.all()));
			updateWeights(learningRate);
			
			INDArray yHatAsDigit = Nd4j.argMax(yHat, 1);

			int errors = 0;
			for(int it = 0; it < yHatAsDigit.rows(); it++){
				if(yHatAsDigit.getDouble(it) != trainingLabelsAsDigits.getDouble(miniBatchStart + it)){
					errors++;
				}
			}
			System.out.println("Epoch " + e + " total errors: " + errors);
			miniBatchStart = Integer.MAX_VALUE;
		}
	}

	public INDArray forwardPropagation(INDArray trainingData) {
		layers.get(0).Z = trainingData;
		
		for(int i = 0; i < layers.size() - 1; i++){
			Layer nextLayer = layers.get(i + 1);
			nextLayer.S = layers.get(i).forwardPropagation();
		}
		return layers.get(layers.size() - 1).forwardPropagation();
	}
	
	
	private void backPropagation(INDArray yHat, INDArray trainingLabels) {
		layers.get(layers.size() - 1).D = yHat.sub(trainingLabels).transpose();

		for(int i = layers.size() - 2; i > 0; i--){
			Layer currLayer = layers.get(i);
			currLayer.D = currLayer.W.mmul(layers.get(i + 1).D).mul(currLayer.F);
		}
	}
	
	private void updateWeights(double learningRate) {
		for(int i = 0; i < layers.size() - 1; i++){
			Layer currLayer = layers.get(i);
			INDArray weightsGradient = layers.get(i + 1).D.mmul(currLayer.Z).transpose()
					.mul(-learningRate);
			currLayer.W.addi(weightsGradient);
		}
	}


	public MNISTResultsWrapper test(INDArray testData, INDArray testLabels) {
		INDArray testLabelsAsDigits =  Nd4j.argMax(testLabels, 1);
		
		INDArray yHat = forwardPropagation(testData);
		INDArray yHatAsDigit = Nd4j.argMax(yHat, 1);
		
		List<Integer> successPositions = new ArrayList<>();
		List<Integer> errorPositions = new ArrayList<>();
		for(int it = 0; it < yHatAsDigit.rows(); it++){
			int digit = (int) testLabelsAsDigits.getDouble(it);
			int guess = (int) yHatAsDigit.getDouble(it);
			if(digit == guess){
				successPositions.add(it);
			}else {
				errorPositions.add(it);
			}
		}
		return new MNISTResultsWrapper(testData, yHat, yHatAsDigit, successPositions, errorPositions);
	}
}