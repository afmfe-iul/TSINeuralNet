package main;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import utils.DataContainer;
import utils.FileUtils;

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

	
	private void train(INDArray trainingData, INDArray trainingLabels, int epochs, double learningRate) {
		INDArray trainingLabelsAsDigits =  Nd4j.argMax(trainingLabels, 1);
		System.out.println("Starting training of " + epochs + " epochs...");
		for(int e = 0; e < epochs; e++){
			System.out.println("Starting epoch " + e);
			INDArray yHat = forwardPropagation(trainingData);
			backPropagation(yHat, trainingLabels);
			updateWeights(learningRate);
			
			INDArray yHatAsDigit = Nd4j.argMax(yHat, 1);

			int errors = 0;
			for(int it = 0; it < yHatAsDigit.rows(); it++){
				if(yHatAsDigit.getDouble(it) != trainingLabelsAsDigits.getDouble(it)){
					errors++;
				}
			}
			System.out.println("Epoch " + e + " total errors: " + errors);
		}
	}

	private INDArray forwardPropagation(INDArray trainingData) {
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


	private void test(INDArray testData, INDArray testLabels) {
		INDArray testLabelsAsDigits =  Nd4j.argMax(testLabels, 1);
		
		INDArray yHat = forwardPropagation(testData);
		INDArray yHatAsDigit = Nd4j.argMax(yHat, 1);
		int errors = 0;
		for(int it = 0; it < yHatAsDigit.rows(); it++){
			int digit = (int) testLabelsAsDigits.getDouble(it);
			int guess = (int) yHatAsDigit.getDouble(it);
			if(digit != guess){
				System.out.print("ERROR!!!\t");
				errors++;
			}else{
				System.out.print("correct!\t");
			}
			System.out.println("(Digit: " + testLabelsAsDigits.getDouble(it) +
					", Guess: " + yHatAsDigit.getDouble(it));
		}
		System.out.println("Total errors in testing data: " + errors);
		System.out.println("Sucess rate in testing data: " + ((double) testLabels.rows() - errors)/testLabels.rows());
	}
	
	public static void main(String[] args) {
		try {
			final int BATCH_SIZE = 1000;
			final int TEST_SIZE = 100;
			final int EPOCHS = 200;
			final double LEARNING_RATE = 0.005;
			ANeuralNetwork network = new ANeuralNetwork(new int[]{784, 150, 150, 10}, BATCH_SIZE);
			
			DataContainer trainingContainer = FileUtils.readCSV("resources/mnist_train.csv", BATCH_SIZE, 255);
			INDArray trainingData = Nd4j.create((double[][])trainingContainer.getContentAt(0));
			INDArray trainingLabels = Nd4j.create((double[][]) trainingContainer.getContentAt(1));
			network.train(trainingData, trainingLabels, EPOCHS, LEARNING_RATE);
			
			DataContainer testContainer = FileUtils.readCSV("resources/mnist_test.csv", TEST_SIZE, 255);
			INDArray testData = Nd4j.create((double[][])testContainer.getContentAt(0));
			INDArray testLabels = Nd4j.create((double[][]) testContainer.getContentAt(1));
			network.test(testData, testLabels);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}