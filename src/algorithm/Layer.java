package algorithm;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Layer {
	// Seed for random weight generation
	public static final long SEED = 123;
	INDArray Z; 		// matrix with the layer output values
	INDArray W; 		// weight matrix to the next layer
	INDArray S; 		// matrix with the inputs to this layer
	INDArray D; 		// matrix with the deltas for this layer
	INDArray F; 		// matrix with the derivatives of the activation function transposed
	INDArray BIAS;
	boolean isInputLayer;
	boolean isOutputLayer;

	public Layer(int[] weightSize, int batchSize, boolean isInputLayer, boolean isOutputLayer) {
		// The input layer does not have a S and D matrix, and it's Z matrix is
		// the training/testing inputs and is only assigned during training/testing
		if(!isInputLayer){
			Z = Nd4j.zeros(batchSize, weightSize[0]);
			S = Nd4j.zeros(batchSize, weightSize[0]);
			D = Nd4j.zeros(batchSize, weightSize[0]);
		}
		
		// The output layer does not have a W matrix
		if(!isOutputLayer){
			W = Nd4j.randn(weightSize[0] + 1, weightSize[1], SEED);
		}
		
		// Only the hidden layers have and F matrix
		if(!isInputLayer && !isOutputLayer){
			F = Nd4j.zeros(weightSize[0], batchSize);
		}
		
		this.BIAS = Nd4j.ones(batchSize, 1);
		this.isInputLayer = isInputLayer;
		this.isOutputLayer = isOutputLayer;
	}
	
	public INDArray forwardPropagation(){
		// for the input layer returns Z.W right away (to the input layer Z it's never
		// applied the activation function)
		if(isInputLayer){
			return Z.mmul(W);
		}
		
		// for the output layer computes Z with softMax and returns it
		if(isOutputLayer){
			Z = Transforms.softmax(S);
			return Z;
		}
		
		// for the hidden layers computes F with sigmoid activation and returns Z.W
		Z = Transforms.sigmoid(S, true);
		Z = Nd4j.hstack(Z, BIAS);
		F = Transforms.sigmoidDerivative(S, true).transpose();
		return Z.mmul(W);
	}
}