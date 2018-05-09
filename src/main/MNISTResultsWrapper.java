package main;

import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;

public class MNISTResultsWrapper {

	private final INDArray testData;
	private final INDArray rawResults;
	private final INDArray resultsAsDigits;
	private final List<Integer> successPositions;
	private final List<Integer> errorPositions;

	public MNISTResultsWrapper(INDArray testData, INDArray rawResults, INDArray resultsAsDigits, List<Integer> successPositions, List<Integer> errorPositions) {
		this.testData = testData;
		this.rawResults = rawResults;
		this.resultsAsDigits = resultsAsDigits;
		this.successPositions = successPositions;
		this.errorPositions = errorPositions;
	}

	public INDArray getTestData() {
		return testData;
	}
	
	public INDArray getRawResults() {
		return rawResults;
	}
	
	public INDArray getResultsAsDigits() {
		return resultsAsDigits;
	}


	public List<Integer> getSuccessPositions() {
		return successPositions;
	}

	public List<Integer> getErrorPositions() {
		return errorPositions;
	}
}