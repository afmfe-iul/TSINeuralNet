package utils;

import java.io.File;
import java.util.Arrays;
import java.util.Scanner;
import javax.swing.SwingWorker;

public class FileReaderWorker extends SwingWorker<Void, Void>{
	final String filePath;
	final int lines;
	final int maxInputValue;
	final DataContainer dataContainer;
	
	public FileReaderWorker(String filePath, int lines, int maxInputValue) {
		this.filePath = filePath;
		this.lines = lines;
		this.maxInputValue = maxInputValue;
		this.dataContainer = new DataContainer();
	}
	
	public DataContainer getContainer() {
		return dataContainer;
	}
	
	@Override
	protected Void doInBackground() throws Exception {
		double[][] inputs = new double[lines][];
		double[][] labels = new double[lines][];
		Scanner scanner = new Scanner(new File(filePath));
        int counter = 0;
        setProgress(counter);
		while(scanner.hasNext() && counter < lines){
        	String line = scanner.next();
        	double[] labelRow = new double[10];
        	labelRow[Integer.parseInt("" + line.charAt(0))] = 1;
        	labels[counter] = labelRow;
        	double[] input = Arrays.stream(line.substring(2).split(",")).mapToDouble(Double::parseDouble).toArray();
        	for(int i = 0; i < input.length; i++){
        		input[i] /= maxInputValue;
        	}
        	inputs[counter] = input;
        	counter++;
        	setProgress(counter/lines);
		}
        scanner.close();
        dataContainer.add(inputs);
        dataContainer.add(labels);
        return null;
	}
}
