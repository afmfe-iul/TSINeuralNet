package utils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class ANNFileUtils {
	public static final String TRAINING_OUTPUT_FOLDER = "resources/trainingData.txt";

	public static DataContainer readCSV(String filePath, int lines, int maxInputValue) throws FileNotFoundException {
		double[][] inputs = new double[lines][];
		double[][] labels = new double[lines][];
		Scanner scanner = new Scanner(new File(filePath));
        int counter = 0;
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
		}
        scanner.close();
        DataContainer c = new DataContainer();
        c.add(inputs);
        c.add(labels);
		return c;
	}
	
	public static void writeTrainingResults(List<Integer> results) {
		try {
			PrintWriter writer = new PrintWriter(TRAINING_OUTPUT_FOLDER);
			for(int i = 0; i < results.size(); i++){
				writer.println(results.get(i));
			}
			writer.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	public static List<Integer> readTrainingResults(){
		List<Integer> results = new ArrayList<>();
		try {
			Scanner scanner = new Scanner(new File(TRAINING_OUTPUT_FOLDER));
			while(scanner.hasNext()){
				results.add(Integer.parseInt(scanner.next()));
			}
			scanner.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		return results;
	}
}