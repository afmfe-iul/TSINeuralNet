package utils;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Scanner;

public class FileUtils {

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
        	for(int i = 1; i < input.length; i++){
        		input[i] /= 255;
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
}