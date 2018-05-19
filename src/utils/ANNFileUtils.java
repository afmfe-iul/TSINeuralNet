package utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.net.URL;
import java.net.URLDecoder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ANNFileUtils {
	private static final String TRAINING_OUTPUT_FOLDER = "/trainingData.txt";

	public static DataContainer readCSV(String filePath, int lines, int maxInputValue) throws IOException {
		double[][] inputs = new double[lines][];
		double[][] labels = new double[lines][];
		ClassLoader classloader = Thread.currentThread().getContextClassLoader();
		InputStream in = classloader.getResourceAsStream(filePath); 
		BufferedReader reader = new BufferedReader(new InputStreamReader(in));
        int counter = 0;
		while(reader.ready() && counter < lines){
        	String line = reader.readLine();
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
        DataContainer c = new DataContainer();
        c.add(inputs);
        c.add(labels);
		return c;
	}
	
	public static void writeTrainingResults(List<Integer> results) {
		try {
			PrintWriter writer = new PrintWriter(getJarPath() + TRAINING_OUTPUT_FOLDER);
			for(int i = 0; i < results.size(); i++){
				writer.println(results.get(i));
			}
			writer.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}
	}
	
	public static List<Integer> readTrainingResults(){
		List<Integer> results = new ArrayList<>();
		try {
			ClassLoader classloader = Thread.currentThread().getContextClassLoader();
			InputStream in = classloader.getResourceAsStream(getJarPath() + TRAINING_OUTPUT_FOLDER); 
			BufferedReader reader = new BufferedReader(new InputStreamReader(in));
			while(reader.ready()){
				results.add(Integer.parseInt(reader.readLine()));
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		return results;
	}
	
	public static String getJarPath() throws UnsupportedEncodingException {
	      URL url = ANNFileUtils.class.getProtectionDomain().getCodeSource().getLocation();
	      String jarPath = URLDecoder.decode(url.getFile(), "UTF-8");
	      String parentPath = new File(jarPath).getParentFile().getPath();
	      return parentPath;
	   }
}