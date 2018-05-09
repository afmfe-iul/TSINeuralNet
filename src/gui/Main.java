package gui;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.FileNotFoundException;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import main.ANeuralNetwork;
import main.MNISTResultsWrapper;
import utils.DataContainer;
import utils.FileUtils;

public class Main {
	private static final int FRAME_W = 320;
	private static final int FRAME_H = 280;
	private static final int ICON_W = 28;
	private static final int ICON_H = 28;
	
	JFrame frame;
	JLabel guessLabel;
	JLabel imgLabel;
	JLabel[] decisionLabels;
	int selectedError;
	int selectedSuccess;
	boolean showingSuccesses;
	
	ANeuralNetwork network;
	MNISTResultsWrapper testResults;
	
	public Main() {
		selectedError = 0;
		selectedSuccess = 0;
		showingSuccesses = true;
		
		promptForANNSpecs();
		initUI();
		changeSelectedExample(0);

		frame.setSize(FRAME_W, FRAME_H);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setResizable(false);
		frame.setLocationRelativeTo(null);
		frame.setVisible(true);
	}
	
	public void initUI() {
		frame = new JFrame();
		frame.setLayout(new BorderLayout());

		// North Panel
		JPanel northPanel = new JPanel(new GridLayout(2,1));
		northPanel.add(new JLabel("Sucess percentage: " + String.format("%.2f", ((double) testResults.getTestData().rows() - 
				testResults.getErrorPositions().size()) / testResults.getTestData().rows() * 100)));
		northPanel.add(new JLabel("Total errors: " + testResults.getErrorPositions().size()));
		frame.getContentPane().add(northPanel, BorderLayout.NORTH);
		
		// Center Panel
		imgLabel = new JLabel();
		imgLabel.setHorizontalAlignment(JLabel.CENTER);
		guessLabel = new JLabel();
		guessLabel.setHorizontalAlignment(JLabel.CENTER);
		JPanel centerPanel = new JPanel(new GridLayout(2, 1));
		centerPanel.add(imgLabel);
		centerPanel.add(guessLabel);
		frame.getContentPane().add(centerPanel, BorderLayout.CENTER);
		
		
		// South Panel
		JButton btnPrevious = new JButton("<");
		btnPrevious.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				changeSelectedExample(-1);
			}
		});
		
		JButton btnNext = new JButton(">");
		btnNext.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				changeSelectedExample(1);
			}
		});
		
		JButton btnSuccesses = new JButton("Show successes");
		btnSuccesses.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				showingSuccesses = true;
				changeSelectedExample(0);
			}
		});
		if(testResults.getSuccessPositions().size() == 0) {
			btnSuccesses.setEnabled(false);
			showingSuccesses = false;
		}
		
		JButton btnFailures = new JButton("Show failures");
		btnFailures.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				showingSuccesses = false;
				changeSelectedExample(0);
			}
		});
		if(testResults.getErrorPositions().size() == 0) {
			btnFailures.setEnabled(false);
			showingSuccesses = true;
		}
		
		JPanel buttonPanel = new JPanel(new GridLayout(2,2));
		buttonPanel.add(btnPrevious);
		buttonPanel.add(btnNext);
		buttonPanel.add(btnSuccesses);
		buttonPanel.add(btnFailures);
		frame.getContentPane().add(buttonPanel, BorderLayout.SOUTH);
		
		
		// East Panel
		JPanel decisionPanel = new JPanel(new GridLayout(11, 2));
		decisionPanel.add(new JLabel("Digit |"));
		decisionPanel.add(new JLabel(" Certainty"));
		decisionLabels = new JLabel[10];
		for(int i = 0; i < 10; i++) {
			decisionPanel.add(new JLabel(i + ": "));
			decisionLabels[i] = new JLabel();
			decisionPanel.add(decisionLabels[i]);
		}
		frame.getContentPane().add(decisionPanel, BorderLayout.EAST);
	}
	

	private void promptForANNSpecs() {
		 JDialog dialog = new JDialog(frame);
        dialog.setTitle("Artificial Neural Network Parameters");
        dialog.setSize(new Dimension(800, 200));
        dialog.setLayout(new GridLayout(7,2));
        
        dialog.add(new JLabel("Layers configuration (format: { #neuronsL1, #neuronsL2, etc}): "));
        JTextField layerConfiguration = new JTextField("{784, 100, 100, 10}");
        dialog.add(layerConfiguration);
        
        dialog.add(new JLabel("Number of training examples from MNIST dataset (max = 55000): "));
        JTextField trainingExamples = new JTextField("30000");
        dialog.add(trainingExamples);
        
        dialog.add(new JLabel("Number of test examples from MNIST dataset (max = 10000): "));
        JTextField testExamples = new JTextField("5000");
        dialog.add(testExamples);
        
        dialog.add(new JLabel("Number of Epochs (train cycles): "));
        JTextField epochs = new JTextField("750");
        dialog.add(epochs);
        
        dialog.add(new JLabel("Mini batch size (should be smaller than # training examples): "));
        JTextField miniBatchSize = new JTextField("32");
        dialog.add(miniBatchSize);
        
        dialog.add(new JLabel("Learning rate (from 0 to 1): "));
        JTextField learningRate = new JTextField("0.06");
        dialog.add(learningRate);

        JButton btnStartTraining = new JButton("Start Training");
        btnStartTraining.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				String[] s = layerConfiguration.getText().replaceAll("\\{", "").replaceAll("\\}", "").split(",");
				int[] layerConfigurationArray = new int[s.length];
				for(int i = 0; i < layerConfigurationArray.length; i++) {
					layerConfigurationArray[i] = Integer.parseInt(s[i].trim());
				}
				
				try {
					runTraining(dialog, layerConfigurationArray, Integer.parseInt(trainingExamples.getText()), Integer.parseInt(testExamples.getText()),
							Integer.parseInt(epochs.getText()), Integer.parseInt(miniBatchSize.getText()), 
							Double.parseDouble(learningRate.getText()));
					dialog.dispose();
				} catch (NumberFormatException e1) {
					JOptionPane.showMessageDialog(dialog, "All fields should be filled correctly",
							"Error!", JOptionPane.ERROR_MESSAGE);
				} catch (FileNotFoundException e1) {
					JOptionPane.showMessageDialog(dialog, "Couldn't find the mnist_train.csv or mnist_test.csv inside the /resources folder",
							"Error!", JOptionPane.ERROR_MESSAGE);
				}
			}
		});
        dialog.add(new JLabel()); // filler label
        dialog.add(btnStartTraining);
        
        dialog.setLocationRelativeTo(frame);
        dialog.setModal(true);
        dialog.setResizable(false);
        dialog.setVisible(true);
	}
	
	private void runTraining(JDialog dialog, int[] layerConfigurationArray, int trainingExamples, int testExamples,
			int epochs, int miniBatchSize, double learningRate) throws FileNotFoundException {
		network = new ANeuralNetwork(layerConfigurationArray, miniBatchSize);

		// TODO file reading progress bar
		DataContainer trainingContainer = FileUtils.readCSV("resources/mnist_train.csv", trainingExamples, 255);
		INDArray trainingData = Nd4j.create((double[][]) trainingContainer.getContentAt(0));
		INDArray trainingLabels = Nd4j.create((double[][]) trainingContainer.getContentAt(1));

		// TODO training progress bar
		network.train(trainingData, trainingLabels, epochs, trainingExamples, miniBatchSize, learningRate);
		
		// TODO testing progress bar
		DataContainer testContainer = FileUtils.readCSV("resources/mnist_test.csv", testExamples, 255);
		INDArray testData = Nd4j.create((double[][])testContainer.getContentAt(0));
		INDArray testLabels = Nd4j.create((double[][]) testContainer.getContentAt(1));
		testResults = network.test(testData, testLabels);
	}
	
	private void changeSelectedExample(int direction) {
		int selectedExample;
		if(showingSuccesses) {
			selectedSuccess = selectedSuccess + direction < 0 ?
					testResults.getSuccessPositions().size() - 1 : (selectedSuccess + direction) % testResults.getSuccessPositions().size();
			selectedExample = testResults.getSuccessPositions().get(selectedSuccess);
		}else {
			selectedError = selectedError + direction < 0 ?
					testResults.getErrorPositions().size() - 1 : (selectedError + direction) % testResults.getErrorPositions().size();
			selectedExample = testResults.getErrorPositions().get(selectedError);
		}
		
		int[] result = testResults.getTestData().getRow(selectedExample).mul(255).data().asInt();
		for(int i = 0;i < result.length; i++) {
			result[i] = 255 - result[i];
		}
		
		BufferedImage outputImage = new BufferedImage(ICON_W, ICON_H, BufferedImage.TYPE_BYTE_GRAY);
		WritableRaster raster = outputImage.getRaster();
		raster.setSamples(0, 0, ICON_W, ICON_H, 0, result);
		ImageIcon imgIcon = new ImageIcon(outputImage.getScaledInstance(ICON_W*2, ICON_H*2, Image.SCALE_SMOOTH));
		imgLabel.setIcon(imgIcon);
		guessLabel.setText("Neural Network guess: " + testResults.getResultsAsDigits().getInt(selectedExample));
	
		INDArray yHat = network.forwardPropagation(testResults.getTestData().getRow(selectedExample));
		for(int i = 0; i < 10; i++) {
			decisionLabels[i].setText(String.format("%.2f", yHat.getColumn(i).getDouble(0) * 100) + " %");
		}
	}
	

	public static void main(String[] args) {
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				@SuppressWarnings("unused")
				Main m = new Main();
			}
		});
	}
}