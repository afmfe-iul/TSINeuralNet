## Requisites:

1. **Download the MNIST train and test set** from https://pjreddie.com/projects/mnist-in-csv/
2. Make sure the downloaded files are named **"mnist\_test.csv"** and **"mnist\_train.csv"** and put them on the **resources folder** inside the project 

**Warning: to run all the 60.000 training examples, and depending on the Neural Network configuration, you might have to increase your JVM heap size. Add the following parameters to your run configurations:**

    -Xms<size>        // sets the initial Java heap size
    -Xmx<size>        // sets the maximum Java heap size

**Also if you want to run the app from a jar file you must have a 64bit JVM. The command to launch the app with 1500MB of heap space would be:**
    
    	java -jar -d64 -Xmx1500m <jar path>