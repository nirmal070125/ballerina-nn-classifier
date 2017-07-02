import ballerina.lang.system;
import wso2.ballerina.math;
import ballerina.lang.strings;

float globalError = 0.0;
int inputCount = 36;
int hiddenCount = 72;
int outputCount = 1;
int neuronCount = inputCount + hiddenCount + outputCount;
int weightCount = (inputCount * hiddenCount) + (hiddenCount * outputCount);
float learnRate = 0.1;
float[] fire = [];
float[] matrix = [];
float[] error = [];
float[] accMatrixDelta = [];
float[] thresholds = [];
float[] matrixDelta = [];
float[] accThresholdDelta = [];
float[] thresholdDelta = [];
float[] errorDelta = [];
float momentum = 0.9;


function main (string[] args) {
    init(1);
    reset(1);
    string[] files = math:allfilenames(args[0]);
    int i = 0;
    int j = 0;
    int cats = 0;
    int dogs = 0;
    float[][] inputs = [];
    float[][] outputs = [];
    int cutoff_train = 1000;
    system:println("Training started...");
    while (i < files.length) {
        if (cats > cutoff_train && dogs > cutoff_train) {
            break;
        }
        if (strings:contains(files[i], "cat")) {
            if (cats > cutoff_train) {
                i = i + 1;
                continue;
            }
            outputs[j] = [0.0];
            cats = cats + 1;
        } else {

            if (dogs > cutoff_train) {
                i = i + 1;
                continue;
            }
            // dog
            outputs[j] = [1.0];
            dogs = dogs + 1;
        }
        //system:println("FIle: "+files[i]);
        float[] pixels = math:image2vec(files[i], "6", "6");
        inputs[j] = pixels;
        i = i + 1;
        j = j + 1;
    }

    neural_net(inputs, outputs, 100);

    system:println("Predictions.....");

    files = math:allfilenames(args[1]);
    i=0;
    int total =0;
    system:println("PS: Cat: 0.0 ----- Dog: 1.0 ");
    while (i < files.length && total < 5) {
        if (randomNumber() > 0.6) {
            float[] pixels = math:image2vec(files[i], "6", "6");
            float[] out = computeOutputs(inputs[i]);
            system:println("File: " + files[i] + " --- Prediction: " + out[0]);
            total = total +1;
        }
        i = i+1;

    }

    system:println("***** Done *****");
}

function init (int dummy) {
    matrix[weightCount - 1] = 0.0;
}

function randomNumber () (float) {
    //returns a random number between 0 & 1
    return math:random();
}

function reset (int dummy) {
    //Reset the weight matrix and the thresholds.
    int i = 0;
    while (i < neuronCount) {
        thresholds[i] = 0.5 - randomNumber();
        thresholdDelta[i] = 0;
        accThresholdDelta[i] = 0;
        i = i + 1;
    }
    i = 0;
    while (i < matrix.length) {
        matrix[i] = 0.5 - randomNumber();
        matrixDelta[i] = 0;
        accMatrixDelta[i] = 0;
        i = i + 1;
    }
}

function learn () {
    // Modify the weight matrix and thresholds based on the last call to calcError.
    int i = 0;
    // process the matrix
    while (i < matrix.length) {
        matrixDelta[i] = (learnRate * accMatrixDelta[i]) + (momentum * matrixDelta[i]);
        matrix[i] = matrix[i] + matrixDelta[i];
        accMatrixDelta[i] = 0;
        i = i + 1;
    }

    i = inputCount;
    // process the thresholds
    while (i < neuronCount) {
        thresholdDelta[i] = learnRate * accThresholdDelta[i] + (momentum * thresholdDelta[i]);
        thresholds[i] = thresholds[i] + thresholdDelta[i];
        accThresholdDelta[i] = 0;
        i = i + 1;
    }
}

function calcError (float[] ideal) {
    //Calculate the error for the recognition just done.
    //@param ideal What the output neurons should have yielded.
    int i;
    int j;
    int hiddenIndex = inputCount;
    int outputIndex = inputCount + hiddenCount;

    // clear hidden layer errors
    i = inputCount;
    while (i < neuronCount) {
        error[i] = 0;
        i = i + 1;
    }

    // layer errors and deltas for output layer
    i = outputIndex;
    while (i < neuronCount) {
        error[i] = ideal[i - outputIndex] - fire[i];
        globalError = globalError + error[i] * error[i];
        errorDelta[i] = error[i] * fire[i] * (1 - fire[i]);
        i = i + 1;
    }

    // hidden layer errors
    int winx = inputCount * hiddenCount;
    i = outputIndex;
    while (i < neuronCount) {
        j = hiddenIndex;
        while (j < outputIndex) {
            accMatrixDelta[winx] = accMatrixDelta[winx] + errorDelta[i] * fire[j];
            error[j] = error[j] + matrix[winx] * errorDelta[i];
            winx = winx + 1;
            j = j + 1;
        }
        accThresholdDelta[i] = accThresholdDelta[i] + errorDelta[i];
        i = i + 1;
    }

    // hidden layer deltas
    i = hiddenIndex;
    while (i < outputIndex) {
        errorDelta[i] = error[i] * fire[i] * (1 - fire[i]);
        i = i + 1;
    }

    // input layer errors
    winx = 0; // offset into weight array
    i = hiddenIndex;
    while (i < outputIndex) {
        j = 0;
        while (j < hiddenIndex) {
            accMatrixDelta[winx] = accMatrixDelta[winx] + errorDelta[i] * fire[j];
            error[j] = error[j] + matrix[winx] * errorDelta[i];
            winx = winx + 1;
            j = j + 1;
        }
        accThresholdDelta[i] = accThresholdDelta[i] + errorDelta[i];
        i = i + 1;
    }
}

function computeOutputs (float[] input) (float[]) {
    //Compute the output for a given input to the neural network.
    // @param input The input provide to the neural network.
    // @return The results from the output neurons.
    int i;
    int j;
    int hiddenIndex = inputCount;
    int outIndex = inputCount + hiddenCount;

    i = 0;
    while (i < inputCount) {
        fire[i] = input[i];
        i = i + 1;
    }

    // first layer
    int inx = 0;
    i = hiddenIndex;
    while (i < outIndex) {
        float sum = thresholds[i];
        j = 0;
        while (j < inputCount) {
            //system:println("input: "+ matrix[inx] + " ** j: "+j);
            sum = sum + fire[j] * matrix[inx];
            inx = inx + 1;
            j = j + 1;
        }
        fire[i] = threshold(sum);
        i = i + 1;
    }

    // hidden layer
    float[] result = [];
    i = outIndex;
    while (i < neuronCount) {
        float sum = thresholds[i];
        j = hiddenIndex;
        while (j < outIndex) {
            sum = sum + fire[j] * matrix[inx];
            inx = inx + 1;
            j = j + 1;
        }
        fire[i] = threshold(sum);
        result[i - outIndex] = fire[i];
        i = i + 1;
    }

    return result;
}

function threshold (float sum) (float) {
    //The threshold method. (RELU = max(x,0))
    // @param sum The activation from the neuron.
    //@return The activation applied to the threshold method.
    // if (sum > 0) {
    //     return sum;
    // } else {
    //     return 0;
    // }

    // Sigmoid
    return 1.0 / (1 + math:exp(-1.0 * sum));
}

function sqrt (float x) (float) {
    // using Newton's method :-)
    if (x == 0) {
        return 0;
    }

    float last = 0.0;
    float res = 1.0;
    while (res != last) {
        last = res;
        res = (res + x / res) / 2;
    }
    return res;
}

function getError (int len) (float) {
    //Returns the root mean square error for a complete training set.
    //@param len The length of a complete training set.
    //@return The current error for the neural network.
    float err = math:pow(globalError / (len * outputCount), 0.5);
    globalError = 0; // clear the accumulator
    return err;
}

function neural_net (float[][] inputs, float[][] outputs, int epochs) {

    int i = 0;
    int j = 0;
    while (i < epochs) {
        j = 0;
        while (j < inputs.length) {
            computeOutputs(inputs[j]);
            calcError(outputs[j]);
            learn();
            j = j + 1;
        }
        system:println("Trial #" + i + ",Error:" + getError(inputs.length));
        i = i + 1;
    }

    //i = 0;
    //j = 0;
    //system:println("Inputs len: " + inputs.length);
    //
    //while (i < inputs.length) {
    //    j = 0;
    //    while (j < inputs[0].length) {
    //        system:print(inputs[i][j] + ",");
    //        j = j + 1;
    //    }
    //    system:print(":");
    //    float[] out = computeOutputs(inputs[i]);
    //    system:println("=" + out[0]);
    //    i = i + 1;
    //}
}



