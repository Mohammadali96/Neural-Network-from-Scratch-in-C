#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// A simple NeuralNetwork

double init_weights() { return ((double)rand()) / ((double)RAND_MAX); } // Generating random numbers between 0 and 1 for weights

void shuffle(int *array, size_t n)
{
    if (n > 1)
    {
        size_t i;
        for (i = 0; i < n; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1
#define numTrainingSets 4

double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double dsigmoid(double x) { return x * (1 - x); }

// Function to read CSV and load data into arrays
void read_csv(const char *filename, double inputs[numTrainingSets][numInputs], double outputs[numTrainingSets][numOutputs])
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        printf("Could not open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    char buffer[1024];
    int row = 0;
    while (fgets(buffer, sizeof(buffer), file) && row < numTrainingSets)
    {
        // Parse each line, assuming each row contains input1, input2, output (for XOR example)
        sscanf(buffer, "%lf,%lf,%lf", &inputs[row][0], &inputs[row][1], &outputs[row][0]);
        row++;
    }

    fclose(file);
}

int main(void)
{
    const double lr = 0.1f;
    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputs];

    double hiddenLayerBias[numHiddenNodes];
    double outputLayerBias[numOutputs];

    double hiddenWeights[numInputs][numHiddenNodes];
    double outputWeights[numHiddenNodes][numOutputs];

    double trainingInputs[numTrainingSets][numInputs];
    double trainingOutputs[numTrainingSets][numOutputs];

    // Read CSV file to load training inputs and outputs
    read_csv("New.csv", trainingInputs, trainingOutputs);

    // Initialize weights and biases
    for (int i = 0; i < numInputs; i++)
    {
        for (int j = 0; j < numHiddenNodes; j++)
        {
            hiddenWeights[i][j] = init_weights();
        }
    }

    for (int i = 0; i < numHiddenNodes; i++)
    {
        for (int j = 0; j < numOutputs; j++)
        {
            outputWeights[i][j] = init_weights();
        }
    }

    for (int i = 0; i < numOutputs; i++)
    {
        outputLayerBias[i] = init_weights();
    }

    int trainingSetOrder[] = {0, 1, 2, 3};
    int numberofEpochs = 10000;

    // Train the Network
    for (int epoch = 0; epoch < numberofEpochs; epoch++)
    {
        shuffle(trainingSetOrder, numTrainingSets);

        for (int x = 0; x < numTrainingSets; x++)
        {
            int i = trainingSetOrder[x];

            // Forward pass

            // Hidden Layer Activation
            for (int j = 0; j < numHiddenNodes; j++)
            {
                double activation = hiddenLayerBias[j];

                for (int k = 0; k < numInputs; k++)
                {
                    activation += trainingInputs[i][k] * hiddenWeights[k][j];
                }
                hiddenLayer[j] = sigmoid(activation);
            }

            // Output Layer Activation
            for (int j = 0; j < numOutputs; j++)
            {
                double activation = outputLayerBias[j];

                for (int k = 0; k < numHiddenNodes; k++)
                {
                    activation += hiddenLayer[k] * outputWeights[k][j];
                }
                outputLayer[j] = sigmoid(activation);
            }

            // Back Propagation
            double deltaoutput[numOutputs];
            for (int j = 0; j < numOutputs; j++)
            {
                double error = (trainingOutputs[i][j] - outputLayer[j]);
                deltaoutput[j] = error * dsigmoid(outputLayer[j]);
            }

            double deltaHidden[numHiddenNodes];
            for (int j = 0; j < numOutputs; j++)
            {
                double error = 0.0f;
                for (int k = 0; k < numOutputs; k++)
                {
                    error += deltaoutput[k] * outputWeights[j][k];
                }
                deltaHidden[j] = error * dsigmoid(hiddenLayer[j]);
            }

            // Apply Changes in Output Weights
            for (int j = 0; j < numOutputs; j++)
            {
                outputLayerBias[j] += deltaoutput[j] * lr;
                for (int k = 0; k < numHiddenNodes; k++)
                {
                    outputWeights[k][j] += hiddenLayer[k] * deltaoutput[j] * lr;
                }
            }

            // Apply Changes in hidden Weights
            for (int j = 0; j < numHiddenNodes; j++)
            {
                hiddenLayerBias[j] += deltaHidden[j] * lr;
                for (int k = 0; k < numInputs; k++)
                {
                    hiddenWeights[k][j] += trainingInputs[i][k] * deltaHidden[j] * lr;
                }
            }
        }
    }

    // Print final weights after training
    printf("Final Hidden Weights\n[ ");
    for (int j = 0; j < numHiddenNodes; j++)
    {
        printf("[ ");
        for (int k = 0; k < numInputs; k++)
        {
            printf("%f ", hiddenWeights[k][j]);
        }
        printf("] ");
    }
    printf("]\nFinal Hidden Biases\n[ ");
    for (int j = 0; j < numHiddenNodes; j++)
    {
        printf("%f ", hiddenLayerBias[j]);
    }
    printf("]\nFinal Output Weights\n");
    for (int j = 0; j < numOutputs; j++)
    {
        printf("[ ");
        for (int k = 0; k < numHiddenNodes; k++)
        {
            printf("%f ", outputWeights[k][j]);
        }
        printf("]\n");
    }

    printf("Final Output Biases\n[ ");
    for (int j = 0; j < numOutputs; j++)
    {
        printf("%f ", outputLayerBias[j]);
    }
    printf("]\n");

    return 0;
}
