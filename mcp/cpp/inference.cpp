#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>

using namespace std;

vector<vector<float>> loadMatrix(string filename, int rows, int cols)
{
    ifstream file(filename);

    vector<vector<float>> matrix(rows, vector<float>(cols));

    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            file >> matrix[i][j];
        }
    }

    file.close();

    return matrix;
}

vector<float> loadVector(string filename, int size)
{
    ifstream file(filename);

    vector<float> vec(size);

    for(int i=0;i<size;i++)
        file >> vec[i];

    file.close();

    return vec;
}

float sigmoid(float x)
{
    return 1.0f/(1.0f+exp(-x));
}


vector<float> softmax(vector<float> x)
{
    float maxVal = *max_element(x.begin(), x.end());

    float sum = 0.0f;

    for(int i=0;i<x.size();i++)
    {
        x[i] = exp(x[i]-maxVal);
        sum += x[i];
    }

    for(int i=0;i<x.size();i++)
        x[i] /= sum;

    return x;
}

vector<float> dense(
        vector<vector<float>> &W,
        vector<float> &b,
        vector<float> &x)
{
    int output_size = W.size();
    int input_size = W[0].size();

    vector<float> z(output_size);

    for(int i=0;i<output_size;i++)
    {
        z[i] = b[i];

        for(int j=0;j<input_size;j++)
        {
            z[i] += W[i][j] * x[j];
        }
    }

    return z;
}

vector<float> applySigmoid(vector<float> z)
{
    for(int i=0;i<z.size();i++)
        z[i]=sigmoid(z[i]);

    return z;
}

int argmax(vector<float> x)
{
    return max_element(x.begin(), x.end()) - x.begin();
}

vector<float> predict(
    vector<float> x,
    vector<vector<float>> &W1,
    vector<float> &b1,
    vector<vector<float>> &W2,
    vector<float> &b2)
{
    // Layer 1
    vector<float> hidden = dense(W1, b1, x);

    // Activation
    hidden = applySigmoid(hidden);

    // Layer 2
    vector<float> logits = dense(W2, b2, hidden);

    // Output probabilities
    return softmax(logits);
}

int main()
{
    // Load trained weights
    auto W1 = loadMatrix("W1.txt",8,4);
    auto b1 = loadVector("b1.txt",8);

    auto W2 = loadMatrix("W2.txt",3,8);
    auto b2 = loadVector("b2.txt",3);

    // One Iris sample
    vector<float> x =
    {
        5.1,
        3.5,
        1.4,
        0.2
    };

    const int NUM_RUNS = 100000;

    vector<float> probs;

    auto start = chrono::high_resolution_clock::now();

    for(int i=0;i<NUM_RUNS;i++)
    {
        probs = predict(x,W1,b1,W2,b2);
    }

    auto stop = chrono::high_resolution_clock::now();

    auto duration =
        chrono::duration_cast<chrono::microseconds>(stop-start);

    cout << "----------------------------------\n";
    cout << "Inference Benchmark\n";
    cout << "----------------------------------\n";
    cout << "Total runs           : " << NUM_RUNS << endl;
    cout << "Total time           : "
         << duration.count()
         << " us\n";

    cout << "Average inference    : "
         << (double)duration.count()/NUM_RUNS
         << " us\n";

#ifdef _OPENMP
    cout << "OpenMP threads       : "
         << omp_get_max_threads()
         << endl;
#endif

    cout << "\nProbabilities\n";

    for(float p : probs)
        cout << p << endl;

    cout << "\nPrediction = "
         << argmax(probs)
         << endl;

    return 0;
}