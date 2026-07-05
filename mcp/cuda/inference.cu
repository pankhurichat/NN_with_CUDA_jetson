#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

#include <cuda_runtime.h>

using namespace std;

#define INPUT_SIZE 4
#define HIDDEN_SIZE 8
#define OUTPUT_SIZE 3

vector<float> loadMatrix(string filename,int rows,int cols)
{
    ifstream file(filename);

    vector<float> matrix(rows*cols);

    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            file>>matrix[i*cols+j];
        }
    }

    return matrix;
}

vector<float> loadVector(string filename,int size)
{
    ifstream file(filename);

    vector<float> v(size);

    for(int i=0;i<size;i++)
        file>>v[i];

    return v;
}

__global__
void denseKernel(
        float *W,
        float *b,
        float *x,
        float *z,
        int input_size)
{
    int neuron = threadIdx.x;

    float sum = b[neuron];

    for(int j=0;j<input_size;j++)
    {
        sum += W[neuron*input_size+j] * x[j];
    }

    z[neuron]=sum;
}

__global__
void sigmoidKernel(float *x)
{
    int i=threadIdx.x;

    x[i]=1.0f/(1.0f+expf(-x[i]));
}

// Softmax runs on the host (same as the CPU/OMP versions it is not parallelised)
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

int argmax(vector<float> x)
{
    return max_element(x.begin(), x.end()) - x.begin();
}

int main()
{
    // Load trained weights (flattened row-major)
    vector<float> W1 = loadMatrix("W1.txt",HIDDEN_SIZE,INPUT_SIZE);
    vector<float> b1 = loadVector("b1.txt",HIDDEN_SIZE);

    vector<float> W2 = loadMatrix("W2.txt",OUTPUT_SIZE,HIDDEN_SIZE);
    vector<float> b2 = loadVector("b2.txt",OUTPUT_SIZE);

    // One Iris sample
    vector<float> x =
    {
        5.1f,
        3.5f,
        1.4f,
        0.2f
    };

    // Device buffers
    float *d_W1,*d_b1,*d_W2,*d_b2,*d_x,*d_hidden,*d_logits;

    cudaMalloc(&d_W1, W1.size()*sizeof(float));
    cudaMalloc(&d_b1, b1.size()*sizeof(float));
    cudaMalloc(&d_W2, W2.size()*sizeof(float));
    cudaMalloc(&d_b2, b2.size()*sizeof(float));
    cudaMalloc(&d_x, INPUT_SIZE*sizeof(float));
    cudaMalloc(&d_hidden, HIDDEN_SIZE*sizeof(float));
    cudaMalloc(&d_logits, OUTPUT_SIZE*sizeof(float));

    // Weights and input never change -> copy to the GPU once, before timing
    cudaMemcpy(d_W1, W1.data(), W1.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1.data(), b1.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2.data(), W2.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2.data(), b2.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,  x.data(),  INPUT_SIZE*sizeof(float), cudaMemcpyHostToDevice);

    const int NUM_RUNS = 100000;

    vector<float> logits(OUTPUT_SIZE);
    vector<float> probs;

    auto start = chrono::high_resolution_clock::now();

    for(int i=0;i<NUM_RUNS;i++)
    {
        // Layer 1 : dense + sigmoid  (one GPU thread per hidden neuron)
        denseKernel<<<1,HIDDEN_SIZE>>>(d_W1,d_b1,d_x,d_hidden,INPUT_SIZE);
        sigmoidKernel<<<1,HIDDEN_SIZE>>>(d_hidden);

        // Layer 2 : dense            (one GPU thread per output neuron)
        denseKernel<<<1,OUTPUT_SIZE>>>(d_W2,d_b2,d_hidden,d_logits,HIDDEN_SIZE);

        // Copy logits back and finish with softmax on the host
        cudaMemcpy(logits.data(), d_logits, OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
        probs = softmax(logits);
    }

    cudaDeviceSynchronize();

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

    cout << "Execution            : CUDA (GPU)\n";

    cout << "\nProbabilities\n";

    for(float p : probs)
        cout << p << endl;

    cout << "\nPrediction = "
         << argmax(probs)
         << endl;

    cudaFree(d_W1);
    cudaFree(d_b1);
    cudaFree(d_W2);
    cudaFree(d_b2);
    cudaFree(d_x);
    cudaFree(d_hidden);
    cudaFree(d_logits);

    return 0;
}
