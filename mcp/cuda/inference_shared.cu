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
        for(int j=0;j<cols;j++)
            file>>matrix[i*cols+j];
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

// Shared-memory dense: the input vector for this sample is loaded ONCE into
// on-chip shared memory, then reused by every output neuron in the block.
// Same math as before -- only the memory access pattern changes.
__global__
void denseKernelShared(
        float *W, float *b, float *X, float *Z,
        int input_size, int output_size)
{
    extern __shared__ float xs[];   // holds this sample's input_size values

    int sample = blockIdx.x;
    int neuron = threadIdx.x;

    // Cooperatively load the input vector into shared memory (once per block)
    for(int j=neuron; j<input_size; j+=blockDim.x)
        xs[j] = X[sample*input_size + j];

    __syncthreads();                // make sure the load is complete

    if(neuron < output_size)
    {
        float sum = b[neuron];
        for(int j=0;j<input_size;j++)
            sum += W[neuron*input_size+j] * xs[j];   // read from shared, not global
        Z[sample*output_size + neuron] = sum;
    }
}

__global__
void sigmoidKernelBatch(float *X, int total)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < total)
        X[i] = 1.0f/(1.0f+expf(-X[i]));
}

int main(int argc, char** argv)
{
    int BATCH = (argc > 1) ? atoi(argv[1]) : 1000;

    vector<float> W1 = loadMatrix("W1.txt",HIDDEN_SIZE,INPUT_SIZE);
    vector<float> b1 = loadVector("b1.txt",HIDDEN_SIZE);
    vector<float> W2 = loadMatrix("W2.txt",OUTPUT_SIZE,HIDDEN_SIZE);
    vector<float> b2 = loadVector("b2.txt",OUTPUT_SIZE);

    vector<float> sample = {5.1f, 3.5f, 1.4f, 0.2f};
    vector<float> X(BATCH*INPUT_SIZE);
    for(int s=0;s<BATCH;s++)
        for(int j=0;j<INPUT_SIZE;j++)
            X[s*INPUT_SIZE+j] = sample[j];

    float *d_W1,*d_b1,*d_W2,*d_b2,*d_X,*d_hidden,*d_logits;
    cudaMalloc(&d_W1, W1.size()*sizeof(float));
    cudaMalloc(&d_b1, b1.size()*sizeof(float));
    cudaMalloc(&d_W2, W2.size()*sizeof(float));
    cudaMalloc(&d_b2, b2.size()*sizeof(float));
    cudaMalloc(&d_X,      BATCH*INPUT_SIZE *sizeof(float));
    cudaMalloc(&d_hidden, BATCH*HIDDEN_SIZE*sizeof(float));
    cudaMalloc(&d_logits, BATCH*OUTPUT_SIZE*sizeof(float));

    cudaMemcpy(d_W1, W1.data(), W1.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1.data(), b1.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2.data(), W2.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2.data(), b2.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X,  X.data(),  X.size()*sizeof(float),  cudaMemcpyHostToDevice);

    vector<float> logits(BATCH*OUTPUT_SIZE);

    int NUM_RUNS = 100000 / BATCH;
    if(NUM_RUNS < 1) NUM_RUNS = 1;

    int sigBlocks = (BATCH*HIDDEN_SIZE + 255) / 256;

    auto start = chrono::high_resolution_clock::now();

    for(int r=0;r<NUM_RUNS;r++)
    {
        // shared-mem size = input length of that layer * sizeof(float)
        denseKernelShared<<<BATCH, HIDDEN_SIZE, INPUT_SIZE*sizeof(float)>>>(
                d_W1,d_b1,d_X,d_hidden,INPUT_SIZE,HIDDEN_SIZE);
        sigmoidKernelBatch<<<sigBlocks, 256>>>(d_hidden, BATCH*HIDDEN_SIZE);
        denseKernelShared<<<BATCH, HIDDEN_SIZE, HIDDEN_SIZE*sizeof(float)>>>(
                d_W2,d_b2,d_hidden,d_logits,HIDDEN_SIZE,OUTPUT_SIZE);
        cudaMemcpy(logits.data(), d_logits, logits.size()*sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaDeviceSynchronize();

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop-start);

    long totalSamples = (long)NUM_RUNS * BATCH;

    cout << "----------------------------------\n";
    cout << "Batched CUDA + Shared Memory\n";
    cout << "----------------------------------\n";
    cout << "Batch size           : " << BATCH << endl;
    cout << "Total samples        : " << totalSamples << endl;
    cout << "Total time           : " << duration.count() << " us\n";
    cout << "Per-sample inference : "
         << (double)duration.count()/totalSamples << " us\n";

    cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_X); cudaFree(d_hidden); cudaFree(d_logits);
    return 0;
}
