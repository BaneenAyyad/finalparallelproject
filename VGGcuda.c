#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define SIZE 224
#define CONV_SIZE 3
#define NUM_LAYERS 13
#define MAX_THREADS 1024
// Layer shapes
int cshape[NUM_LAYERS][4] = {
    {64, 3, CONV_SIZE, CONV_SIZE},
    {64, 64, CONV_SIZE, CONV_SIZE},
    {128, 64, CONV_SIZE, CONV_SIZE},
    // Add remaining shapes...
};

int dshape[3][2] = {
    {25088, 4096},
    {4096, 4096},
    {4096, 1000}
};

// Host dense weights and biases
float ***wd;
float **bd;
__global__ void convolutionKernel(float *input, float *kernel, float *output, int input_size, int kernel_size, int output_channels, int input_channels);
__global__ void reluAndBiasKernel(float *output, float *bias, int size, int channels);
__global__ void maxPoolingKernel(float *input, float *output, int input_size, int channels);
__global__ void denseLayerKernel(float *input, float *weights, float *output, int input_size, int output_size);
__global__ void softmaxKernel(float *input, int size);



// Host memory pointers
float ***image; // Host image data
float *****wc;  // Convolution weights
float **bc;     // Convolution biases
float *flattened_output; // Flattened feature map for dense layers

// GPU memory pointers
float *d_image;          // Device image
float *d_weights[NUM_LAYERS]; // Device convolution weights
float *d_biases[NUM_LAYERS];  // Device biases
float *d_intermediate;   // Intermediate results (e.g., feature maps)
float *d_dense_weights[3];
float *d_dense_biases[3];
float *d_dense_output;

// Kernel prototypes
__global__ void convolutionKernel(float *input, float *kernel, float *output, int input_size, int kernel_size);
__global__ void reluAndBiasKernel(float *output, float *bias, int size);
__global__ void maxPoolingKernel(float *input, float *output, int input_size);
__global__ void denseLayerKernel(float *input, float *weights, float *output, int input_size, int output_size);
__global__ void softmaxKernel(float *input, int size);

// Function prototypes
void initMemory();
void allocateGPU();
void copyToGPU();
void applyConvolutionLayer(int layer_idx, int input_size, int output_channels);
void applyMaxPoolingLayer(int input_size);
void applyDenseLayer(int layer_idx, int input_size, int output_size);
void softmax(float *output, int size);
void flattenOutput(float *input, float *output, int depth, int height, int width);
void freeMemory();
void freeGPUResources();
void allocateGPU() {
    // Allocate GPU memory for image
    cudaMalloc((void **)&d_image, 3 * SIZE * SIZE * sizeof(float));

    // Allocate GPU memory for convolutional weights and biases
    for (int l = 0; l < NUM_LAYERS; l++) {
        int weight_size = cshape[l][0] * cshape[l][1] * CONV_SIZE * CONV_SIZE;
        cudaMalloc((void **)&d_weights[l], weight_size * sizeof(float));
        cudaMalloc((void **)&d_biases[l], cshape[l][0] * sizeof(float));
    }

    // Allocate GPU memory for intermediate feature maps
    cudaMalloc((void **)&d_intermediate, 512 * SIZE * SIZE * sizeof(float));

    // Allocate GPU memory for dense weights and biases
    for (int l = 0; l < 3; l++) {
        cudaMalloc((void **)&d_dense_weights[l], dshape[l][0] * dshape[l][1] * sizeof(float));
        cudaMalloc((void **)&d_dense_biases[l], dshape[l][1] * sizeof(float));
    }

    // Allocate GPU memory for dense output
    cudaMalloc((void **)&d_dense_output, dshape[2][1] * sizeof(float));
}
void copyToGPU() {
    // Copy image data to GPU
    cudaMemcpy(d_image, image, 3 * SIZE * SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Copy convolutional weights and biases to GPU
    for (int l = 0; l < NUM_LAYERS; l++) {
        int weight_size = cshape[l][0] * cshape[l][1] * CONV_SIZE * CONV_SIZE;
        cudaMemcpy(d_weights[l], wc[l], weight_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_biases[l], bc[l], cshape[l][0] * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Copy dense weights and biases to GPU
    for (int l = 0; l < 3; l++) {
        cudaMemcpy(d_dense_weights[l], wd[l], dshape[l][0] * dshape[l][1] * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dense_biases[l], bd[l], dshape[l][1] * sizeof(float), cudaMemcpyHostToDevice);
    }
}
void initMemory() {
    // Allocate memory for the image
    image = (float ***)malloc(3 * sizeof(float **));
    for (int i = 0; i < 3; i++) {
        image[i] = (float **)malloc(SIZE * sizeof(float *));
        for (int j = 0; j < SIZE; j++) {
            image[i][j] = (float *)malloc(SIZE * sizeof(float));
        }
    }

    // Allocate memory for convolutional weights and biases
    wc = (float *****)malloc(NUM_LAYERS * sizeof(float ****));
    bc = (float **)malloc(NUM_LAYERS * sizeof(float *));
    for (int l = 0; l < NUM_LAYERS; l++) {
        wc[l] = (float ****)malloc(cshape[l][0] * sizeof(float ***));
        for (int i = 0; i < cshape[l][0]; i++) {
            wc[l][i] = (float ***)malloc(cshape[l][1] * sizeof(float **));
            for (int j = 0; j < cshape[l][1]; j++) {
                wc[l][i][j] = (float **)malloc(CONV_SIZE * sizeof(float *));
                for (int k = 0; k < CONV_SIZE; k++) {
                    wc[l][i][j][k] = (float *)malloc(CONV_SIZE * sizeof(float));
                }
            }
        }
        bc[l] = (float *)malloc(cshape[l][0] * sizeof(float));
    }
}
void loadModelWeights(const char *weights_file) {
    float weight_value;
    int filter_idx, input_channel, row, col, layer_idx;
    FILE *file_ptr;
    int levels_loaded = 0;

    file_ptr = fopen(weights_file, "r");
    if (file_ptr == NULL) {
        fprintf(stderr, "Error: Unable to open weights file %s\n", weights_file);
        exit(1);
    }

    // Loop through convolutional layers to read weights and biases
    for (layer_idx = 0; layer_idx < NUM_LAYERS; layer_idx++) {
        printf("Loading weights for convolutional block %d\n", layer_idx);

        for (filter_idx = 0; filter_idx < cshape[layer_idx][0]; filter_idx++) {
            for (input_channel = 0; input_channel < cshape[layer_idx][1]; input_channel++) {
                for (row = 0; row < CONV_SIZE; row++) {
                    for (col = 0; col < CONV_SIZE; col++) {
                        if (fscanf(file_ptr, "%f", &weight_value) != 1) {
                            fprintf(stderr, "Error: Failed to read weight for block %d\n", layer_idx);
                            fclose(file_ptr);
                            exit(1);
                        }
                        wc[layer_idx][filter_idx][input_channel][row][col] = weight_value;
                    }
                }
            }
        }

        // Load biases
        for (filter_idx = 0; filter_idx < cshape[layer_idx][0]; filter_idx++) {
            if (fscanf(file_ptr, "%f", &weight_value) != 1) {
                fprintf(stderr, "Error: Failed to read bias for block %d\n", layer_idx);
                fclose(file_ptr);
                exit(1);
            }
            bc[layer_idx][filter_idx] = weight_value;
        }

        levels_loaded++;
    }

    fclose(file_ptr);
}
void loadImageData(const char *image_file) {
    int row, col, channel;
    FILE *file_ptr;
    float pixel_value;

    file_ptr = fopen(image_file, "r");
    if (file_ptr == NULL) {
        fprintf(stderr, "Error: Unable to open image file %s\n", image_file);
        exit(1);
    }

    // Read image data for all channels
    for (channel = 0; channel < 3; channel++) {
        for (row = 0; row < SIZE; row++) {
            for (col = 0; col < SIZE; col++) {
                if (fscanf(file_ptr, "%f", &pixel_value) != 1) {
                    fprintf(stderr, "Error: Failed to read pixel at row %d, col %d, channel %d\n", row, col, channel);
                    fclose(file_ptr);
                    exit(1);
                }
                image[channel][row][col] = pixel_value;
            }
        }
    }

    fclose(file_ptr);
}


__global__ void convolutionKernel(float *input, float *kernel, float *output, int input_size, int kernel_size, int output_channels, int input_channels) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.z;

    if (row < input_size && col < input_size && channel < output_channels) {
        float value = 0.0;
        for (int c = 0; c < input_channels; c++) {
            for (int i = 0; i < kernel_size; i++) {
                for (int j = 0; j < kernel_size; j++) {
                    int x = row + i - kernel_size / 2;
                    int y = col + j - kernel_size / 2;
                    if (x >= 0 && x < input_size && y >= 0 && y < input_size) {
                        value += input[c * input_size * input_size + x * input_size + y] *
                                 kernel[channel * input_channels * kernel_size * kernel_size + c * kernel_size * kernel_size + i * kernel_size + j];
                    }
                }
            }
        }
        output[channel * input_size * input_size + row * input_size + col] = value;
    }
}

__global__ void reluAndBiasKernel(float *output, float *bias, int size, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = idx / (size * size);
    if (idx < size * size * channels) {
        output[idx] += bias[channel];
        if (output[idx] < 0.0) {
            output[idx] = 0.0;
        }
    }
}

__global__ void maxPoolingKernel(float *input, float *output, int input_size, int channels) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.z;

    if (row < input_size / 2 && col < input_size / 2 && channel < channels) {
        int idx = channel * input_size * input_size + (2 * row) * input_size + (2 * col);
        float max_val = input[idx];
        max_val = fmaxf(max_val, input[idx + 1]);
        max_val = fmaxf(max_val, input[idx + input_size]);
        max_val = fmaxf(max_val, input[idx + input_size + 1]);

        output[channel * (input_size / 2) * (input_size / 2) + row * (input_size / 2) + col] = max_val;
    }
}

void applyConvolutionLayer(int layer_idx, int input_size, int output_channels, int input_channels) {
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((input_size + 15) / 16, (input_size + 15) / 16, output_channels);

    convolutionKernel<<<numBlocks, threadsPerBlock>>>(d_intermediate, d_weights[layer_idx], d_intermediate, input_size, CONV_SIZE, output_channels, input_channels);
    cudaDeviceSynchronize();

    dim3 reluThreadsPerBlock(256);
    dim3 reluNumBlocks((input_size * input_size * output_channels + 255) / 256);
    reluAndBiasKernel<<<reluNumBlocks, reluThreadsPerBlock>>>(d_intermediate, d_biases[layer_idx], input_size, output_channels);
    cudaDeviceSynchronize();
}

__global__ void denseLayerKernel(float *input, float *weights, float *output, int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < output_size) {
        float value = 0.0;
        for (int i = 0; i < input_size; i++) {
            value += input[i] * weights[i * output_size + idx];
        }
        output[idx] = value;
    }
}
__global__ void reluDenseKernel(float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size && output[idx] < 0.0) {
        output[idx] = 0.0;
    }
}
__global__ void softmaxKernel(float *input, int size) {
    __shared__ float max_val;
    __shared__ float sum_val;

    int idx = threadIdx.x;

    if (idx == 0) {
        max_val = input[0];
        for (int i = 1; i < size; i++) {
            max_val = fmaxf(max_val, input[i]);
        }
    }
    __syncthreads();

    float exp_val = expf(input[idx] - max_val);

    if (idx == 0) {
        sum_val = 0.0;
        for (int i = 0; i < size; i++) {
            sum_val += expf(input[i] - max_val);
        }
    }
    __syncthreads();

    input[idx] = exp_val / sum_val;
}

void applyDenseLayer(int layer_idx, int input_size, int output_size) {
    dim3 threadsPerBlock(256);
    dim3 numBlocks((output_size + 255) / 256);

    denseLayerKernel<<<numBlocks, threadsPerBlock>>>(d_intermediate, d_dense_weights[layer_idx], d_dense_output, input_size, output_size);
    cudaDeviceSynchronize();

    reluDenseKernel<<<numBlocks, threadsPerBlock>>>(d_dense_output, output_size);
    cudaDeviceSynchronize();
}
void softmax(float *output, int size) {
    dim3 threadsPerBlock(size);
    softmaxKernel<<<1, threadsPerBlock>>>(output, size);
    cudaDeviceSynchronize();
}
void flattenOutput(float *input, float *output, int depth, int height, int width) {
    int idx = 0;
    for (int d = 0; d < depth; d++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                output[idx++] = input[d * height * width + h * width + w];
            }
        }
    }
}
int getPrediction(float *output, int size) {
    int max_idx = 0;
    float max_val = output[0];
    for (int i = 1; i < size; i++) {
        if (output[i] > max_val) {
            max_val = output[i];
            max_idx = i;
        }
    }
    return max_idx;
}
void freeMemory() {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < SIZE; j++) {
            free(image[i][j]);
        }
        free(image[i]);
    }
    free(image);

    for (int l = 0; l < NUM_LAYERS; l++) {
        for (int i = 0; i < cshape[l][0]; i++) {
            for (int j = 0; j < cshape[l][1]; j++) {
                for (int k = 0; k < CONV_SIZE; k++) {
                    free(wc[l][i][j][k]);
                }
                free(wc[l][i][j]);
            }
            free(wc[l][i]);
        }
        free(wc[l]);
        free(bc[l]);
    }
    free(wc);
    free(bc);
    free(flattened_output);
}

void freeGPUResources() {
    cudaFree(d_image);
    for (int l = 0; l < NUM_LAYERS; l++) {
        cudaFree(d_weights[l]);
        cudaFree(d_biases[l]);
    }
    cudaFree(d_intermediate);
    for (int l = 0; l < 3; l++) {
        cudaFree(d_dense_weights[l]);
        cudaFree(d_dense_biases[l]);
    }
    cudaFree(d_dense_output);
}
int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <weights file> <image file> <output file>\n", argv[0]);
        return 1;
    }

    char *weights_file = argv[1];
    char *image_file = argv[2];
    char *output_file = argv[3];

    // Initialize memory
    initMemory();
    allocateGPU();

    // Load weights and image (load_model_weights and load_image_data functions)
    loadModelWeights(weights_file);
    loadImageData(image_file);

    // Copy data to GPU
    copyToGPU();

    // Apply convolutional and max-pooling layers
    int input_size = SIZE;
    for (int l = 0; l < NUM_LAYERS; l++) {
        applyConvolutionLayer(l, input_size, cshape[l][0], cshape[l][1]);
        if ((l + 1) % 2 == 0) { // Assuming max-pooling every 2 layers
            applyMaxPoolingLayer(input_size);
            input_size /= 2;
        }
    }

    // Flatten the last layer's output
    flattenOutput(d_intermediate, flattened_output, 512, 7, 7);

    // Apply dense layers
    int input_size_dense = 512 * 7 * 7;
    for (int l = 0; l < 3; l++) {
        applyDenseLayer(l, input_size_dense, dshape[l][1]);
        input_size_dense = dshape[l][1];
    }

    // Apply softmax and get the prediction
    softmax(d_dense_output, dshape[2][1]);
    int predicted_class = getPrediction(d_dense_output, dshape[2][1]);

    // Write prediction to output file
    FILE *out_file = fopen(output_file, "w");
    if (out_file == NULL) {
        printf("Error: Could not open output file.\n");
        return 1;
    }
    fprintf(out_file, "Predicted Class: %d\n", predicted_class);
    fclose(out_file);

    // Free memory
    freeMemory();
    freeGPUResources();

    return 0;
}
