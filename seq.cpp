#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cmath>

// Function to apply a 2D convolution
void seqConv2D(const float *input, float *output, int w, int h, int c, int k, int filter_w, int filter_h,
               int pad_w, int pad_h, int stride_w, int stride_h) {
    int out_h = (h + 2 * pad_h - filter_h) / stride_h + 1;
    int out_w = (w + 2 * pad_w - filter_w) / stride_w + 1;

    // Create random filters for convolution
    std::vector<std::vector<std::vector<std::vector<float>>>> filters(k,
        std::vector<std::vector<std::vector<float>>>(c,
            std::vector<std::vector<float>>(filter_h,
                std::vector<float>(filter_w, 0.0f))));
    
    for (int nk = 0; nk < k; nk++) {
        for (int nc = 0; nc < c; nc++) {
            for (int i = 0; i < filter_h; i++) {
                for (int j = 0; j < filter_w; j++) {
                    filters[nk][nc][i][j] = static_cast<float>(rand()) / RAND_MAX;
                }
            }
        }
    }

    // Perform convolution
    for (int nk = 0; nk < k; nk++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float sum = 0.0f;
                for (int nc = 0; nc < c; nc++) {
                    for (int fh = 0; fh < filter_h; fh++) {
                        for (int fw = 0; fw < filter_w; fw++) {
                            int ih = oh * stride_h - pad_h + fh;
                            int iw = ow * stride_w - pad_w + fw;
                            if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                                sum += input[(nc * h + ih) * w + iw] * filters[nk][nc][fh][fw];
                            }
                        }
                    }
                }
                output[(nk * out_h + oh) * out_w + ow] = sum;
            }
        }
    }
}

// Function to perform max pooling
void seqMaxPool(const float *input, float *output, int w, int h, int c, int pool_h, int pool_w, int stride_h, int stride_w) {
    int out_h = h / stride_h;
    int out_w = w / stride_w;

    for (int nc = 0; nc < c; nc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float max_val = -INFINITY;
                for (int ph = 0; ph < pool_h; ph++) {
                    for (int pw = 0; pw < pool_w; pw++) {
                        int ih = oh * stride_h + ph;
                        int iw = ow * stride_w + pw;
                        if (ih < h && iw < w) {
                            max_val = std::max(max_val, input[(nc * h + ih) * w + iw]);
                        }
                    }
                }
                output[(nc * out_h + oh) * out_w + ow] = max_val;
            }
        }
    }
}

// Function to simulate a fully connected layer
void seqFC(const float *input, float *output, int in_size, int out_size) {
    std::vector<std::vector<float>> weights(out_size, std::vector<float>(in_size));
    std::vector<float> bias(out_size);

    // Initialize weights and biases randomly
    for (int i = 0; i < out_size; i++) {
        bias[i] = static_cast<float>(rand()) / RAND_MAX;
        for (int j = 0; j < in_size; j++) {
            weights[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Perform matrix multiplication and add bias
    for (int i = 0; i < out_size; i++) {
        output[i] = bias[i];
        for (int j = 0; j < in_size; j++) {
            output[i] += input[j] * weights[i][j];
        }
    }
}

int main() {
    srand(time(0));

    float *input = new float[224 * 224 * 3];
    for (int i = 0; i < 224 * 224 * 3; i++) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *output;

    // =============== Layer 1 =====================
    std::cout << "CONV 224x224x64\n";
    output = new float[224 * 224 * 64];
    auto start = std::chrono::steady_clock::now();
    seqConv2D(input, output, 224, 224, 3, 64, 3, 3, 1, 1, 1, 1);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    delete[] input;
    input = output;

    std::cout << "CONV 224x224x64\n";
    output = new float[224 * 224 * 64];
    start = std::chrono::steady_clock::now();
    seqConv2D(input, output, 224, 224, 64, 64, 3, 3, 1, 1, 1, 1);
    end = std::chrono::steady_clock::now();
    std::cout << "Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    delete[] input;
    input = output;

    std::cout << "POOLMAX 112x112x64\n";
    output = new float[112 * 112 * 64];
    start = std::chrono::steady_clock::now();
    seqMaxPool(input, output, 224, 224, 64, 2, 2, 2, 2);
    end = std::chrono::steady_clock::now();
    std::cout << "Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    delete[] input;
    input = output;

    // =============== Layer 2 =====================
    std::cout << "CONV 112x112x128\n";
    output = new float[112 * 112 * 128];
    start = std::chrono::steady_clock::now();
    seqConv2D(input, output, 112, 112, 64, 128, 3, 3, 1, 1, 1, 1);
    end = std::chrono::steady_clock::now();
    std::cout << "Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    delete[] input;
    input = output;

    std::cout << "CONV 112x112x128\n";
    output = new float[112 * 112 * 128];
    start = std::chrono::steady_clock::now();
    seqConv2D(input, output, 112, 112, 128, 128, 3, 3, 1, 1, 1, 1);
    end = std::chrono::steady_clock::now();
    std::cout << "Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    delete[] input;
    input = output;

    std::cout << "POOLMAX 56x56x128\n";
    output = new float[56 * 56 * 128];
    start = std::chrono::steady_clock::now();
    seqMaxPool(input, output, 112, 112, 128, 2, 2, 2, 2);
    end = std::chrono::steady_clock::now();
    std::cout << "Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    delete[] input;
    input = output;

    // =============== Layer 3 =====================
    std::cout << "CONV 56x56x256\n";
    output = new float[56 * 56 * 256];
    start = std::chrono::steady_clock::now();
    seqConv2D(input, output, 56, 56, 128, 256, 3, 3, 1, 1, 1, 1);
    end = std::chrono::steady_clock::now();
    std::cout << "Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    delete[] input;
    input = output;

    std::cout << "CONV 56x56x256\n";
    output = new float[56 * 56 * 256];
    start = std::chrono::steady_clock::now();
    seqConv2D(input, output, 56, 56, 256, 256, 3, 3, 1, 1, 1, 1);
    end = std::chrono::steady_clock::now();
    std::cout << "Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    delete[] input;
    input = output;

    std::cout << "POOLMAX 28x28x256\n";
    output = new float[28 * 28 * 256];
    start = std::chrono::steady_clock::now();
    seqMaxPool(input, output, 56, 56, 256, 2, 2, 2, 2);
    end = std::chrono::steady_clock::now();
    std::cout << "Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    delete[] input;
    input = output;

        // =============== Layer 4 =====================
    std::cout << "CONV 28x28x512\n";
    output = new float[28 * 28 * 512];
    start = std::chrono::steady_clock::now();
    seqConv2D(input, output, 28, 28, 256, 512, 3, 3, 1, 1, 1, 1);
    end = std::chrono::steady_clock::now();
    std::cout << "Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    delete[] input;
    input = output;

    std::cout << "CONV 28x28x512\n";
    output = new float[28 * 28 * 512];
    start = std::chrono::steady_clock::now();
    seqConv2D(input, output, 28, 28, 512, 512, 3, 3, 1, 1, 1, 1);
    end = std::chrono::steady_clock::now();
    std::cout << "Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    delete[] input;
    input = output;

    std::cout << "POOLMAX 14x14x512\n";
    output = new float[14 * 14 * 512];
    start = std::chrono::steady_clock::now();
    seqMaxPool(input, output, 28, 28, 512, 2, 2, 2, 2);
    end = std::chrono::steady_clock::now();
    std::cout << "Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    delete[] input;
    input = output;

    // =============== Layer 5 =====================
    std::cout << "CONV 14x14x512\n";
    output = new float[14 * 14 * 512];
    start = std::chrono::steady_clock::now();
    seqConv2D(input, output, 14, 14, 512, 512, 3, 3, 1, 1, 1, 1);
    end = std::chrono::steady_clock::now();
    std::cout << "Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    delete[] input;
    input = output;

    std::cout << "CONV 14x14x512\n";
    output = new float[14 * 14 * 512];
    start = std::chrono::steady_clock::now();
    seqConv2D(input, output, 14, 14, 512, 512, 3, 3, 1, 1, 1, 1);
    end = std::chrono::steady_clock::now();
    std::cout << "Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    delete[] input;
    input = output;

    std::cout << "POOLMAX 7x7x512\n";
    output = new float[7 * 7 * 512];
    start = std::chrono::steady_clock::now();
    seqMaxPool(input, output, 14, 14, 512, 2, 2, 2, 2);
    end = std::chrono::steady_clock::now();
    std::cout << "Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    delete[] input;
    input = output;

    // =============== Fully Connected Layers =====================
    std::cout << "FC 4096\n";
    output = new float[4096];
    start = std::chrono::steady_clock::now();
    seqFC(input, output, 7 * 7 * 512, 4096);
    end = std::chrono::steady_clock::now();
    std::cout << "Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    delete[] input;
    input = output;

    std::cout << "FC 4096\n";
    output = new float[4096];
    start = std::chrono::steady_clock::now();
    seqFC(input, output, 4096, 4096);
    end = std::chrono::steady_clock::now();
    std::cout << "Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    delete[] input;
    input = output;

    std::cout << "FC 1000\n";
    output = new float[1000];
    start = std::chrono::steady_clock::now();
    seqFC(input, output, 4096, 1000);
    end = std::chrono::steady_clock::now();
    std::cout << "Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    delete[] input;

    delete[] output;
    return 0;

}

