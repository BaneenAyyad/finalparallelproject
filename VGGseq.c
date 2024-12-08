/* 
	Pretrained VGG16 convolutional neural network in C language
	GitHUB Page: https://github.com/ZFTurbo/VGG16-Pretrained-C
	Author: ZFTurbo
	
	Compilation: gcc -O3 -fopenmp -lm ZFC_VGG16_CPU.c -o ZFC_VGG16_CPU.exe
	Usage: ZFC_VGG16_CPU.exe <weights_path> <file_with_list_of_images> <output file> <output convolution features (optional)>
	Example: ZFC_VGG16_CPU.exe "weights.txt" "image_list.txt" "results.txt" 1
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <ctype.h>  // For isspace()
#include <string.h> // For strlen()
#include <sys/time.h> // For gettimeofday()

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS 1
void gettimeofday(time_t *tp, char *_)
{
	*tp = clock();
	return;
}

double get_seconds(time_t timeStart, time_t timeEnd) {
	return (double)(timeEnd - timeStart) / CLOCKS_PER_SEC;
}
#else
double get_seconds(struct timeval timeStart, struct timeval timeEnd) {
	return ((timeEnd.tv_sec - timeStart.tv_sec) * 1000000 + timeEnd.tv_usec - timeStart.tv_usec) / 1.e6;
}
#endif

#define SIZE 224
#define CONV_SIZE 3
int numthreads;


// Weights and image block START
float ***image;
int cshape[13][4] = { 
	{ 64, 3, CONV_SIZE, CONV_SIZE },
	{ 64, 64, CONV_SIZE, CONV_SIZE },
	{ 128, 64, CONV_SIZE, CONV_SIZE },
	{ 128, 128, CONV_SIZE, CONV_SIZE },
	{ 256, 128, CONV_SIZE, CONV_SIZE },
	{ 256, 256, CONV_SIZE, CONV_SIZE },
	{ 256, 256, CONV_SIZE, CONV_SIZE },
	{ 512, 256, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE }
};
float *****wc;
float **bc;
int dshape[3][2] = {
	{ 25088, 4096 },
	{ 4096, 4096 },
	{ 4096, 1000 }
};
float ***wd;
float **bd;


// Blocks for intermediate convolutions
int mem_block_shape[3] = {512, SIZE, SIZE};
float ***mem_block1;
float ***mem_block2;
// Blocks for dense flatten layers
int mem_block_dense_shape = { 512 * 7 * 7 };
float *mem_block1_dense;
float *mem_block2_dense;

// Weights and image block END


void reset_mem_block(float ***mem) {
	int i, j, k;
	for (i = 0; i < mem_block_shape[0]; i++) {
		for (j = 0; j < mem_block_shape[1]; j++) {
			for (k = 0; k < mem_block_shape[2]; k++) {
				mem[i][j][k] = 0.0;
			}
		}
	}
}


void reset_mem_block_dense(float *mem) {
	int i;
	for (i = 0; i < mem_block_dense_shape; i++) {
		mem[i] = 0.0;
	}
}


void init_memory() {
	int i, j, k, l;

	// Init image memory
	image = malloc(3 * sizeof(float**));
	for (i = 0; i < 3; i++) {
		image[i] = malloc(SIZE * sizeof(float*));
		for (j = 0; j < SIZE; j++) {
			image[i][j] = malloc(SIZE * sizeof(float));
		}
	}

	// Init convolution weights
	wc = malloc(13 * sizeof(float****));
	bc = malloc(13 * sizeof(float*));
	for (l = 0; l < 13; l++) {
		wc[l] = malloc(cshape[l][0] * sizeof(float***));
		for (i = 0; i < cshape[l][0]; i++) {
			wc[l][i] = malloc(cshape[l][1] * sizeof(float**));
			for (j = 0; j < cshape[l][1]; j++) {
				wc[l][i][j] = malloc(cshape[l][2] * sizeof(float*));
				for (k = 0; k < cshape[l][2]; k++) {
					wc[l][i][j][k] = malloc(cshape[l][3] * sizeof(float));
				}
			}
		}
		bc[l] = malloc(cshape[l][0] * sizeof(float));
	}

	// Init dense weights
	wd = malloc(3 * sizeof(float**));
	bd = malloc(3 * sizeof(float*));
	for (l = 0; l < 3; l++) {
		wd[l] = malloc(dshape[l][0] * sizeof(float*));
		for (i = 0; i < dshape[l][0]; i++) {
			wd[l][i] = malloc(dshape[l][1] * sizeof(float));
		}
		bd[l] = malloc(dshape[l][1] * sizeof(float));
	}

	// Init mem_blocks
	mem_block1 = malloc(mem_block_shape[0] * sizeof(float**));
	mem_block2 = malloc(mem_block_shape[0] * sizeof(float**));
	for (i = 0; i < mem_block_shape[0]; i++) {
		mem_block1[i] = malloc(mem_block_shape[1] * sizeof(float*));
		mem_block2[i] = malloc(mem_block_shape[1] * sizeof(float*));
		for (j = 0; j < mem_block_shape[1]; j++) {
			mem_block1[i][j] = malloc(mem_block_shape[2] * sizeof(float));
			mem_block2[i][j] = malloc(mem_block_shape[2] * sizeof(float));
		}
	}
	reset_mem_block(mem_block1);
	reset_mem_block(mem_block2);

	// Init mem blocks dense
	mem_block1_dense = calloc(mem_block_dense_shape, sizeof(float));
	mem_block2_dense = calloc(mem_block_dense_shape, sizeof(float));
}


void free_memory() {
	int i, j, k, l;

	// Free image memory
	for (i = 0; i < 3; i++) {
		for (j = 0; j < SIZE; j++) {
			free(image[i][j]);
		}
		free(image[i]);
	}
	free(image);

	// Free convolution weights
	for (l = 0; l < 13; l++) {
		for (i = 0; i < cshape[l][0]; i++) {
			for (j = 0; j < cshape[l][1]; j++) {
				for (k = 0; k < cshape[l][2]; k++) {
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

	// Free dense weights
	for (l = 0; l < 3; l++) {
		for (i = 0; i < dshape[l][0]; i++) {
			free(wd[l][i]);
		}
		free(wd[l]);
		free(bd[l]);
	}
	free(wd);
	free(bd);

	// Free memblocks
	for (i = 0; i < mem_block_shape[0]; i++) {
		for (j = 0; j < mem_block_shape[1]; j++) {
			free(mem_block1[i][j]);
			free(mem_block2[i][j]);
		}
		free(mem_block1[i]);
		free(mem_block2[i]);
	}
	free(mem_block1);
	free(mem_block2);

	free(mem_block1_dense);
	free(mem_block2_dense);
}

void load_model_weights(char *file_path, int max_levels) {
    float weight_value;
    int filter_idx, input_channel, row, col, layer_idx;
    FILE *file_ptr;
    int levels_loaded = 0;

    file_ptr = fopen(file_path, "r");
    if (file_ptr == NULL) {
        fprintf(stderr, "Error: Unable to open file %s\n", file_path);
        exit(1);
    }

    // Loop through convolutional layers to read weights and biases
    for (layer_idx = 0; layer_idx < 13; layer_idx++) {
        if (levels_loaded >= max_levels && max_levels != -1)
            break;

        printf("Loading weights for convolutional block %d\n", layer_idx);

        for (filter_idx = 0; filter_idx < cshape[layer_idx][0]; filter_idx++) {
            for (input_channel = 0; input_channel < cshape[layer_idx][1]; input_channel++) {
                for (row = 0; row < cshape[layer_idx][2]; row++) {
                    for (col = 0; col < cshape[layer_idx][3]; col++) {
                        if (fscanf(file_ptr, "%f", &weight_value) != 1) {
                            fprintf(stderr, "Error: Failed to read weight for block %d\n", layer_idx);
                            fclose(file_ptr);
                            exit(1);
                        }
                        wc[layer_idx][filter_idx][input_channel][CONV_SIZE - row - 1][CONV_SIZE - col - 1] = weight_value;
                    }
                }
            }
        }

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




void load_image_data(char *file_path) {
    int row, col, channel;
    FILE *file_ptr;
    float pixel_value;

    file_ptr = fopen(file_path, "r");
    if (file_ptr == NULL) {
        fprintf(stderr, "Error: Unable to open file %s\n", file_path);
        exit(1);
    }

    // Read image data for all channels
    for (row = 0; row < SIZE; row++) {
        for (col = 0; col < SIZE; col++) {
            for (channel = 0; channel < 3; channel++) {
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




void normalize_image() {
	int i, j, l;
	float coef[3] = { 103.939, 116.779, 123.68 };

	for (l = 0; l < 3; l++) {
		for (i = 0; i < SIZE; i++) {
			for (j = 0; j < SIZE; j++) {
				image[l][i][j] -= coef[l];
			}
		}
	}
}

void apply_convolution_3x3(float **input_matrix, float **filter, float **output_matrix, int dimension) {
    int row, col;
    float pixel_sum;
    float padded_matrix[SIZE + 2][SIZE + 2] = { 0.0 };

    // Apply zero-padding to the input matrix
    for (row = 0; row < dimension; row++) {
        for (col = 0; col < dimension; col++) {
            padded_matrix[row + 1][col + 1] = input_matrix[row][col];
        }
    }

    // Perform 3x3 convolution
    for (row = 0; row < dimension; row++) {
        for (col = 0; col < dimension; col++) {
            pixel_sum = padded_matrix[row][col] * filter[0][0] +
                        padded_matrix[row + 1][col] * filter[1][0] +
                        padded_matrix[row + 2][col] * filter[2][0] +
                        padded_matrix[row][col + 1] * filter[0][1] +
                        padded_matrix[row + 1][col + 1] * filter[1][1] +
                        padded_matrix[row + 2][col + 1] * filter[2][1] +
                        padded_matrix[row][col + 2] * filter[0][2] +
                        padded_matrix[row + 1][col + 2] * filter[1][2] +
                        padded_matrix[row + 2][col + 2] * filter[2][2];
            output_matrix[row][col] += pixel_sum;
        }
    }
}



void apply_bias_and_relu(float **output_matrix, float bias, int dimension) {
    int row, col;

    for (row = 0; row < dimension; row++) {
        for (col = 0; col < dimension; col++) {
            output_matrix[row][col] += bias;
            if (output_matrix[row][col] < 0)
                output_matrix[row][col] = 0.0;
        }
    }
}


void add_bias_and_relu_flatten(float *out, float *bs, int size, int relu) {
	int i;
	for (i = 0; i < size; i++) {
		out[i] += bs[i];
		if (relu == 1) {
			if (out[i] < 0)
				out[i] = 0.0;
		}
	}
}


float max_of_4(float a, float b, float c, float d) {
	if (a >= b && a >= c && a >= d) {
		return a;
	}
	if (b >= c && b >= d) {
		return b;
	}
	if (c >= d) {
		return c;
	}
	return d;
}


void perform_maxpooling(float **input_matrix, int input_size) {
    int row, col;
    for (row = 0; row < input_size; row += 2) {
        for (col = 0; col < input_size; col += 2) {
            input_matrix[row / 2][col / 2] = max_of_4(
                input_matrix[row][col],
                input_matrix[row + 1][col],
                input_matrix[row][col + 1],
                input_matrix[row + 1][col + 1]
            );
        }
    }
}



void flatten(float ***in, float *out, int sh0, int sh1, int sh2) {
	int i, j, k, total = 0;
	for (i = 0; i < sh0; i++) {
		for (j = 0; j < sh1; j++) {
			for (k = 0; k < sh2; k++) {
				out[total] = in[i][j][k];
				total += 1;
			}
		}
	}
}


void dense(float *in, float **weights, float *out, int sh_in, int sh_out) {
	int i, j;
	for (i = 0; i < sh_out; i++) {
		float sum = 0.0;
		for (j = 0; j < sh_in; j++) {
			sum += in[j] * weights[j][i];
		}
		out[i] = sum;
	}
}


void softmax(float *out, int sh_out) {
	int i;
	float max_val, sum;
	max_val = out[0];
	for (i = 1; i < sh_out; i++) {
		if (out[i] > max_val)
			max_val = out[i];
	}
	sum = 0.0;
	for (i = 0; i < sh_out; i++) {
		out[i] = exp(out[i] - max_val);
		sum += out[i];
	}
	for (i = 0; i < sh_out; i++) {
		out[i] /= sum;
	}
}



void dump_memory_structure_conv(float ***mem, int sh0, int sh1, int sh2) {
	int i, j, k;
	for (i = 0; i < sh0; i++) {
		for (j = 0; j < sh1; j++) {
			for (k = 0; k < sh2; k++) {
				printf("%.12lf\n", mem[i][j][k]);
			}
		}
	}
}

void dump_memory_structure_conv_to_file(float ***mem, int sh0, int sh1, int sh2) {
	FILE *out;
	int i, j, k;
	out = fopen("debug_c.txt", "w");
	for (i = 0; i < sh0; i++) {
		for (j = 0; j < sh1; j++) {
			for (k = 0; k < sh2; k++) {
				fprintf(out, "%.12lf\n", mem[i][j][k]);
			}
		}
	}
	fclose(out);
}


void dump_memory_structure_dense(float *mem, int sh0) {
	int i;
	for (i = 0; i < sh0; i++) {
		printf("%.12lf\n", mem[i]);
	}
}


void dump_memory_structure_dense_to_file(float *mem, int sh0) {
	FILE *out;
	int i;
	out = fopen("debug_c.txt", "w");
	for (i = 0; i < sh0; i++) {
		fprintf(out, "%.12lf\n", mem[i]);
	}
	fclose(out);
}

void dump_image() {
	int i, j, k;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < SIZE; j++) {
			for (k = 0; k < SIZE; k++) {
				printf("%.12lf\n", image[i][j][k]);
			}
		}
	}
}


void get_VGG16_predict(int only_convolution) {
	int layer_index, channel_idx;
	int level, cur_size;

	// Init intermediate memory
	reset_mem_block(mem_block1);
	reset_mem_block(mem_block2);
	reset_mem_block_dense(mem_block1_dense);
	reset_mem_block_dense(mem_block2_dense);

	// Layer 1 (Convolution 3 -> 64)
	level = 0;
	cur_size = SIZE;
	for (layer_index = 0; layer_index < cshape[level][0]; layer_index++) {
		for (channel_idx = 0; channel_idx < cshape[level][1]; channel_idx++) {
			apply_convolution_3x3(image[channel_idx], wc[level][layer_index][channel_idx], mem_block1[layer_index], cur_size);
		}
		apply_bias_and_relu(mem_block1[layer_index], bc[level][layer_index], cur_size);
	}
	
	// Layer 2 (Convolution 64 -> 64)
	level = 1;
	for (layer_index = 0; layer_index < cshape[level][0]; layer_index++) {
		for (channel_idx = 0; channel_idx < cshape[level][1]; channel_idx++) {
			apply_convolution_3x3(mem_block1[channel_idx], wc[level][layer_index][channel_idx], mem_block2[layer_index], cur_size);
		}
		apply_bias_and_relu(mem_block2[layer_index], bc[level][layer_index], cur_size);
	}
	reset_mem_block(mem_block1);
	
	// Layer 3 (perform_maxpooling)

	for (layer_index= 0; layer_index< cshape[level][0]; layer_index++) {
		perform_maxpooling(mem_block2[layer_index], cur_size);
	}
	cur_size /= 2;
	
	// Layer 4 (Convolution 64 -> 128)
	level = 2;

	for (layer_index= 0; layer_index< cshape[level][0]; layer_index++) {
		for (channel_idx = 0; channel_idx < cshape[level][1]; channel_idx++) {
			apply_convolution_3x3(mem_block2[channel_idx], wc[level][layer_index][channel_idx], mem_block1[layer_index], cur_size);
		}
		apply_bias_and_relu(mem_block1[layer_index], bc[level][layer_index], cur_size);
	}
	reset_mem_block(mem_block2);

	// Layer 5 (Convolution 128 -> 128)
	level = 3;

	for (layer_index= 0; layer_index< cshape[level][0]; layer_index++) {
		for (channel_idx = 0; channel_idx < cshape[level][1]; channel_idx++) {
			apply_convolution_3x3(mem_block1[channel_idx], wc[level][layer_index][channel_idx], mem_block2[layer_index], cur_size);
		}
		apply_bias_and_relu(mem_block2[layer_index], bc[level][layer_index], cur_size);
	}
	reset_mem_block(mem_block1);
	
	// Layer 6 (perform_maxpooling)

	for (layer_index= 0; layer_index< cshape[level][0]; layer_index++) {
		perform_maxpooling(mem_block2[layer_index], cur_size);
	}
	cur_size /= 2;

	// Layer 7 (Convolution 128 -> 256)
	level = 4;


	for (layer_index= 0; layer_index< cshape[level][0]; layer_index++) {
		for (channel_idx = 0; channel_idx < cshape[level][1]; channel_idx++) {
			apply_convolution_3x3(mem_block2[channel_idx], wc[level][layer_index][channel_idx], mem_block1[layer_index], cur_size);
		}
		apply_bias_and_relu(mem_block1[layer_index], bc[level][layer_index], cur_size);
	}
	reset_mem_block(mem_block2);

	// Layer 8 (Convolution 256 -> 256)
	level = 5;


	for (layer_index= 0; layer_index< cshape[level][0]; layer_index++) {
		for (channel_idx = 0; channel_idx < cshape[level][1]; channel_idx++) {
			apply_convolution_3x3(mem_block1[channel_idx], wc[level][layer_index][channel_idx], mem_block2[layer_index], cur_size);
		}
		apply_bias_and_relu(mem_block2[layer_index], bc[level][layer_index], cur_size);
	}
	reset_mem_block(mem_block1);

	// Layer 9 (Convolution 256 -> 256)
	level = 6;


	for (layer_index= 0; layer_index< cshape[level][0]; layer_index++) {
		for (channel_idx = 0; channel_idx < cshape[level][1]; channel_idx++) {
			apply_convolution_3x3(mem_block2[channel_idx], wc[level][layer_index][channel_idx], mem_block1[layer_index], cur_size);
		}
		apply_bias_and_relu(mem_block1[layer_index], bc[level][layer_index], cur_size);
	}
	reset_mem_block(mem_block2);
	
	// Layer 10 (perform_maxpooling)
	for (layer_index= 0; layer_index< cshape[level][0]; layer_index++) {
		perform_maxpooling(mem_block1[layer_index], cur_size);
	}
	cur_size /= 2;
	
	// Layer 11 (Convolution 256 -> 512)
	level = 7;
	for (layer_index= 0; layer_index< cshape[level][0]; layer_index++) {
		for (channel_idx = 0; channel_idx < cshape[level][1]; channel_idx++) {
			apply_convolution_3x3(mem_block1[channel_idx], wc[level][layer_index][channel_idx], mem_block2[layer_index], cur_size);
		}
		apply_bias_and_relu(mem_block2[layer_index], bc[level][layer_index], cur_size);
	}
	reset_mem_block(mem_block1);

	// Layer 12 (Convolution 512 -> 512)
	level = 8;
	for (layer_index= 0; layer_index< cshape[level][0]; layer_index++) {
		for (channel_idx = 0; channel_idx < cshape[level][1]; channel_idx++) {
			apply_convolution_3x3(mem_block2[channel_idx], wc[level][layer_index][channel_idx], mem_block1[layer_index], cur_size);
		}
		apply_bias_and_relu(mem_block1[layer_index], bc[level][layer_index], cur_size);
	}
	reset_mem_block(mem_block2);

	// Layer 13 (Convolution 512 -> 512)
	level = 9;
	for (layer_index= 0; layer_index< cshape[level][0]; layer_index++) {
		for (channel_idx = 0; channel_idx < cshape[level][1]; channel_idx++) {
			apply_convolution_3x3(mem_block1[channel_idx], wc[level][layer_index][channel_idx], mem_block2[layer_index], cur_size);
		}
		apply_bias_and_relu(mem_block2[layer_index], bc[level][layer_index], cur_size);
	}
	reset_mem_block(mem_block1);
	
	// Layer 14 (perform_maxpooling)
	for (layer_index= 0; layer_index< cshape[level][0]; layer_index++) {
		perform_maxpooling(mem_block2[layer_index], cur_size);
	}
	cur_size /= 2;
	
	// Layer 15 (Convolution 512 -> 512)
	level = 10;
	for (layer_index= 0; layer_index< cshape[level][0]; layer_index++) {
		for (channel_idx = 0; channel_idx < cshape[level][1]; channel_idx++) {
			apply_convolution_3x3(mem_block2[channel_idx], wc[level][layer_index][channel_idx], mem_block1[layer_index], cur_size);
		}
		apply_bias_and_relu(mem_block1[layer_index], bc[level][layer_index], cur_size);
	}
	reset_mem_block(mem_block2);

	// Layer 16 (Convolution 512 -> 512)
	level = 11;
	for (layer_index= 0; layer_index< cshape[level][0]; layer_index++) {
		for (channel_idx = 0; channel_idx < cshape[level][1]; channel_idx++) {
			apply_convolution_3x3(mem_block1[channel_idx], wc[level][layer_index][channel_idx], mem_block2[layer_index], cur_size);
		}
		apply_bias_and_relu(mem_block2[layer_index], bc[level][layer_index], cur_size);
	}
	reset_mem_block(mem_block1);

	// Layer 17 (Convolution 512 -> 512)
	level = 12;
	for (layer_index= 0; layer_index< cshape[level][0]; layer_index++) {
		for (channel_idx = 0; channel_idx < cshape[level][1]; channel_idx++) {
			apply_convolution_3x3(mem_block2[channel_idx], wc[level][layer_index][channel_idx], mem_block1[layer_index], cur_size);
		}
		apply_bias_and_relu(mem_block1[layer_index], bc[level][layer_index], cur_size);
	}
	reset_mem_block(mem_block2);
	
	// Layer 18 (perform_maxpooling)
	for (layer_index= 0; layer_index< cshape[level][0]; layer_index++) {
		perform_maxpooling(mem_block1[layer_index], cur_size);
	}
	cur_size /= 2;
	
	// Layer 19 (Flatten)
	flatten(mem_block1, mem_block1_dense, cshape[level][0], cur_size, cur_size);
	if (only_convolution == 1) {
		return;
	}

	// Layer 20 (Dense)
	level = 0;
	dense(mem_block1_dense, wd[level], mem_block2_dense, dshape[level][0], dshape[level][1]);
	add_bias_and_relu_flatten(mem_block2_dense, bd[level], dshape[level][1], 1);
	reset_mem_block_dense(mem_block1_dense);

	// Layer 21 (Dense)
	level = 1;
	dense(mem_block2_dense, wd[level], mem_block1_dense, dshape[level][0], dshape[level][1]);
	add_bias_and_relu_flatten(mem_block1_dense, bd[level], dshape[level][1], 1);
	reset_mem_block_dense(mem_block2_dense);
	
	// Layer 22 (Dense)
	level = 2;
	dense(mem_block1_dense, wd[level], mem_block2_dense, dshape[level][0], dshape[level][1]);
	add_bias_and_relu_flatten(mem_block2_dense, bd[level], dshape[level][1], 1);
	softmax(mem_block2_dense, dshape[level][1]);
	// dump_memory_structure_dense_to_file(mem_block2_dense, dshape[level][1]);
	
	return;
}


void output_predictions(FILE *out, int only_convolution) {
	int i;
	if (only_convolution == 1) {
		for (i = 0; i < 512*7*7; i++) {
			fprintf(out, "%g ", mem_block1_dense[i]);
		}
	}
	else {
		for (i = 0; i < dshape[2][1]; i++) {
			fprintf(out, "%g ", mem_block2_dense[i]);
		}
	}
	fprintf(out, "\n");
}


char *trimwhitespace(char *str)
{
	char *end;

	// Trim leading space
	while (isspace((unsigned char)*str)) str++;

	if (*str == 0)  // All spaces?
		return str;

	// Trim trailing space
	end = str + strlen(str) - 1;
	while (end > str && isspace((unsigned char)*end)) end--;

	// Write new null terminator
	*(end + 1) = 0;

	return str;
}


int main(int argc, char *argv[]) {
	FILE *file_list, *results;
	char buf[1024];
#ifndef _WIN32
	struct timeval timeStart, timeEnd;
#else
	time_t timeStart, timeEnd;
#endif
	double deltaTime;
	char *weights_file;
	char *image_list_file;
	char *output_file;
	int lvls = -1;
	int only_convolution = 0;

/*#ifdef _OPENMP
	numthreads =1;
#endif
	if (numthreads < 1)
		numthreads = 1;
	// numthreads = 2;
	printf("Using %d threads\n", numthreads);

	if (argc != 4 && argc != 5) {
		printf("Usage: <program.exe> <weights file> <images list file> <output file> <only_convolution [optional]>\n");
		return 0;
	}*/
	weights_file = argv[1];
	image_list_file = argv[2];
	output_file = argv[3];
	if (argc == 5) {
		lvls = 13;
		only_convolution = 1;
	}

	init_memory();
	file_list = fopen(image_list_file, "r");
	if (file_list == NULL) {
		printf("Check file list location: %s", image_list_file);
		return 1;
	}
	results = fopen(output_file, "w");
	if (results == NULL) {
		printf("Couldn't open file for writing: %s", output_file);
		return 1;
	}

	gettimeofday(&timeStart, NULL);
	load_model_weights(weights_file, lvls);
	gettimeofday(&timeEnd, NULL);
	deltaTime = get_seconds(timeStart, timeEnd);
	printf("Reading weights: %.3lf sec\n", deltaTime);

	while (!feof(file_list)) {
		gettimeofday(&timeStart, NULL);
		fgets(buf, 1024, file_list);
		if (strlen(buf) == 0) {
			break;
		}
        printf("%zu\n", strlen(buf));
		load_image_data(trimwhitespace(buf));
		normalize_image();
		// dump_image();
		get_VGG16_predict(only_convolution);
		output_predictions(results, only_convolution);
		gettimeofday(&timeEnd, NULL);
		deltaTime = get_seconds(timeStart, timeEnd);
		printf("Infer image %s: %.3lf sec\n", buf, deltaTime);
	}
    


	free_memory();
	fclose(file_list);
	return 0;
}