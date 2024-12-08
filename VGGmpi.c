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
#include <mpi.h>
int rank, size; // Declare rank and size as global

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

void initialize_MPI(int *rank, int *size) {
    MPI_Comm_rank(MPI_COMM_WORLD, rank); // Get the rank of the current processor
    MPI_Comm_size(MPI_COMM_WORLD, size); // Get the total number of processors
}


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

void load_model_weights(char *file_path) {
    int file_availability = 0;
    float weight_value;
    int filter_idx, input_channel, row, col, layer_idx;
    FILE *file_ptr;

    // Step 1: Rank 0 opens the file and reads the weights
    if (rank == 0) {
        file_ptr = fopen(file_path, "r");
        if (file_ptr == NULL) {
            fprintf(stderr, "Error: Unable to open file %s\n", file_path);
            exit(1);
        }

        // Loop through convolutional layers to read weights and biases
        for (layer_idx = 0; layer_idx < 13; layer_idx++) {
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
        }

        // Mark that file is available
        file_availability = 1;

        fclose(file_ptr);
    }

    // Step 2: Broadcast the file availability flag to all processes
    MPI_Bcast(&file_availability, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Step 3: Parallelized reading of weights for each convolution and dense weight matrix
    for (layer_idx = 0; layer_idx < 13; layer_idx++) {
        // Only rank 0 does the actual reading of weights, others receive the data
        if (rank != 0) {
            // Receive weights for each layer
            for (filter_idx = 0; filter_idx < cshape[layer_idx][0]; filter_idx++) {
                for (input_channel = 0; input_channel < cshape[layer_idx][1]; input_channel++) {
                    for (row = 0; row < cshape[layer_idx][2]; row++) {
                        for (col = 0; col < cshape[layer_idx][3]; col++) {
                            MPI_Bcast(&wc[layer_idx][filter_idx][input_channel][CONV_SIZE - row - 1][CONV_SIZE - col - 1], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
                        }
                    }
                }
            }

            // Receive biases
            for (filter_idx = 0; filter_idx < cshape[layer_idx][0]; filter_idx++) {
                MPI_Bcast(&bc[layer_idx][filter_idx], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
            }
        } else {
            // Rank 0 broadcasts weights to all other ranks
            for (filter_idx = 0; filter_idx < cshape[layer_idx][0]; filter_idx++) {
                for (input_channel = 0; input_channel < cshape[layer_idx][1]; input_channel++) {
                    for (row = 0; row < cshape[layer_idx][2]; row++) {
                        for (col = 0; col < cshape[layer_idx][3]; col++) {
                            MPI_Bcast(&wc[layer_idx][filter_idx][input_channel][CONV_SIZE - row - 1][CONV_SIZE - col - 1], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
                        }
                    }
                }
            }

            // Broadcast biases
            for (filter_idx = 0; filter_idx < cshape[layer_idx][0]; filter_idx++) {
                MPI_Bcast(&bc[layer_idx][filter_idx], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
            }
        }
    }
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

void apply_convolution_3x3(float **matrix, float **kernel, float **output, int size) {
    int row, col, start_row, end_row;
    float pixel_sum;
    int chunk_size = size / size;  // Distribute rows across ranks
    int remainder = size % size;  // Handle uneven division

    // Determine the portion of rows for this rank
    start_row = rank * chunk_size + (rank < remainder ? rank : remainder);
    end_row = start_row + chunk_size + (rank < remainder);

    // Apply zero-padding to the input matrix locally
    float padded_matrix[size + 2][size + 2];
    memset(padded_matrix, 0, sizeof(padded_matrix));

    for (row = 0; row < size; row++) {
        for (col = 0; col < size; col++) {
            padded_matrix[row + 1][col + 1] = matrix[row][col];
        }
    }

    // Perform convolution for assigned rows
    for (row = start_row; row < end_row; row++) {
        for (col = 0; col < size; col++) {
            pixel_sum = 
                padded_matrix[row][col] * kernel[0][0] +
                padded_matrix[row + 1][col] * kernel[1][0] +
                padded_matrix[row + 2][col] * kernel[2][0] +
                padded_matrix[row][col + 1] * kernel[0][1] +
                padded_matrix[row + 1][col + 1] * kernel[1][1] +
                padded_matrix[row + 2][col + 1] * kernel[2][1] +
                padded_matrix[row][col + 2] * kernel[0][2] +
                padded_matrix[row + 1][col + 2] * kernel[1][2] +
                padded_matrix[row + 2][col + 2] * kernel[2][2];
            output[row][col] = pixel_sum;
        }
    }

    // Gather results from all processes into the output matrix
    MPI_Gather(&output[start_row][0], (end_row - start_row) * size, MPI_FLOAT,
               &output[0][0], (end_row - start_row) * size, MPI_FLOAT, 0, MPI_COMM_WORLD);
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


void perform_maxpooling(float **input_matrix, float **output_matrix, int input_size) {
    int row, col, start_row, end_row;
    int chunk_size = input_size / size;  // Divide rows among processes
    int remainder = input_size % size;

    // Calculate the start and end rows for each rank
    start_row = rank * chunk_size + (rank < remainder ? rank : remainder);
    end_row = start_row + chunk_size + (rank < remainder);

    // Allocate local output matrix for each process
    int local_output_size = (end_row - start_row) / 2;
    float local_output[local_output_size][input_size / 2];

    // Perform max-pooling for the assigned rows
    for (row = start_row; row < end_row; row += 2) {
        for (col = 0; col < input_size; col += 2) {
            local_output[(row - start_row) / 2][col / 2] = max_of_4(
                input_matrix[row][col],
                input_matrix[row + 1][col],
                input_matrix[row][col + 1],
                input_matrix[row + 1][col + 1]
            );
        }
    }

    // Gather results from all processes
    MPI_Gather(
        &local_output[0][0],                          // Local output buffer
        (end_row - start_row) * input_size / 4,       // Number of elements to send
        MPI_FLOAT,                                    // Data type
        &output_matrix[0][0],                         // Global output buffer
        (end_row - start_row) * input_size / 4,       // Number of elements to receive per process
        MPI_FLOAT,                                    // Data type
        0,                                            // Root process
        MPI_COMM_WORLD                                // Communicator
    );
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


void dense(float *input, float **weights, float *output, int input_size, int output_size) {
    int i, j, start_idx, end_idx;
    float *local_output = calloc(output_size, sizeof(float)); // Local output buffer

    // Divide the work among processors
    int chunk_size = output_size / size;  // Outputs per rank
    int remainder = output_size % size;

    // Determine the portion of outputs for this rank
    start_idx = rank * chunk_size + (rank < remainder ? rank : remainder);
    end_idx = start_idx + chunk_size + (rank < remainder);

    // Compute the local output values
    for (i = start_idx; i < end_idx; i++) {
        float sum = 0.0;
        for (j = 0; j < input_size; j++) {
            sum += input[j] * weights[j][i];
        }
        local_output[i] = sum;
    }

    // Synchronize and combine the results across all processors
    MPI_Allreduce(MPI_IN_PLACE, local_output, output_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    // Copy the final output back to the output array
    for (i = 0; i < output_size; i++) {
        output[i] = local_output[i];
    }

    free(local_output);
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
    int layer_index, channel_idx, level, cur_size;

    // Reset intermediate memory
    reset_mem_block(mem_block1);
    reset_mem_block(mem_block2);
    reset_mem_block_dense(mem_block1_dense);
    reset_mem_block_dense(mem_block2_dense);

    // Convolutional Layers (Parallelized)
    cur_size = SIZE;
    for (level = 0; level < 13; level++) {
        // Alternate memory blocks between mem_block1 and mem_block2
        float ***input_block = (level % 2 == 0) ? mem_block1 : mem_block2;
        float ***output_block = (level % 2 == 0) ? mem_block2 : mem_block1;

        // Parallelize convolution + ReLU
        for (layer_index = 0; layer_index < cshape[level][0]; layer_index++) {
            for (channel_idx = 0; channel_idx < cshape[level][1]; channel_idx++) {
                apply_convolution_3x3(input_block[channel_idx],
                                      wc[level][layer_index][channel_idx],
                                      output_block[layer_index],
                                      cur_size);
            }
            apply_bias_and_relu(output_block[layer_index], bc[level][layer_index], cur_size);
        }

        // Apply max-pooling if applicable
        if (level == 1 || level == 3 || level == 6 || level == 9 || level == 12) {
            float ***pooled_block = malloc(cshape[level][0] * sizeof(float **));
            for (layer_index = 0; layer_index < cshape[level][0]; layer_index++) {
                pooled_block[layer_index] = malloc((cur_size / 2) * sizeof(float *));
                for (int row = 0; row < cur_size / 2; row++) {
                    pooled_block[layer_index][row] = malloc((cur_size / 2) * sizeof(float));
                }
            }

            for (layer_index = 0; layer_index < cshape[level][0]; layer_index++) {
                perform_maxpooling(output_block[layer_index], pooled_block[layer_index], cur_size);
            }

            // Update the output_block reference to pooled_block
            for (layer_index = 0; layer_index < cshape[level][0]; layer_index++) {
                free(output_block[layer_index]);  // Free old output memory
                output_block[layer_index] = pooled_block[layer_index];
            }
            cur_size /= 2;

            // Free pooled_block pointer array
            free(pooled_block);
        }

        // Reset unused memory block
        reset_mem_block(input_block);
    }

    if (only_convolution) {
        return; // Exit if only convolution layers are needed
    }

    // Flatten the 3D output to a 1D array
    flatten(mem_block2, mem_block1_dense, cshape[12][0], cur_size, cur_size);

    // Dense Layers (Parallelized)
    for (level = 0; level < 3; level++) {
        float *input_dense = (level % 2 == 0) ? mem_block1_dense : mem_block2_dense;
        float *output_dense = (level % 2 == 0) ? mem_block2_dense : mem_block1_dense;

        // Perform dense computation with parallelized matrix multiplication
        dense(input_dense, wd[level], output_dense, dshape[level][0], dshape[level][1]);
        add_bias_and_relu_flatten(output_dense, bd[level], dshape[level][1], level < 2);

        // Reset unused memory block
        reset_mem_block_dense(input_dense);
    }

    // Apply softmax to the final output
    softmax(mem_block2_dense, dshape[2][1]);
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
    FILE *file_list = NULL, *results = NULL;
    char buf[1024];
    char *weights_file = NULL, *image_list_file = NULL, *output_file = NULL;
    int lvls = -1, only_convolution = 0;
    double start_time, end_time, deltaTime;

    // MPI Initialization
    MPI_Init(&argc, &argv);
    initialize_MPI(&rank, &size);

    if (rank == 0) {
        if (argc < 4 || argc > 5) {
            printf("Usage: %s <weights_file> <image_list_file> <output_file> [only_convolution]\n", argv[0]);
            MPI_Finalize();
            return 1;
        }

        weights_file = argv[1];
        image_list_file = argv[2];
        output_file = argv[3];
        if (argc == 5) {
            lvls = 13;
            only_convolution = 1;
        }

        file_list = fopen(image_list_file, "r");
        if (file_list == NULL) {
            printf("Check file list location: %s\n", image_list_file);
            MPI_Finalize();
            return 1;
        }

        results = fopen(output_file, "w");
        if (results == NULL) {
            printf("Couldn't open file for writing: %s\n", output_file);
            fclose(file_list);
            MPI_Finalize();
            return 1;
        }
    }

    // Broadcast parameters
    MPI_Bcast(&lvls, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&only_convolution, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Initialize memory and load model weights
    init_memory();
    if (rank == 0) {
        printf("Loading model weights...\n");
    }
    start_time = MPI_Wtime();
    load_model_weights(weights_file);
    end_time = MPI_Wtime();
    if (rank == 0) {
    }

    // Process each image
    while (1) {
        if (rank == 0) {
            if (fgets(buf, sizeof(buf), file_list) == NULL || strlen(buf) == 0) {
                break;
            }
            buf[strcspn(buf, "\n")] = 0; // Trim newline
        }

        // Broadcast image filename
        MPI_Bcast(buf, sizeof(buf), MPI_CHAR, 0, MPI_COMM_WORLD);

        if (strlen(buf) == 0) {
            break; // Exit loop when no more images
        }

        if (rank == 0) {
            printf("Processing image: %s\n", buf);
        }

        


        // Predict using VGG16
        start_time = MPI_Wtime();
       get_VGG16_predict(only_convolution);
        end_time = MPI_Wtime();
        if (rank == 0) {
            printf("Prediction completed in %.3lf seconds\n", end_time - start_time);
        }

        // Write predictions to the output file
        if (rank == 0) {
            start_time = MPI_Wtime();
            output_predictions(results, only_convolution);
            end_time = MPI_Wtime();
            printf("Predictions written in %.3lf seconds\n", end_time - start_time);
        }
    }

    // Finalize and free resources
    free_memory();
    if (rank == 0) {
        fclose(file_list);
        fclose(results);
    }

    MPI_Finalize();
    return 0;
}

