// CUDA implementation of PageRank and HITS using COO format
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>


#define MAX_ITERATIONS 100
#define EPSILON 1e-6
#define DAMPING 0.85f


// COO Matrix
typedef struct {
    int num_rows, num_cols, num_nonzeros;
    int* rows;
    int* cols;
} coo_matrix;


FILE* output_file;


__global__ void pagerank_kernel(const int* d_rows, const int* d_cols, const float* d_pr, float* d_new_pr, const int* d_out_degree, int nnz, float damping, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) {
        int src = d_rows[i];
        int dst = d_cols[i];
        if (d_out_degree[src] > 0)
            atomicAdd(&d_new_pr[dst], damping * d_pr[src] / d_out_degree[src]);
    }
}


__global__ void hits_auth_kernel(const int* d_rows, const int* d_cols, const float* d_hub, float* d_auth, int nnz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) {
        int src = d_rows[i];
        int dst = d_cols[i];
        atomicAdd(&d_auth[dst], d_hub[src]);
    }
}


__global__ void hits_hub_kernel(const int* d_rows, const int* d_cols, const float* d_auth, float* d_hub, int nnz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) {
        int src = d_rows[i];
        int dst = d_cols[i];
        atomicAdd(&d_hub[src], d_auth[dst]);
    }
}


__global__ void normalize_kernel(float* vec, int N, float norm) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && norm > 0)
        vec[i] /= norm;
}


__global__ void compute_diff_kernel(const float* a, const float* b, float* diff, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float d = fabsf(a[i] - b[i]);
        atomicAdd(diff, d);
    }
}


__global__ void reset_vector(float* vec, float val, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        vec[i] = val;
}


void read_matrix_market(const char* filename, coo_matrix* coo) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        perror("Cannot open file");
        exit(1);
    }


    while (fgetc(f) == '%') {
        while (fgetc(f) != '\n');
    }
    fseek(f, -1, SEEK_CUR);


    fscanf(f, "%d %d %d", &coo->num_rows, &coo->num_cols, &coo->num_nonzeros);
    coo->rows = (int*)malloc(coo->num_nonzeros * sizeof(int));
    coo->cols = (int*)malloc(coo->num_nonzeros * sizeof(int));


    for (int i = 0; i < coo->num_nonzeros; ++i) {
        int r, c;
        fscanf(f, "%d %d", &r, &c);
        coo->rows[i] = r - 1;
        coo->cols[i] = c - 1;
    }
    fclose(f);
}


void run_cuda_pagerank(coo_matrix* coo) {
    int N = coo->num_rows, NNZ = coo->num_nonzeros;
    float *pr, *new_pr, *d_pr, *d_new_pr, *d_diff;
    int *d_rows, *d_cols, *out_degree, *d_out_degree;


    pr = (float*)malloc(N * sizeof(float));
    new_pr = (float*)malloc(N * sizeof(float));
    out_degree = (int*)calloc(N, sizeof(int));


    for (int i = 0; i < NNZ; ++i)
        out_degree[coo->rows[i]]++;


    for (int i = 0; i < N; ++i)
        pr[i] = 1.0f / N;


    cudaMalloc(&d_rows, NNZ * sizeof(int));
    cudaMalloc(&d_cols, NNZ * sizeof(int));
    cudaMalloc(&d_out_degree, N * sizeof(int));
    cudaMalloc(&d_pr, N * sizeof(float));
    cudaMalloc(&d_new_pr, N * sizeof(float));
    cudaMalloc(&d_diff, sizeof(float));


    cudaMemcpy(d_rows, coo->rows, NNZ * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, coo->cols, NNZ * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_degree, out_degree, N * sizeof(int), cudaMemcpyHostToDevice);


    int blockSize = 256;
    int numBlocks_N = (N + blockSize - 1) / blockSize;
    int numBlocks_NNZ = (NNZ + blockSize - 1) / blockSize;


    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent);


    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        float host_diff = 0.0f;
        cudaMemcpy(d_diff, &host_diff, sizeof(float), cudaMemcpyHostToDevice);


        cudaMemcpy(d_pr, pr, N * sizeof(float), cudaMemcpyHostToDevice);
        reset_vector<<<numBlocks_N, blockSize>>>(d_new_pr, (1.0f - DAMPING) / N, N);


        pagerank_kernel<<<numBlocks_NNZ, blockSize>>>(d_rows, d_cols, d_pr, d_new_pr, d_out_degree, NNZ, DAMPING, N);


        compute_diff_kernel<<<numBlocks_N, blockSize>>>(d_new_pr, d_pr, d_diff, N);


        cudaMemcpy(&host_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
        if (host_diff < EPSILON) break;


        cudaMemcpy(pr, d_new_pr, N * sizeof(float), cudaMemcpyDeviceToHost);
    }


    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
    fprintf(output_file, "PageRank: %.4f", milliseconds / 1000.0);


    fflush(output_file);


    free(pr); free(new_pr); free(out_degree);
    cudaFree(d_rows); cudaFree(d_cols); cudaFree(d_pr);
    cudaFree(d_new_pr); cudaFree(d_out_degree); cudaFree(d_diff);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}


void run_cuda_hits(coo_matrix* coo) {
    using namespace std::chrono;
    auto start = high_resolution_clock::now();


    int N = coo->num_rows, NNZ = coo->num_nonzeros;
    float *auth, *hub, *d_auth, *d_hub, *d_diff;


    auth = (float*)calloc(N, sizeof(float));
    hub = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) hub[i] = 1.0;


    int *d_rows, *d_cols;
    cudaMalloc(&d_rows, NNZ * sizeof(int));
    cudaMalloc(&d_cols, NNZ * sizeof(int));
    cudaMalloc(&d_auth, N * sizeof(float));
    cudaMalloc(&d_hub, N * sizeof(float));
    cudaMalloc(&d_diff, sizeof(float));


    cudaMemcpy(d_rows, coo->rows, NNZ * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, coo->cols, NNZ * sizeof(int), cudaMemcpyHostToDevice);


    int blockSize = 256;
    int numBlocks_N = (N + blockSize - 1) / blockSize;
    int numBlocks_NNZ = (NNZ + blockSize - 1) / blockSize;


    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        cudaMemcpy(d_hub, hub, N * sizeof(float), cudaMemcpyHostToDevice);
        reset_vector<<<numBlocks_N, blockSize>>>(d_auth, 0.0f, N);
        hits_auth_kernel<<<numBlocks_NNZ, blockSize>>>(d_rows, d_cols, d_hub, d_auth, NNZ);


        float norm_auth = 0;
        cudaMemcpy(auth, d_auth, N * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; ++i) norm_auth += auth[i] * auth[i];
        norm_auth = sqrtf(norm_auth);
        normalize_kernel<<<numBlocks_N, blockSize>>>(d_auth, N, norm_auth);


        reset_vector<<<numBlocks_N, blockSize>>>(d_hub, 0.0f, N);
        hits_hub_kernel<<<numBlocks_NNZ, blockSize>>>(d_rows, d_cols, d_auth, d_hub, NNZ);


        float norm_hub = 0;
        cudaMemcpy(hub, d_hub, N * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; ++i) norm_hub += hub[i] * hub[i];
        norm_hub = sqrtf(norm_hub);
        normalize_kernel<<<numBlocks_N, blockSize>>>(d_hub, N, norm_hub);


        float host_diff = 0.0f;
        cudaMemcpy(d_diff, &host_diff, sizeof(float), cudaMemcpyHostToDevice);
        compute_diff_kernel<<<numBlocks_N, blockSize>>>(d_auth, auth, d_diff, N);
        compute_diff_kernel<<<numBlocks_N, blockSize>>>(d_hub, hub, d_diff, N);
        cudaMemcpy(&host_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
        if (host_diff < EPSILON) break;


        cudaMemcpy(auth, d_auth, N * sizeof(float), cudaMemcpyDeviceToHost);
    }


    auto end = high_resolution_clock::now();
    double duration = duration_cast<milliseconds>(end - start).count() / 1000.0;
    fprintf(output_file, "HITS: %.4f", duration);
    fflush(output_file);


    free(auth); free(hub);
    cudaFree(d_rows); cudaFree(d_cols); cudaFree(d_auth); cudaFree(d_hub); cudaFree(d_diff);
}


int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <matrix_file.mtx>\n", argv[0]);
        return 1;
    }


    output_file = fopen("cuda_result.txt", "w");
    if (!output_file) {
        perror("Failed to open cuda_result.txt");
        return 1;
    }


    coo_matrix coo;
    read_matrix_market(argv[1], &coo);


    run_cuda_hits(&coo);
    run_cuda_pagerank(&coo);


    fclose(output_file);
    free(coo.rows);
    free(coo.cols);
    return 0;
}
