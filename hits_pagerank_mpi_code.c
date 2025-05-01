#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "cmdline.h"
#include "input.h"
#include "config.h"
#include "timer.h"
#include "formats.h"

#define MAX_ITERATIONS 100
#define EPSILON 1e-6

void normalize_vector(float* vec, int size, MPI_Comm comm) {
    float local_sum = 0.0, global_sum = 0.0;
    for (int i = 0; i < size; ++i)
        local_sum += vec[i] * vec[i];
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, comm);
    global_sum = sqrt(global_sum);
    if (global_sum == 0) return;
    for (int i = 0; i < size; ++i)
        vec[i] /= global_sum;
}

void run_pagerank(coo_matrix *coo, int max_iter, float damping, float* pr, int rank, int size, MPI_Comm comm) {
    int N = coo->num_rows;
    float* new_pr = (float*)calloc(N, sizeof(float));
    int* out_degree = (int*)calloc(N, sizeof(int));

    for (int i = 0; i < coo->num_nonzeros; ++i)
        out_degree[coo->rows[i]]++;

    for (int i = 0; i < N; ++i)
        pr[i] = 1.0 / N;

    int chunk = N / size;
    int start = rank * chunk;
    int end = (rank == size - 1) ? N : start + chunk;

    for (int iter = 0; iter < max_iter; ++iter) {
        for (int i = start; i < end; ++i)
            new_pr[i] = (1.0 - damping) / N;

        for (int i = 0; i < coo->num_nonzeros; ++i) {
            int src = coo->rows[i];
            int dst = coo->cols[i];
            if (dst >= start && dst < end && out_degree[src] > 0)
                new_pr[dst] += damping * pr[src] / out_degree[src];
        }

        float* global_pr = (float*)calloc(N, sizeof(float));
        MPI_Allreduce(new_pr, global_pr, N, MPI_FLOAT, MPI_SUM, comm);

        float local_diff = 0.0, global_diff = 0.0;
        for (int i = 0; i < N; ++i)
            local_diff += fabs(global_pr[i] - pr[i]);
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_FLOAT, MPI_SUM, comm);

        if (rank == 0)
            printf("PageRank Iteration %d, diff = %.6f\n", iter + 1, global_diff);

        if (global_diff < EPSILON) {
            free(global_pr);
            break;
        }

        for (int i = 0; i < N; ++i)
            pr[i] = global_pr[i];
        free(global_pr);
    }

    free(new_pr);
    free(out_degree);
}

void run_hits(coo_matrix *coo, int max_iter, float* auth, float* hub, int rank, int size, MPI_Comm comm) {
    int N = coo->num_rows;
    float* new_auth = (float*)calloc(N, sizeof(float));
    float* new_hub = (float*)calloc(N, sizeof(float));

    for (int i = 0; i < N; ++i)
        hub[i] = 1.0;

    int chunk = N / size;
    int start = rank * chunk;
    int end = (rank == size - 1) ? N : start + chunk;

    for (int iter = 0; iter < max_iter; ++iter) {
        for (int i = start; i < end; ++i)
            new_auth[i] = 0.0;

        for (int i = 0; i < coo->num_nonzeros; ++i)
            if (coo->cols[i] >= start && coo->cols[i] < end)
                new_auth[coo->cols[i]] += hub[coo->rows[i]];

        float* global_auth = (float*)calloc(N, sizeof(float));
        MPI_Allreduce(new_auth, global_auth, N, MPI_FLOAT, MPI_SUM, comm);
        normalize_vector(global_auth, N, comm);

        for (int i = start; i < end; ++i)
            new_hub[i] = 0.0;

        for (int i = 0; i < coo->num_nonzeros; ++i)
            if (coo->rows[i] >= start && coo->rows[i] < end)
                new_hub[coo->rows[i]] += global_auth[coo->cols[i]];

        float* global_hub = (float*)calloc(N, sizeof(float));
        MPI_Allreduce(new_hub, global_hub, N, MPI_FLOAT, MPI_SUM, comm);
        normalize_vector(global_hub, N, comm);

        float local_diff = 0.0, global_diff = 0.0;
        for (int i = 0; i < N; ++i)
            local_diff += fabs(global_auth[i] - auth[i]) + fabs(global_hub[i] - hub[i]);
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_FLOAT, MPI_SUM, comm);

        if (rank == 0)
            printf("HITS Iteration %d, diff = %.6f\n", iter + 1, global_diff);

        if (global_diff < EPSILON) {
            free(global_auth);
            free(global_hub);
            break;
        }

        for (int i = 0; i < N; ++i) {
            auth[i] = global_auth[i];
            hub[i] = global_hub[i];
        }

        free(global_auth);
        free(global_hub);
    }

    free(new_auth);
    free(new_hub);
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) printf("Usage: %s matrix_file.mtx\n", argv[0]);
        MPI_Finalize();
        return -1;
    }

    coo_matrix coo;
    if (rank == 0) read_coo_matrix(&coo, argv[1]);
    MPI_Bcast(&coo.num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&coo.num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&coo.num_nonzeros, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        coo.rows = malloc(coo.num_nonzeros * sizeof(int));
        coo.cols = malloc(coo.num_nonzeros * sizeof(int));
        coo.vals = malloc(coo.num_nonzeros * sizeof(float));
    }

    MPI_Bcast(coo.rows, coo.num_nonzeros, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(coo.cols, coo.num_nonzeros, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(coo.vals, coo.num_nonzeros, MPI_FLOAT, 0, MPI_COMM_WORLD);

    float* pr = (float*)malloc(coo.num_rows * sizeof(float));
    float* auth = (float*)calloc(coo.num_rows, sizeof(float));
    float* hub = (float*)calloc(coo.num_rows, sizeof(float));

    FILE* out = NULL;
    if (rank == 0) {
        out = fopen("output_mpi_hits_pagerank.txt", "w");
        if (!out) {
            fprintf(stderr, "Failed to open output file.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    timer t_hits, t_pr;
    double hits_time = 0.0, pr_time = 0.0;

    if (rank == 0) timer_start(&t_hits);
    run_hits(&coo, MAX_ITERATIONS, auth, hub, rank, size, MPI_COMM_WORLD);
    if (rank == 0) {
        hits_time = seconds_elapsed(&t_hits);
        fprintf(out, "[HITS] Execution Time: %.6f seconds\n", hits_time);
    }

    if (rank == 0) timer_start(&t_pr);
    run_pagerank(&coo, MAX_ITERATIONS, 0.85f, pr, rank, size, MPI_COMM_WORLD);
    if (rank == 0) {
        pr_time = seconds_elapsed(&t_pr);
        fprintf(out, "[PageRank] Execution Time: %.6f seconds\n", pr_time);
    }

    if (rank == 0) {
        fprintf(out, "\nFinal Authority Scores:\n");
        for (int i = 0; i < coo.num_rows; ++i)
            fprintf(out, "Node %d -> Auth: %.10f\n", i, auth[i]);

        fprintf(out, "\nFinal Hub Scores:\n");
        for (int i = 0; i < coo.num_rows; ++i)
            fprintf(out, "Node %d -> Hub: %.10f\n", i, hub[i]);

        fprintf(out, "\nFinal PageRank Scores:\n");
        for (int i = 0; i < coo.num_rows; ++i)
            fprintf(out, "Node %d -> PR: %.10f\n", i, pr[i]);

        fclose(out);
    }

    free(pr);
    free(auth);
    free(hub);
    delete_coo_matrix(&coo);
    MPI_Finalize();
    return 0;
}

