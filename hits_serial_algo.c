#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cmdline.h"
#include "input.h"
#include "config.h"
#include "timer.h"
#include "formats.h"

#define MAX_ITERATIONS 100
#define EPSILON 1e-6

void usage(int argc, char** argv)
{
    printf("Usage: %s [my_matrix.mtx]\n", argv[0]);
    printf("Note: my_matrix.mtx must be a sparse matrix in MatrixMarket format (COO).\n");
}

// Normalize a vector
void normalize_vector(float* vec, int size) {
    float norm = 0.0;
    for (int i = 0; i < size; ++i)
        norm += vec[i] * vec[i];
    norm = sqrt(norm);
    if (norm == 0) return;
    for (int i = 0; i < size; ++i)
        vec[i] /= norm;
}

void run_pagerank(coo_matrix *coo, int max_iter, float damping, FILE* out) {
    int N = coo->num_rows;
    float* pr = (float*)malloc(N * sizeof(float));
    float* new_pr = (float*)malloc(N * sizeof(float));

    // Out-degree count
    int* out_degree = (int*)calloc(N, sizeof(int));
    for (int i = 0; i < coo->num_nonzeros; ++i)
        out_degree[coo->rows[i]]++;

    // Initialize scores
    for (int i = 0; i < N; ++i)
        pr[i] = 1.0 / N;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Reset new scores
        for (int i = 0; i < N; ++i)
            new_pr[i] = (1.0 - damping) / N;

        // Spread PageRank through in-links
        for (int i = 0; i < coo->num_nonzeros; ++i) {
            int src = coo->rows[i];
            int dst = coo->cols[i];
            if (out_degree[src] > 0)
                new_pr[dst] += damping * pr[src] / out_degree[src];
        }

        // Check convergence
        float diff = 0.0;
        for (int i = 0; i < N; ++i)
            diff += fabs(new_pr[i] - pr[i]);

        fprintf(out, "PageRank Iteration %d, diff = %.6f\n", iter + 1, diff);

        if (diff < EPSILON)
            break;

        // Swap
        float* temp = pr;
        pr = new_pr;
        new_pr = temp;
    }

    // Print final PR
    fprintf(out, "\nFinal PageRank Scores:\n");
    for (int i = 0; i < N; ++i)
        fprintf(out, "Node %d -> PR: %.10f\n", i, pr[i]);

    free(pr);
    free(new_pr);
    free(out_degree);
}

// HITS Algorithm using COO format
void run_hits(coo_matrix *coo, int max_iter, FILE* out) {
    int N = coo->num_rows;

    float* auth = (float*)calloc(N, sizeof(float));
    float* hub = (float*)calloc(N, sizeof(float));
    float* new_auth = (float*)calloc(N, sizeof(float));
    float* new_hub = (float*)calloc(N, sizeof(float));

    // Init hubs to 1
    for (int i = 0; i < N; ++i)
        hub[i] = 1.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Reset vectors
        for (int i = 0; i < N; ++i) {
            new_auth[i] = 0.0;
            new_hub[i] = 0.0;
        }

        // Update authority scores: A = A^T * H
        for (int i = 0; i < coo->num_nonzeros; ++i)
            new_auth[coo->cols[i]] += hub[coo->rows[i]];

        // Normalize authority
        normalize_vector(new_auth, N);

        // Update hub scores: H = A * A
        for (int i = 0; i < coo->num_nonzeros; ++i)
            new_hub[coo->rows[i]] += new_auth[coo->cols[i]];

        // Normalize hub
        normalize_vector(new_hub, N);

        // Convergence check (optional)
        float diff = 0.0;
        for (int i = 0; i < N; ++i)
            diff += fabs(new_auth[i] - auth[i]) + fabs(new_hub[i] - hub[i]);

        fprintf(out, "Iteration %d, diff = %.6f\n", iter + 1, diff);
        if (diff < EPSILON)
            break;

        // Swap pointers
        float* temp = auth;
        auth = new_auth;
        new_auth = temp;

        temp = hub;
        hub = new_hub;
        new_hub = temp;
    }

    // Output final scores
    fprintf(out, "\nFinal Authority Scores (full precision):\n");
    for (int i = 0; i < N; ++i)
        fprintf(out, "Node %d -> Auth: %.10f\n", i, auth[i]);
    
    fprintf(out, "\nFinal Hub Scores (full precision):\n");
    for (int i = 0; i < N; ++i)
        fprintf(out, "Node %d -> Hub: %.10f\n", i, hub[i]);

    free(auth);
    free(hub);
    free(new_auth);
    free(new_hub);
}

int main(int argc, char** argv)
{
    if (get_arg(argc, argv, "help") != NULL) {
        usage(argc, argv);
        return 0;
    }

    if (argc < 2) {
        printf("Give a MatrixMarket file.\n");
        return -1;
    }

    char * mm_filename = argv[1];
    coo_matrix coo;
    read_coo_matrix(&coo, mm_filename);

    printf("\nfile=%s rows=%d cols=%d nonzeros=%d\n", mm_filename, coo.num_rows, coo.num_cols, coo.num_nonzeros);
    fflush(stdout);

    FILE* out = fopen("output_hits_pagerank.txt", "w");
    if (!out) {
        perror("Error opening output file");
        return -1;
    }

    // Run HITS Algorithm
    timer t_hits;
    timer_start(&t_hits);
    run_hits(&coo, MAX_ITERATIONS, out);
    double hits_time = seconds_elapsed(&t_hits);
    fprintf(out, "\n[HITS] Execution Time: %.6f seconds\n", hits_time);

    // Run PageRank with timing
    timer t_pr;
    timer_start(&t_pr);
    run_pagerank(&coo, MAX_ITERATIONS, 0.85f, out);
    double pr_time = seconds_elapsed(&t_pr);
    fprintf(out, "\n[PageRank] Execution Time: %.6f seconds\n", pr_time);

    fclose(out);
    delete_coo_matrix(&coo);
    return 0;
}
