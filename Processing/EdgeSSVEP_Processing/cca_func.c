/**
 * @file cca_processing.c
 * @brief Canonical Correlation Analysis (CCA) for SSVEP Detection
 * @author manhdatbn93@gmail.com
 * @date 2025-25-24
 * @version 1.0
 *
 * @note Implements CCA-based frequency detection for SSVEP BCI systems
 *       Uses optimized matrix operations and power iteration for eigenvalue computation
 */

#define _CRT_SECURE_NO_WARNINGS
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "cca_coeffs.h"
#include "cca_func.h"
#include "filter_func.h"

/* Constants */
#define REGULARIZATION 1e-6f  // Regularization term for covariance matrices
#define TOL 1e-6f             // Convergence tolerance for power iteration
#define MAX_ITER 100          // Maximum power iteration attempts

/* Global Buffers (reused for memory efficiency) */
static float last_samples[CCA_NUM_CHANNELS][CCA_SAMPLES];
static float Y[CCA_NUM_FEATURES][CCA_SAMPLES];
static float X_T[CCA_SAMPLES][CCA_NUM_CHANNELS];
static float Cxx[CCA_NUM_CHANNELS][CCA_NUM_CHANNELS];
static float Y_T[CCA_SAMPLES][CCA_NUM_FEATURES];
static float Cyy[CCA_NUM_FEATURES][CCA_NUM_FEATURES];
static float Cxy[CCA_NUM_CHANNELS][CCA_NUM_FEATURES];
static float inv_Cxx[CCA_NUM_CHANNELS][CCA_NUM_CHANNELS];
static float inv_Cyy[CCA_NUM_FEATURES][CCA_NUM_FEATURES];
static float Cxy_T[CCA_NUM_FEATURES][CCA_NUM_CHANNELS];
static float M[CCA_NUM_CHANNELS][CCA_NUM_CHANNELS];
static float aug[CCA_NUM_CHANNELS][2 * CCA_NUM_CHANNELS];
static float eigenvectors[CCA_NUM_CHANNELS];
static double data_filtered[PATTERN_NUM_SAMPLES];

/* Statistics */
static int pattern_trials = 0;
static int valid_trials = 0;
static float correct_trials = 0;

/**
 * @brief Matrix-vector multiplication (optimized for square matrices)
 * @param A Input matrix (size x size)
 * @param x Input vector (size)
 * @param size Dimension of matrix/vector
 * @param B Output vector (size)
 */
static void matrix_vector_multiply(float* A, float* x, int size, float* B)
{
    for (int i = 0; i < size; i++) 
    {
        B[i] = 0;
        for (int j = 0; j < size; j++) 
        {
            B[i] += A[i * size + j] * x[j];
        }
    }
}

/**
 * @brief Normalize vector to unit length
 * @param x Vector to normalize (modified in-place)
 * @param size Vector length
 */
static void vector_normalize(float* x, int size)
{
    float norm = 0;
    for (int i = 0; i < size; i++) 
    {
        norm += x[i] * x[i];
    }
    norm = sqrtf(norm);
    if (norm > 0) {
        for (int i = 0; i < size; i++) 
        {
            x[i] /= norm;
        }
    }
}

/**
 * @brief Power iteration method for dominant eigenvalue/vector
 * @param A Input matrix (size x size)
 * @param size Matrix dimension
 * @param eigenvector Output eigenvector (size)
 * @return Dominant eigenvalue
 */
static float power_iteration(float* A, int size, float* eigenvector)
{
    float b_k[CCA_NUM_CHANNELS] = { 1.0f };
    float b_k1[CCA_NUM_CHANNELS];
    float eigenvalue = 0.0f;
    float diff = 0;

    for (int iter = 0; iter < MAX_ITER; iter++) 
    {
        matrix_vector_multiply(A, b_k, size, b_k1);
        vector_normalize(b_k1, size);

        /* Check convergence */
        for (int i = 0; i < size; i++) 
        {
            diff += fabsf(b_k1[i] - b_k[i]);
        }
        if (diff < TOL) break;

        memcpy(b_k, b_k1, size * sizeof(float));
    }

    /* Compute Rayleigh quotient for eigenvalue */
    matrix_vector_multiply(A, b_k, size, b_k1);
    for (int i = 0; i < size; i++) 
    {
        eigenvalue += b_k[i] * b_k1[i];
    }
    memcpy(eigenvector, b_k, size * sizeof(float));

    return eigenvalue;
}

/**
 * @brief Center matrix rows (remove DC component)
 * @param X Matrix to center (rows x cols)
 * @param rows Number of rows
 * @param cols Number of columns
 */
static void matrix_center(float* X, int rows, int cols)
{
    for (int i = 0; i < rows; i++) 
    {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) 
        {
            sum += X[i * cols + j];
        }
        float mean = sum / cols;
        for (int j = 0; j < cols; j++) 
        {
            X[i * cols + j] -= mean;
        }
    }
}

// Compute the covariance matrix
void matrix_compute_covariance(float* X, int rows, int cols, int num_samples)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            X[i * cols + j] /= (num_samples - 1);
        }
    }
}

/**
 * @brief Matrix multiplication C = A * B
 * @param A First matrix (m x n)
 * @param B Second matrix (n x p)
 * @param C Output matrix (m x p)
 * @param m Rows in A
 * @param n Columns in A / Rows in B
 * @param p Columns in B
 */
static void matrix_mult(float* A, float* B, float* C, int m, int n, int p)
{
    memset(C, 0, m * p * sizeof(float));
    for (int i = 0; i < m; i++) 
    {
        for (int k = 0; k < n; k++) 
        {
            float a = A[i * n + k];
            for (int j = 0; j < p; j++) 
            {
                C[i * p + j] += a * B[k * p + j];
            }
        }
    }
}

/**
 * @brief Matrix transposition
 * @param A Input matrix (rows x cols)
 * @param rows Number of rows
 * @param cols Number of columns
 * @param A_T Output transposed matrix (cols x rows)
 */
static void matrix_transpose(float* A, int rows, int cols, float* A_T)
{
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            A_T[j * rows + i] = A[i * cols + j];
        }
    }
}

/**
 * @brief Invert a square matrix using Gauss-Jordan elimination with partial pivoting
 *
 * @param A Input matrix (n x n) stored in row-major order (A[n][n])
 * @param n Dimension of the matrix
 * @param inv Output inverse matrix (n x n)
 * @return int 0 if successful, -1 if matrix is singular
 *
 */
int matrix_inverse(float* A, int n, float* inv)
{
    memset(aug, 0, sizeof(aug));

    // Augment with identity matrix
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            aug[i][j] = A[i * n + j];
        }

        aug[i][n + i] = 1.0;
    }

    // Gauss-Jordan elimination
    for (int i = 0; i < n; i++)
    {
        // Pivot selection
        int max_row = i;
        for (int k = i; k < n; k++)
        {
            if (fabs(aug[k][i]) > fabs(aug[max_row][i]))
            {
                max_row = k;
            }
        }

        if (aug[max_row][i] == 0.0)
        {
            return -1; // Singular matrix
        }

        // Swap rows
        if (max_row != i)
        {
            for (int j = 0; j < 2 * n; j++)
            {
                float tmp = aug[i][j];
                aug[i][j] = aug[max_row][j];
                aug[max_row][j] = tmp;
            }
        }

        // Normalize pivot row
        float pivot = aug[i][i];
        for (int j = i; j < 2 * n; j++)
        {
            aug[i][j] /= pivot;
        }

        // Eliminate other rows
        for (int k = 0; k < n; k++)
        {
            if (k != i)
            {
                float factor = aug[k][i];
                for (int j = i; j < 2 * n; j++)
                {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }

    // Extract inverse matrix
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            inv[i * n + j] = aug[i][j + n];
        }
    }

    return 0;
}

void print_matrix(float* A, int n, int times)
{
    printf("A [%2d]= \r\n", times);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%2.20f ", A[i * n + j]);
        }
        printf("\r\n");
    }
}

/**
 * @brief Save filtered data array to text file
 * @param filename Output file path
 * @param data Array of filtered data
 * @param num_samples Number of samples to save
 * @return 0 on success, -1 on failure
 */
int save_filtered_data_to_file(const char* filename, double* data, int num_samples)
{
    FILE* file = fopen(filename, "w");
    if (file == NULL)
    {
        printf("Error: Unable to open file %s for writing\n", filename);
        return -1;
    }

    for (int i = 0; i < num_samples; i++)
    {
        fprintf(file, "%.6f\n", data[i]);
    }

    fclose(file);
    return 0;
}
/**
 * @brief CCA-based SSVEP frequency classification
 * @param eeg_data Input EEG data [channels][samples]
 * @param num_samples Number of valid samples
 * @return 1-based frequency index (0 if no detection)
 */
int cca_ssvep_classification(float eeg_data[PATTERN_NUM_CHANNELS][PATTERN_NUM_SAMPLES], int num_samples)
{
    float max_corr = 0.0f;
    int best_freq_idx = -1;
    float temp1[CCA_NUM_CHANNELS][CCA_NUM_FEATURES];
    float temp2[CCA_NUM_CHANNELS][CCA_NUM_FEATURES];
    float max_eigen = 0.0;
    float canonical_corr = 0;
    /* 1. Preprocess EEG data */
#ifdef USE_EEG_PATTERN
    for (int ch = 0; ch < PATTERN_NUM_CHANNELS; ch++)
    {
        for (int s = 0; s < num_samples; s++)
        {
            data_filtered[s] = eeg_data[ch][s];
        }

        apply_filtfilt(data_filtered, num_samples);
		//save_filtered_data_to_file("filtered_data.txt", data_filtered, num_samples);

        if (num_samples > CCA_SAMPLES)
        {
            for (int i = 0; i < CCA_SAMPLES; i++)
            {

                last_samples[ch][i] = (float)data_filtered[num_samples - CCA_SAMPLES + i];
            }
        }
    }   
#endif

    /* 2. Prepare EEG data matrix */
    matrix_center(&last_samples[0][0], CCA_NUM_CHANNELS, CCA_SAMPLES);

    /* 3. Compute covariance matrices */
    matrix_transpose(&last_samples[0][0], CCA_NUM_CHANNELS, CCA_SAMPLES, &X_T[0][0]);
    matrix_mult(&last_samples[0][0], &X_T[0][0], &Cxx[0][0], CCA_NUM_CHANNELS, CCA_SAMPLES, CCA_NUM_CHANNELS);
    matrix_compute_covariance(&Cxx[0][0], CCA_NUM_CHANNELS, CCA_NUM_CHANNELS, CCA_SAMPLES);

    /* Add regularization to Cxx */
    for (int i = 0; i < CCA_NUM_CHANNELS; i++)
    {
        Cxx[i][i] += REGULARIZATION;
    }

    /* 4. Test each reference frequency */
    //printf("\r\nCCA Freq value: ");
    for (int freq_idx = 0; freq_idx < CCA_NUM_FREQS; freq_idx++)
    {
        /* Load reference signals */
        for (int i = 0; i < CCA_NUM_FEATURES; i++)
        {
            for (int j = 0; j < CCA_SAMPLES; j++)
            {
                Y[i][j] = cca_reference_signals[freq_idx][i][j];
            }
        }

        /* Center reference signals */ 
        matrix_center(&Y[0][0], CCA_NUM_FEATURES, CCA_SAMPLES);
        matrix_transpose(&Y[0][0], CCA_NUM_FEATURES, CCA_SAMPLES, &Y_T[0][0]);

        /* Compute cross-covariance matrices */
        matrix_mult(&Y[0][0], &Y_T[0][0], &Cyy[0][0], CCA_NUM_FEATURES, CCA_SAMPLES, CCA_NUM_FEATURES);
        matrix_compute_covariance(&Cyy[0][0], CCA_NUM_FEATURES, CCA_NUM_FEATURES, CCA_SAMPLES);

        matrix_mult(&last_samples[0][0], &Y_T[0][0], &Cxy[0][0], CCA_NUM_CHANNELS, CCA_SAMPLES, CCA_NUM_FEATURES);
        matrix_compute_covariance(&Cxy[0][0], CCA_NUM_CHANNELS, CCA_NUM_FEATURES, CCA_SAMPLES);

        /* Invert Cxx and Cyy */
        if (matrix_inverse(&Cxx[0][0], CCA_NUM_CHANNELS, &inv_Cxx[0][0]))
        {
            printf("Cxx inversion failed for frequency %d\n", freq_idx);
            continue;
        }
        if (matrix_inverse(&Cyy[0][0], CCA_NUM_FEATURES, &inv_Cyy[0][0]))
        {
            printf("Cyy inversion failed for frequency %d\n", freq_idx);
            continue;
        }

        /* Compute M = inv_Cxx * Cxy * inv_Cyy * Cxy ^ T */
        matrix_mult(&inv_Cxx[0][0], &Cxy[0][0], &temp1[0][0], CCA_NUM_CHANNELS, CCA_NUM_CHANNELS, CCA_NUM_FEATURES);
        matrix_mult(&temp1[0][0], &inv_Cyy[0][0], &temp2[0][0], CCA_NUM_CHANNELS, CCA_NUM_FEATURES, CCA_NUM_FEATURES);

        matrix_transpose(&Cxy[0][0], CCA_NUM_CHANNELS, CCA_NUM_FEATURES, &Cxy_T[0][0]);
        matrix_mult(&temp2[0][0], &Cxy_T[0][0], &M[0][0], CCA_NUM_CHANNELS, CCA_NUM_FEATURES, CCA_NUM_CHANNELS);

        /* Find maximum eigenvalue(canonical correlation) */
        max_eigen = power_iteration(&M[0][0], CCA_NUM_CHANNELS, &eigenvectors[0]);
        if (max_eigen > 0)
        {
            canonical_corr = sqrt(max_eigen);
        }
        //printf("%.4f ", canonical_corr);
        if (canonical_corr > max_corr)
        {
            max_corr = canonical_corr;
            best_freq_idx = freq_idx;
        }
    }
    return (best_freq_idx + 1);
}