/**
 * @file filter_func.c
 * @brief Zero-phase digital filtering implementation
 * @author manhdatbn93@gmail.com
 * @date 2025-05-24
 * @version 1.0
 *
 * @note Implements forward-backward filtering (filtfilt) using:
 *       - 3rd-order Butterworth bandpass coefficients (2-45Hz)
 *       - 500Hz sampling rate
 *       - Zero-phase distortion
 */

#define _CRT_SECURE_NO_WARNINGS
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include "filter_coeffs.h"
#include "filter_func.h"

/* Signal buffers */
static double extended_data[MAX_VALUES + 2 * PADLEN];  ///< Padded input signal (input + mirrored extensions)
static double filter_temp[MAX_VALUES];               ///< Intermediate filtered signal storage
static double filter_out[MAX_VALUES];                ///< Final output buffer

/**
 * @brief Apply mirror padding with edge correction to input signal
 *
 * @param[in] x Input signal array (length samples)
 * @param[in] length Number of samples in input signal
 * @param[in] padlen Number of samples to pad at each end (must be <= length - 1)
 * @param[out] ext Output extended signal (length + 2*padlen samples)
 *
 * @details Implements symmetric padding with edge correction using:
 * - Left padding:  ext[i] = 2*x[0] - x[padlen - i]          (0 ≤ i < padlen)
 * - Right padding: ext[length+padlen+i] = 2*x[length-1] - x[length-2-i]  (0 ≤ i < padlen)
 *
 * @note The padding formula:
 * - Creates a mirrored version of the signal at each end
 * - Uses edge correction (2*x[edge] - x[n]) to maintain continuity
 * - Ensures smooth transitions at boundaries to reduce filtering artifacts
 *
 * @warning Behavior is undefined if padlen > length - 1
 * @warning The output array ext must have capacity for length + 2*padlen samples
 */
static void pad_signal(double* x, int length, int padlen, double* ext)
{
    for (int i = 0; i < padlen; i++)
    {
        ext[i] = 2 * x[0] - x[padlen - i];
        ext[length + padlen + i] = 2 * x[length - 1] - x[length - 2 - i];
    }
    for (int i = 0; i < length; i++)
    {
        ext[padlen + i] = x[i];
    }
}

/**
 * @brief Apply IIR filter using Direct Form II Transposed structure
 *
 * @param[in]  input   Input signal array (length samples)
 * @param[out] output  Filtered output array (length samples)
 * @param[in]  length  Number of samples to process
 * @param[in]  zi      Initial filter state vector (FILTER_ORDER-1 samples)
 * 
 */
static void lfilter(double* input, double* output, int length, double* zi)
{
    double d[FILTER_ORDER - 1]; // State variables
    for (int i = 0; i < FILTER_ORDER - 1; i++)
    {
        d[i] = zi[i];
    }

    for (int n = 0; n < length; n++)
    {
        double temp = b_coeffs[0] * input[n] + d[0];
        for (int i = 1; i < FILTER_ORDER - 1; i++)
        {
            d[i - 1] = b_coeffs[i] * input[n] + d[i] - a_coeffs[i] * temp;
        }
        d[FILTER_ORDER - 2] = b_coeffs[FILTER_ORDER - 1] * input[n] - a_coeffs[FILTER_ORDER - 1] * temp;
        output[n] = temp;
    }
}

/**
 * @brief Reverse a signal array in-place
 *
 * @param[in,out] x      Input/output signal array (length samples)
 * @param[in]      length Number of samples in the array
 * 
 */
static void reverse_signal(double* x, int length)
{
    for (int i = 0; i < length / 2; i++)
    {
        double temp = x[i];
        x[i] = x[length - 1 - i];
        x[length - 1 - i] = temp;
    }
}

/**
 * @brief Apply zero-phase digital filtering using forward-backward processing
 *
 * @param[in,out] data Input/output signal array (modified in-place)
 * @param[in] length Number of samples in the signal (must be <= MAX_VALUES)
 * 
 */
void apply_filtfilt(double* data, int length)
{
    // Pad the signal
    pad_signal(data, length, PADLEN, extended_data);

    // Calculate initial conditions
    double x0 = extended_data[0];
    double zi_new[FILTER_ORDER - 1];

    for (int i = 0; i < FILTER_ORDER - 1; i++)
    {
        zi_new[i] = zi_coeffs[i] * x0;
    }

    // Forward filter
    lfilter(extended_data, filter_temp, length + 2 * PADLEN, zi_new);

    // Reverse the signal
    reverse_signal(filter_temp, length + 2 * PADLEN);

    double y0 = filter_temp[0];

    for (int i = 0; i < FILTER_ORDER - 1; i++)
    {
        zi_new[i] = zi_coeffs[i] * y0;
    }

    // Backward filter
    lfilter(filter_temp, filter_out, length + 2 * PADLEN, zi_new);

    // Reverse the signal again
    reverse_signal(filter_out, length + 2 * PADLEN);

    // Remove padding
    for (int i = 0; i < length; i++)
    {
        data[i] = filter_out[PADLEN + i];
    }
}
