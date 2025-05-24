/**
 * @file filter_func.h
 * @brief Zero-phase digital filtering implementation
 * @author manhdatbn93@gmail.com
 * @date 2025-05-24
 * @version 1.0
 *
 * @note Uses forward-backward filtering (filtfilt) to achieve zero-phase distortion
 *       Requires filter coefficients defined in filter_coeffs.h
 */

#ifndef FILTER_FUNC_H
#define FILTER_FUNC_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Apply zero-phase digital filtering to input signal
 *
 * @param data Pointer to input/output signal array
 * @param length Number of samples in the input signal
 *
 * @details Implements filtfilt processing which:
 * 1. Filters the signal forward using specified coefficients
 * 2. Reverses the filtered signal
 * 3. Filters it again
 * 4. Reverses the result to restore original time alignment
 *
 * @note Characteristics:
 * - Zero-phase distortion (no time shift)
 * - Filter order effectively doubled
 * - Requires proper initialization to handle edge effects
 * - Uses coefficients from filter_coeffs.h (Butterworth 2-45Hz)
 *
 * @warning Input length must not exceed MAX_VALUES (4000) defined in filter_coeffs.h
 *          For best results, ensure signal length >> filter order (FILTER_ORDER)
 */
void apply_filtfilt(double* data, int length);

#ifdef __cplusplus
}
#endif

#endif /* FILTER_FUNC_H */