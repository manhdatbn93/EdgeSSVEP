/**
 * @file cca_func.h
 * @brief Header file for Canonical Correlation Analysis (CCA) based SSVEP classification
 * @author manhdatbn93@gmail.com
 * @date 2025-05-24
 * @version 1.0
 *
 * This header defines functions and constants for SSVEP classification using CCA.
 * It supports processing of EEG data from multiple channels to detect SSVEP responses.
 */

#ifndef CCA_FUNC_H
#define CCA_FUNC_H

#ifdef __cplusplus
extern "C" {
#endif

/* Constants */
#define CCA_NUM_CHANNELS            8    ///< Number of EEG channels for CCA processing
#define PATTERN_NUM_CHANNELS        8    ///< Number of channels in the input EEG pattern
#define PATTERN_NUM_SAMPLES         3000 ///< Number of samples per channel in the input pattern

/* Configuration Flags */
#define USE_EEG_PATTERN             ///< Define to enable EEG pattern processing

/**
 * @brief Classifies SSVEP frequency using Canonical Correlation Analysis
 *
 * @param eeg_data 2D array of EEG data [channels][samples]
 * @param num_samples Number of samples per channel (should be <= PATTERN_NUM_SAMPLES)
 * @return int Detected SSVEP frequency index (1-based frequency index)
 *
 * @note The function expects pre-processed EEG data with proper referencing
 * and filtering (typically bandpass filtered in the SSVEP frequency range).
 * The sampling rate should be consistent with the expected SSVEP frequencies.
 */
int cca_ssvep_classification(float eeg_data[PATTERN_NUM_CHANNELS][PATTERN_NUM_SAMPLES], int num_samples);

#ifdef __cplusplus
}
#endif

#endif /* CCA_FUNC_H */