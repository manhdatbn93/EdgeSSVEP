/**
 * @file main.c
 * @brief EEG Data Processing and SSVEP Classification System
 * @author manhdatbn93@gmail.com
 * @date 2025-05-24
 * @version 1.0
 *
 * @note This program implements a complete SSVEP classification pipeline:
 *       - EEG data reading from file
 *       - Signal preprocessing
 *       - CCA-based frequency detection
 */

#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include "cca_func.h"

/**
 * @brief Global EEG data buffer for a single trial
 *
 * Stores multi-channel EEG data in format:
 * [PATTERN_NUM_CHANNELS][PATTERN_NUM_SAMPLES]
 *
 */
static float eeg_data[PATTERN_NUM_CHANNELS][PATTERN_NUM_SAMPLES];


/**
 * @brief Read EEG data from text file into global buffer
 *
 * @param[in] filename Path to the EEG data file
 * @return int Number of samples successfully read, or -1 on error
 *
 */
static int read_eeg_data(const char* filename)
{
    FILE* file = fopen(filename, "r");
    if (file == NULL)
    {
        printf("Error: Cannot open file %s\n", filename);
        return -1;
    }

    int sample = 0;
    while (sample < PATTERN_NUM_SAMPLES)
    {
        for (int ch = 0; ch < PATTERN_NUM_CHANNELS; ch++)
        {
            if (fscanf(file, "%f", &eeg_data[ch][sample]) != 1)
            {
                fclose(file);
                return sample; // Return number of samples read
            }
        }
        sample++;
    }

    fclose(file);
    return sample; // Total samples read
}


void main()
{
    int correct_index = 0;
    char subject_name[10];
    int estimate_labels[24];
    int true_labels[24];
    int correct_count = 0;

    printf("EdgeSSVEP: Real-time SSVEP classification system for edge devices\r\n");

    /* Initialize true labels (1-6 repeated 4 times) */
    for (int i = 0; i < 24; i++) 
    {
        true_labels[i] = (i % 6) + 1;
    }

    // Loop through each subject folder (S01 to S10)
    for (int subject = 1; subject <= 10; subject++) 
    {
        correct_count = 0;
        sprintf(subject_name, "S%02d", subject);

        /* Process 24 trials */
        for (int i = 0; i < 24; i++)
        {
            char filename[100];

            sprintf(filename, "Dataset_txt/%s/trial_%d.txt", subject_name, i); // Linux/macOS

            /* Load and process EEG data */
            int samples_read = read_eeg_data(filename);
            if (samples_read > 0)
            {
                /* Perform CCA classification */
                correct_index = cca_ssvep_classification(eeg_data, samples_read);  // Pass EEG data to classification
                estimate_labels[i] = correct_index;
                //printf("\r\nTrial %d: Label %d", i + 1, correct_index);

                /* Validate prediction */
                if (correct_index == true_labels[i])
                {
                    correct_count++;
                }
            }
            else
            {
                printf("Trial %d in %s: No valid data found.\n", i + 1, subject_name);
            }
        }

        /* Calculate and display accuracy */
        float accuracy = (float)correct_count / 24.0f * 100.0f;
        printf("Accuracy for %s: %.2f%% (%d/24 correct)\r\n", subject_name, accuracy, correct_count);
    }
}