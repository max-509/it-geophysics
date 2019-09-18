#include "coherent_summation.h"

#include <algorithm>
#include <omp.h>

void compute(const float* rec_samples, const float* rec_coords, const float* displaces_vectors, const int32_t* sources_times, ptrdiff_t n_samples, ptrdiff_t n_rec, ptrdiff_t n_xyz, float* result_data) {
	ptrdiff_t rec_block_size = 60;
    ptrdiff_t samples_block_size = 400;

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
    	for (ptrdiff_t c_t = 0; c_t < n_samples; c_t += samples_block_size) {
            for (ptrdiff_t c_r = 0; c_r < n_rec; c_r += rec_block_size) {
            	for (ptrdiff_t i = 0; i < n_xyz; ++i) {
	                for (ptrdiff_t m = c_r; m < std::min(c_r+rec_block_size, n_rec); ++m) {
	                    ptrdiff_t ind = sources_times[i*n_rec+m];
                        for (ptrdiff_t coord = 0; coord < 3; ++coord) {
                            for (ptrdiff_t l = c_t; l < std::min(c_t+samples_block_size, n_samples-ind); ++l) {
                                result_data[i*n_samples+l] += rec_samples[m*(n_samples+coord)+ind+l]*displaces_vectors[i*n_rec+m];
                            }
	                    }
	                }
	            }
            }
        }
    }

}