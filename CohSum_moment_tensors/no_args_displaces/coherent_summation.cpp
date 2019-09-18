#include "coherent_summation.h"

#include <algorithm>
#include <omp.h>
#include <memory>
#include <cmath>
#include <imminthin.h>

inline float calc_norm(float x, float y, float z) {
    return sqrt(x*x+y*y+z*z);
}

inline __m256 vect_calc_norm(__m256 x, __m256 y, __m256 z) {
    return _mm256_sqrt(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y)), _mm256_mul_ps(z, z)));
}

inline __m256 vect_abs

void compute(const float* rec_samples, const float* rec_coords, const float* sources_coords, const int32_t* sources_times, const float* tensor_matrix, ptrdiff_t n_samples, ptrdiff_t n_rec, ptrdiff_t n_xyz, float* result_data) {
    ptrdiff_t rec_block_size = 60;
    ptrdiff_t samples_block_size = 400;

    std::unique_ptr<float[]> displaces_vectors{new float[n_xyz*n_rec]};
    __m256 vect_rec_coord[(n_rec/8)*3];
    #pragma omp parallel
    {
        if (n_rec >= 8) {
            #pragma omp for collaspe(2)
            for (ptrdiff_t m = 0; m < n_rec; m+=8) {
                for (ptrdiff_t i = 0; i < 3; ++i) {
                    vect_rec_coord[(m/8)*3+i] = _mm256_set_ps(rec_coords[(m+7)*3+i], rec_coords[(m+6)*3+i], rec_coords[(m+5)*3+i], rec_coords[(m+4)*3+i], 
                                                              rec_coords[(m+3)*3+i], rec_coords[(m+2)*3+i], rec_coords[(m+1)*3+i], rec_coords[(m+0)*3+i]);
                }
            }

            #pragma omp for collapes(2)
            for (ptrdiff_t i = 0; i < n_xyz; ++i) {
                for (ptrdiff_t m = 0; m < n_rec; m+=8) {
                    __m256 x = _mm256_sub_ps(_mm256_set1_ps(sources_coords[i*3]), vect_rec_coord[(m/8)*3]);
                    __m256 y = _mm256_sub_ps(_mm256_set1_ps(sources_coords[i*3+1]), vect_rec_coord[(m/8)*3+1]);
                    __m256 z = _mm256_sub_ps(_mm256_set1_ps(sources_coords[i*3+2]), vect_rec_coord[(m/8)*3+2]);
                    __m256 norm = vect_calc_norm(x, y, z);

                }
            }

        } else {
            #pragma omp for schedule(static) collapse(2)
            for (ptrdiff_t i = 0; i < n_xyz; ++i) {
                for (ptrdiff_t m = 0; m < n_rec; ++m) {
                    float x = sources_coords[i*3]-rec_coords[m*3];
                    float y = sources_coords[i*3+1]-rec_coords[m*3+1];
                    float z = sources_coords[i*3+2]-rec_coords[m*3+2];
                    float norm = calc_norm(x, y, z);
                    x = x/norm;
                    y = y/norm;
                    z = z/norm;
                    float G = x*y*z;
                    displaces_vectors[i*n_rec+m] = fabs(G*tensor_matrix[1]);
                }
            }
        }
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