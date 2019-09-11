#include "coherent_summation.h"

// #include <iostream>
// #include <cstdio>
#include <algorithm>
// #include <cmath>
// #include <cstdef>
// #include <memory>
#include <omp.h>
// #include <immintrin.h>
// #include <functional>

// namespace {
// 	inline float calc_radius(float dx, float dy, float dz) {
// 	    return sqrt(dx*dx+dy*dy+dz*dz);
// 	}

// 	inline __m256 vect_calc_radius(__m256 dx, __m256 dy, __m256 dz) {
// 	    return _mm256_sqrt_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)), _mm256_mul_ps(dz, dz)));
// 	}
// }

void compute(const float* rec_samples, const float* rec_coords, const int32_t* sources_times, ptrdiff_t n_samples, ptrdiff_t n_rec, ptrdiff_t n_xyz, float* result_data) {
	ptrdiff_t rec_block_size = 60;
    ptrdiff_t samples_block_size = 400;

    // std::unique_ptr<ptrdiff_t[]> min_ind_arr{new ptrdiff_t[n_xyz]};
    // std::unique_ptr<ptrdiff_t[]> ind_arr{new ptrdiff_t[n_xyz*n_rec]};
    // __m256 vect_rec_coord[(n_rec/8)*3];

    #pragma omp parallel
    {
        // if (n_rec >= 8) {
        //     #pragma omp for collapse(2)
        //     for (ptrdiff_t m = 0; m < n_rec; m+=8) {
        //         for (ptrdiff_t i = 0; i < 3; ++i) {
        //             vect_rec_coord[(m/8)*3+i] = _mm256_set_ps(rec_coords[(m+7)*3+i], rec_coords[(m+6)*3+i], rec_coords[(m+5)*3+i], rec_coords[(m+4)*3+i],
        //                                                       rec_coords[(m+3)*3+i], rec_coords[(m+2)*3+i], rec_coords[(m+1)*3+i], rec_coords[m*3+i]);
        //         }
        //     }
        //     __m256 vect_ind_min;
        //     #pragma omp for
        //     for (ptrdiff_t i = 0; i < n_xyz; ++i) {
        //         for (ptrdiff_t m = 0; m < n_rec; m+=8) {
        //             __m256 vect_ind = _mm256_add_ps(_mm256_set1_ps(1.0f),
        //                                             _mm256_round_ps(_mm256_div_ps(vect_calc_radius(
        //                                             _mm256_sub_ps(_mm256_set1_ps(area_coords[i*3]), vect_rec_coord[(m/8)*3]),
        //                                             _mm256_sub_ps(_mm256_set1_ps(area_coords[i*3+1]), vect_rec_coord[(m/8)*3+1]),
        //                                             _mm256_sub_ps(_mm256_set1_ps(area_coords[i*3+2]), vect_rec_coord[(m/8)*3+2])),
        //                                             _mm256_mul_ps(_mm256_set1_ps(vv), _mm256_set1_ps(dt))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)));
        //             if (0 != m) {
        //                 vect_ind_min = _mm256_min_ps(vect_ind_min, vect_ind);
        //             } else {
        //                 vect_ind_min = vect_ind;
        //             }
        //             float temp_ind[8];
        //             _mm256_store_ps(temp_ind, vect_ind);
        //             for (ptrdiff_t v = 0; v < 8; ++v) {
        //                 ind_arr[i*n_rec+m+v] = temp_ind[v];
        //             }

        //         }

        //         float min_indexes[8];
        //         _mm256_store_ps(min_indexes, vect_ind_min);
        //         min_ind_arr[i] = *std::min_element(min_indexes, min_indexes+8);

        //         for (ptrdiff_t m = n_rec-(n_rec%8); m < n_rec; ++m) {
        //             ind_arr[i*n_rec+m] = round(calc_radius(area_coords[i*3]-rec_coords[m*3],
        //                                                    area_coords[i*3+1]-rec_coords[m*3+1],
        //                                                    area_coords[i*3+2]-rec_coords[m*3+2])
        //                                                    /(vv*dt)) + 1;
        //             min_ind_arr[i] = std::min(min_ind_arr[i], ind_arr[i*n_rec+m]);
        //         }
        //     }
        // } else {
        //     #pragma omp for
        //     for (ptrdiff_t i = 0; i < n_xyz; ++i) {
        //         ind_arr[i*n_rec] = round(calc_radius((area_coords[0])-rec_coords[0],
        //                                                  (area_coords[1])-rec_coords[1],
        //                                                  (area_coords[2])-rec_coords[2])
        //                                                  /(vv*dt)) + 1;
        //         min_ind_arr[i] = ind_arr[i*n_rec];

        //         for (ptrdiff_t m = 1; m < n_rec; ++m) {
        //             ind_arr[i*n_rec+m] = round(calc_radius((area_coords[i*3])-rec_coords[m*3],
        //                                                    (area_coords[i*3+1])-rec_coords[m*3+1],
        //                                                    (area_coords[i*3+2])-rec_coords[m*3+2])
        //                                                     /(vv*dt)) + 1;
        //             min_ind_arr[i] = std::min(min_ind_arr[i], ind_arr[i*n_rec+m]);
        //         }

        //     }
        // }

        #pragma omp for schedule(dynamic)
    	for (ptrdiff_t c_t = 0; c_t < n_samples; c_t += samples_block_size) {
            for (ptrdiff_t c_r = 0; c_r < n_rec; c_r += rec_block_size) {
            	for (ptrdiff_t i = 0; i < n_xyz; ++i) {
	                for (ptrdiff_t m = c_r; m < std::min(c_r+rec_block_size, n_rec); ++m) {
	                    ptrdiff_t ind = sources_times[i*n_rec+m];
	                    for (ptrdiff_t l = c_t; l < std::min(c_t+samples_block_size, n_samples-ind); ++l) {
	                        result_data[i*n_samples+l] += rec_samples[m*n_samples+ind+l];
	                    }
	                }
	            }
            }
        }
    }

}