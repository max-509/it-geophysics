#ifndef _AMPLITUDES_CALCULATOR_M256
#define _AMPLITUDES_CALCULATOR_M256

#include "amplitudes_calculator.h"

#include <x86intrin.h>
#include <functional>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <omp.h>

#define ALIGNED(n) __attribute__((aligned(n)))

#define _MM256_TRANSPOSE4_PD(row1, row2, row3, row4) \
__m256d tmp1 = _mm256_unpacklo_pd(row1, row2); \
__m256d tmp2 = _mm256_unpacklo_pd(row1, row3); \
__m256d tmp3 = _mm256_unpackhi_pd(row1, row2); \
__m256d tmp4 = _mm256_unpackhi_pd(row3, row4); \
(row1) = _mm256_permute2f128_pd(tmp1, tmp2, 0x20); \
(row2) = _mm256_permute2f128_pd(tmp1, tmp2, 0x31); \
(row3) = _mm256_permute2f128_pd(tmp3, tmp4, 0x20); \
(row4) = _mm256_permute2f128_pd(tmp3, tmp4, 0x31);

#define _MM256_TRANSPOSE4_PS(src0, src1, src2, src3) \
__m256 tmp0 = _mm256_shuffle_ps(src0, src1, 0x44); \
__m256 tmp2 = _mm256_shuffle_ps(src0, src1, 0xEE); \
__m256 tmp1 = _mm256_shuffle_ps(src2, src3, 0x44); \
__m256 tmp3 = _mm256_shuffle_ps(src2, src3, 0xEE); \
__m256 row0 = _mm256_shuffle_ps(tmp0, tmp1, 0x88); \
__m256 row1 = _mm256_shuffle_ps(tmp0, tmp1, 0xDD); \
__m256 row2 = _mm256_shuffle_ps(tmp2, tmp3, 0x88); \
__m256 row3 = _mm256_shuffle_ps(tmp2, tmp3, 0xDD); \
src0 = _mm256_insertf128_ps(row0, _mm256_castps256_ps128(row1), 1); \
src1 = _mm256_insertf128_ps(row2, _mm256_castps256_ps128(row3), 1); \
src2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_extractf128_ps(row0, 1)), _mm256_extractf128_ps(row1, 1), 1); \
src3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_extractf128_ps(row2, 1)), _mm256_extractf128_ps(row3, 1), 1);

template <typename T>
class AmplitudesCalculatorM256 : public AmplitudesCalculator<T, AmplitudesCalculatorM256<T>> {
public:
	AmplitudesCalculatorM256(const T *sources_coords,
                              const T *rec_coords,
                              const T *tensor_matrix,
                              ptrdiff_t sources_count,
                              ptrdiff_t n_rec,
                              T *amplitudes) : 
		sources_coords_(sources_coords),
		rec_coords_(rec_coords),
		tensor_matrix_(tensor_matrix),
        sources_count(sources_count),
        n_rec(n_rec),
		amplitudes_(amplitudes) 
	{ }

	friend AmplitudesCalculator<T, AmplitudesCalculatorM256<T>>;

private:
	const T *sources_coords_;
    const T *rec_coords_;
    const T *tensor_matrix_;
    ptrdiff_t sources_count;
    ptrdiff_t n_rec;
    T *amplitudes_;
    __m256i mask_d = _mm256_set_epi64x(0, 1, 1, 1);
    __m128i mask_s = _mm_set_epi32(0, 1, 1, 1);

	void realize_calculate() {}

	inline __m256 vect_calc_dist(__m256 x, __m256 y, __m256 z) {
	    return _mm256_add_ps(_mm256_sqrt_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y)), _mm256_mul_ps(z, z))), _mm256_set1_ps(1e-36));
	}

	inline __m256d vect_calc_dist(__m256d x, __m256d y, __m256d z) {
	    return _mm256_add_pd(_mm256_sqrt_pd(_mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(x, x), _mm256_mul_pd(y, y)), _mm256_mul_pd(z, z))), _mm256_set1_pd(1e-300));
	}

    inline void transpose_coord_vect(__m256 *src_vect, float *dest_arr) {
        int count_part = (sizeof(__m256)/sizeof(float))/4;
        __m256 extra_row = _mm256_setzero_ps();

        _MM256_TRANSPOSE4_PS(src_vect[0], src_vect[1], src_vect[2], extra_row);
        for (ptrdiff_t ind = 0; ind < 3; ++ind) {
            for (int part_ind = 0; part_ind < count_part; ++part_ind) {
                _mm_maskstore_ps(dest_arr+(ind*count_part+part_ind)*3, mask_s, _mm256_extractf128_ps(src_vect[ind], part_ind));
            }
        }
        for (int part_ind = 0; part_ind < count_part; ++part_ind) {
            _mm_maskstore_ps(dest_arr+(3*count_part+part_ind)*3, mask_s, _mm256_extractf128_ps(extra_row, part_ind));
        }

        // ptrdiff_t vector_dim = sizeof(__m256)/sizeof(float);
        // ALIGNED(32) long long indeces[vector_dim];

        // for (ptrdiff_t crd = 0; crd < 3; ++crd) {
        //     #pragma omp simd
        //     for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {
        //         indeces[v_s] = vector_dim*((crd*vector_dim+v_s)%3)+(crd*vector_dim+v_s)/3;
        //     }
        //     _mm256_store_ps(dest_arr+crd*vector_dim, _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_i64gather_ps(src_vect, _mm256_load_si256((__m256i*)(indeces)), 1)), _mm256_i64gather_ps(src_vect, _mm256_load_si256((__m256i*)(indeces+vector_dim/2)), 1), 1));
        // }
    }

    inline void transpose_coord_vect(__m256d *src_vect, double *dest_arr) {
        __m256d extra_row = _mm256_setzero_pd();
        
        _MM256_TRANSPOSE4_PD(src_vect[0], src_vect[1], src_vect[2], extra_row);
        for (ptrdiff_t shift = 0; shift < 3; ++shift) {
            _mm256_maskstore_pd(dest_arr+shift*3, mask_d, src_vect[shift]);   
        }
        _mm256_maskstore_pd(dest_arr+9, mask_d, extra_row);   

        
        // ptrdiff_t vector_dim = sizeof(__m256d)/sizeof(double);
        // ALIGNED(32) long long indeces[vector_dim];

        // for (ptrdiff_t crd = 0; crd < 3; ++crd) {
        //     #pragma omp simd
        //     for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {
        //         indeces[v_s] = vector_dim*((crd*vector_dim+v_s)%3)+(crd*vector_dim+v_s)/3;
        //     }
        //     _mm256_store_pd(dest_arr+crd*vector_dim, _mm256_i64gather_pd(src_vect, _mm256_load_si256((__m256i*)(indeces)), 1));
        // }   
    }

};

template <>
void AmplitudesCalculatorM256<float>::realize_calculate() {
    ptrdiff_t matrix_size = 6;
    ptrdiff_t vector_dim = sizeof(__m256)/sizeof(float);
    // std::unique_ptr<__m256[], decltype(free)*> vect_rec_coord{static_cast<__m256*>(aligned_alloc(sizeof(__m256), sizeof(__m256)*(n_rec/vector_dim)*3)), free};

    #pragma omp parallel
    {
        // ALIGNED(32) long long indeces[vector_dim];
        // #pragma omp for collapse(2)
        // for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
        //     for (ptrdiff_t i = 0; i < 3; ++i) {
        //         #pragma omp simd
        //         for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {
        //             indeces[v_s] = (r_ind+v_s)*3+i;
        //         }
        //         vect_rec_coord[(r_ind/vector_dim)*3+i] = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_i64gather_ps(rec_coords_, _mm256_load_si256((__m256i*)(indeces)), 1)), _mm256_i64gather_ps(rec_coords_, _mm256_load_si256((__m256i*)(indeces+vector_dim/2)), 1), 1);
        //         // vect_rec_coord[(r_ind/vector_dim)*3+i] = _mm256_set_ps(rec_coords_[(r_ind+7)*3+i], rec_coords_[(r_ind+6)*3+i], rec_coords_[(r_ind+5)*3+i], rec_coords_[(r_ind+4)*3+i],
        //         //                                                        rec_coords_[(r_ind+3)*3+i], rec_coords_[(r_ind+2)*3+i], rec_coords_[(r_ind+1)*3+i], rec_coords_[(r_ind)*3+i]);
        //     }
        // }

        __m256 coord_vec[3];
        __m256 G_P_vect[matrix_size];
        // ALIGNED(32) float coords[vector_dim*3];
        // ALIGNED(32) float coords_transposed[vector_dim*3];
        // ALIGNED(32) float G_P[matrix_size*vector_dim];
        #pragma omp for collapse(2)
        for (ptrdiff_t i = 0; i < sources_count; ++i) {
            for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm256_sub_ps(_mm256_set_ps(rec_coords_[(r_ind+7)*3+crd], rec_coords_[(r_ind+6)*3+crd], rec_coords_[(r_ind+5)*3+crd], rec_coords_[(r_ind+4)*3+crd],
                                                                rec_coords_[(r_ind+3)*3+crd], rec_coords_[(r_ind+2)*3+crd], rec_coords_[(r_ind+1)*3+crd], rec_coords_[(r_ind)*3+crd]
                                                                ), _mm256_set1_ps(sources_coords_[i*3+crd]));
                }

                __m256 dist = vect_calc_dist(coord_vec[0], coord_vec[1], coord_vec[2]);

                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm256_div_ps(coord_vec[crd], dist);
                    // _mm256_store_ps(coords+crd*vector_dim, coord_vec[crd]);
                    G_P_vect[crd] = _mm256_div_ps(_mm256_mul_ps(coord_vec[crd], coord_vec[crd]), dist);
                }

                G_P_vect[3] = _mm256_div_ps(_mm256_mul_ps(coord_vec[0], coord_vec[1]), dist);
                G_P_vect[4] = _mm256_div_ps(_mm256_mul_ps(coord_vec[0], coord_vec[2]), dist);
                G_P_vect[5] = _mm256_div_ps(_mm256_mul_ps(coord_vec[1], coord_vec[2]), dist);

                // transpose_coord_vect(coord_vec, coords_transposed);

                // for (ptrdiff_t m = 0; m < matrix_size; ++m) {
                //     _mm256_store_ps(G_P+m*vector_dim, G_P_vect[m]);
                //     #pragma omp simd
                //     for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {
                //         for (ptrdiff_t rec_comp = 0; rec_comp < 3; ++rec_comp) {
                //             amplitudes_[i*n_rec*3+(r_ind+v_s)*3+rec_comp] += G_P[m*vector_dim+v_s]*coords_transposed[v_s*3+rec_comp]*tensor_matrix_[m];
                //         }
                //     }
                // }

                ALIGNED(32) float tmp_dot[8] = {};
                for (ptrdiff_t m = 0; m < matrix_size; ++m) {
                    _mm256_store_ps(tmp_dot, _mm256_add_ps(_mm256_load_ps(tmp_dot), _mm256_mul_ps(G_P_vect[m], _mm256_set1_ps(tensor_matrix_[m]))));
                }

                __m256 extra_row = _mm256_setzero_ps();
                _MM256_TRANSPOSE4_PS(coord_vec[0], coord_vec[1], coord_vec[2], extra_row);

                for (ptrdiff_t ind = 0; ind < 3; ++ind) {
                    _mm_storeu_ps(amplitudes_+i*n_rec*3+(r_ind+(ind*(vector_dim/4)))*3, _mm_mul_ps(_mm256_extractf128_ps(coord_vec[ind], 0), _mm_set1_ps(tmp_dot[ind*(vector_dim/4)])));
                    _mm_storeu_ps(amplitudes_+i*n_rec*3+(r_ind+(ind*(vector_dim/4)+1))*3, _mm_mul_ps(_mm256_extractf128_ps(coord_vec[ind], 1), _mm_set1_ps(tmp_dot[ind*(vector_dim/4)+1])));
                }
                _mm_storeu_ps(amplitudes_+i*n_rec*3+r_ind*3+(3*(vector_dim/4)+0)*3, _mm_mul_ps(_mm256_extractf128_ps(extra_row, 0), _mm_set1_ps(tmp_dot[3*(vector_dim/4)])));
                _mm_maskstore_ps(amplitudes_+i*n_rec*3+r_ind*3+(3*(vector_dim/4)+1)*3, mask_s, _mm_mul_ps(_mm256_extractf128_ps(extra_row, 1), _mm_set1_ps(tmp_dot[3*(vector_dim/4)+1])));
            }
        }
    }
    non_vector_calculate_amplitudes(n_rec-(n_rec%vector_dim), sources_coords_, rec_coords_, tensor_matrix_, sources_count, n_rec, amplitudes_);
}

template <>
void AmplitudesCalculatorM256<double>::realize_calculate() {
    ptrdiff_t matrix_size = 6;
    ptrdiff_t vector_dim = sizeof(__m256d)/sizeof(double);
    // ALIGNED(32) __m256d vect_rec_coord[(n_rec/vector_dim)*3];
    // std::unique_ptr<__m256d[], decltype(free)*> vect_rec_coord{static_cast<__m256d*>(aligned_alloc(sizeof(__m256d), sizeof(__m256d)*(n_rec/vector_dim)*3)), free};

    #pragma omp parallel
    {
        // ALIGNED(32) long long indeces[vector_dim];
        // #pragma omp for collapse(2)
        // for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
        //     for (ptrdiff_t i = 0; i < 3; ++i) {
        //         #pragma omp simd
        //         for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {
        //             indeces[v_s] = (r_ind+v_s)*3+i;
        //         }
        //         vect_rec_coord[(r_ind/vector_dim)*3+i] = _mm256_i64gather_pd(rec_coords_, _mm256_load_si256((__m256i*)(indeces)), 1);
        //         // vect_rec_coord[(r_ind/vector_dim)*3+i] = _mm256_set_pd(rec_coords_[(r_ind+3)*3+i], rec_coords_[(r_ind+2)*3+i], rec_coords_[(r_ind+1)*3+i], rec_coords_[(r_ind)*3+i]);
        //     }
        // }

        __m256d coord_vec[3];
        __m256d G_P_vect[matrix_size];
        // ALIGNED(32) double coords[vector_dim*3];
        ALIGNED(32) double coords_transposed[vector_dim*3];
        ALIGNED(32) double G_P[matrix_size*vector_dim];
        #pragma omp for collapse(2)
        for (ptrdiff_t i = 0; i < sources_count; ++i) {
            for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm256_sub_pd(_mm256_set_pd(rec_coords_[(r_ind+3)*3+crd], rec_coords_[(r_ind+2)*3+crd], rec_coords_[(r_ind+1)*3+crd], rec_coords_[(r_ind)*3+crd]), _mm256_set1_pd(sources_coords_[i*3+crd]));
                }

                __m256d dist = vect_calc_dist(coord_vec[0], coord_vec[1], coord_vec[2]);

                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm256_div_pd(coord_vec[crd], dist);
                    // _mm256_store_pd(coords+crd*vector_dim, coord_vec[crd]);
                    G_P_vect[crd] = _mm256_div_pd(_mm256_mul_pd(coord_vec[crd], coord_vec[crd]), dist);
                }

                G_P_vect[3] = _mm256_div_pd(_mm256_mul_pd(coord_vec[0], coord_vec[1]), dist);
                G_P_vect[4] = _mm256_div_pd(_mm256_mul_pd(coord_vec[0], coord_vec[2]), dist);
                G_P_vect[5] = _mm256_div_pd(_mm256_mul_pd(coord_vec[1], coord_vec[2]), dist);

                // transpose_coord_vect(coord_vec, coords_transposed);

                // for (ptrdiff_t m = 0; m < matrix_size; ++m) {
                //     _mm256_store_pd(G_P+m*vector_dim, G_P_vect[m]);
                //     #pragma omp simd
                //     for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {
                //         for (ptrdiff_t rec_comp = 0; rec_comp < 3; ++rec_comp) {
                //             amplitudes_[i*n_rec*3+(r_ind+v_s)*3+rec_comp] += G_P[m*vector_dim+v_s]*coords_transposed[v_s*3+rec_comp]*tensor_matrix_[m];
                //         }
                //     }
                // }

                ALIGNED(32) double tmp_dot[4] = {};
                for (ptrdiff_t m = 0; m < matrix_size; ++m) {
                    _mm256_store_pd(tmp_dot, _mm256_add_pd(_mm256_load_pd(tmp_dot), _mm256_mul_pd(G_P_vect[m], _mm256_set1_pd(tensor_matrix_[m]))));
                }

                __m256d extra_row = _mm256_setzero_pd();
                _MM256_TRANSPOSE4_PD(coord_vec[0], coord_vec[1], coord_vec[2], extra_row);
                coord_vec[0] = _mm256_mul_pd(coord_vec[0], _mm256_set1_pd(tmp_dot[0]));
                coord_vec[1] = _mm256_mul_pd(coord_vec[1], _mm256_set1_pd(tmp_dot[1]));
                coord_vec[2] = _mm256_mul_pd(coord_vec[2], _mm256_set1_pd(tmp_dot[2]));
                extra_row = _mm256_mul_pd(extra_row, _mm256_set1_pd(tmp_dot[3]));

                for (ptrdiff_t shift = 0; shift < 3; ++shift) {
                    _mm256_storeu_pd(amplitudes_+i*n_rec*3+(r_ind+shift)*3, coord_vec[shift]);   
                }
                _mm256_maskstore_pd(amplitudes_+i*n_rec*3+(r_ind+3)*3, mask_d, extra_row);   
            }
        }         
    }
    non_vector_calculate_amplitudes(n_rec-(n_rec%vector_dim), sources_coords_, rec_coords_, tensor_matrix_, sources_count, n_rec, amplitudes_);
}

#endif /*_AMPLITUDES_CALCULATOR_M256*/