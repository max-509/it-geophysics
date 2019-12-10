#ifndef _AMPLITUDES_CALCULATOR_M512
#define _AMPLITUDES_CALCULATOR_M512

#include "amplitudes_calculator.h"

#include <x86intrin.h>
#include <functional>
#include <cstdlib>
#include <cmath>
#include <memory>

#define ALIGNED(n) __attribute__((aligned(n)))

#define _MM512_TRANSPOSE4_PS(src0, src1, src2, src3) \
__m512 tmp0 = _mm512_shuffle_ps(src0, src1, 0x44); \
__m512 tmp1 = _mm512_shuffle_ps(src2, src3, 0x44); \
__m512 tmp2 = _mm512_shuffle_ps(src0, src1, 0xEE); \
__m512 tmp3 = _mm512_shuffle_ps(src2, src3, 0xEE); \
__m512 tmp4 = _mm512_shuffle_ps(tmp0, tmp1, 0x88); \
__m512 tmp5 = _mm512_shuffle_ps(tmp0, tmp1, 0xDD); \
__m512 tmp6 = _mm512_shuffle_ps(tmp2, tmp3, 0x88); \
__m512 tmp7 = _mm512_shuffle_ps(tmp2, tmp3, 0xDD); \
src0 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm512_castps512_ps256(tmp4), _mm512_castps512_ps128(tmp5), 1)), _mm256_insertf128_ps(_mm512_castps512_ps256(tmp6), _mm512_castps512_ps128(tmp5), 1), 1); \
src1 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm512_extractf32x4_ps(tmp4, 1)), _mm512_extractf32x4_ps(tmp5, 1), 1)), \
                          _mm256_insertf128_ps(_mm256_castps128_ps256(_mm512_extractf32x4_ps(tmp6, 1)), _mm512_extractf32x4_ps(tmp7, 1), 1), 1); \
src2 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm512_extractf32x4_ps(tmp4, 2)), _mm512_extractf32x4_ps(tmp5, 2), 1)), \
                          _mm256_insertf128_ps(_mm256_castps128_ps256(_mm512_extractf32x4_ps(tmp6, 2)), _mm512_extractf32x4_ps(tmp7, 2), 1), 1); \
src3 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm512_extractf32x4_ps(tmp4, 3)), _mm512_extractf32x4_ps(tmp5, 3), 1)), \
                          _mm256_insertf128_ps(_mm256_castps128_ps256(_mm512_extractf32x4_ps(tmp6, 3)), _mm512_extractf32x4_ps(tmp7, 3), 1), 1);

#define _MM512_TRANSPOSE4_PD(src0, src1, src2, src3) \
__m512d tmp0 = _mm512_shuffle_pd(src0, src1, 0x44); \
__m512d tmp2 = _mm512_shuffle_pd(src0, src1, 0xEE); \
__m512d tmp1 = _mm512_shuffle_pd(src2, src3, 0x44); \
__m512d tmp3 = _mm512_shuffle_pd(src2, src3, 0xEE); \
__m512d row0 = _mm512_shuffle_pd(tmp0, tmp1, 0x88); \
__m512d row1 = _mm512_shuffle_pd(tmp0, tmp1, 0xDD); \
__m512d row2 = _mm512_shuffle_pd(tmp2, tmp3, 0x88); \
__m512d row3 = _mm512_shuffle_pd(tmp2, tmp3, 0xDD); \
src0 = _mm512_insertf64x4(row0, _mm512_castpd512_pd256(row1), 1); \
src1 = _mm512_insertf64x4(row2, _mm512_castpd512_pd256(row3), 1); \
src2 = _mm512_insertf64x4(_mm512_castpd256_pd512( _mm512_extractf64x4_pd(row0, 1)),  _mm512_extractf64x4_pd(row1, 1), 1); \
src3 = _mm512_insertf64x4(_mm512_castpd256_pd512( _mm512_extractf64x4_pd(row2, 1)),  _mm512_extractf64x4_pd(row3, 1), 1);

template <typename T>
class AmplitudesCalculatorM512 : public AmplitudesCalculator<T, AmplitudesCalculatorM512<T>> {
public:
	AmplitudesCalculatorM512(const T *sources_coords,
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

	friend AmplitudesCalculator<T, AmplitudesCalculatorM512<T>>;

private:
	const T *sources_coords_;
    const T *rec_coords_;
    const T *tensor_matrix_;
    ptrdiff_t sources_count;
    ptrdiff_t n_rec;
    T *amplitudes_;
    __m256i mask_d = _mm256_set_epi64x(0, 1, 1, 1);
    __m128i mask_s = _mm_set_epi32(0, 1, 1, 1);

	void realize_calculate();

	inline __m512 vect_calc_dist(__m512 x, __m512 y, __m512 z) {
	    return _mm512_add_ps(_mm512_sqrt_ps(_mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(x, x), _mm512_mul_ps(y, y)), _mm512_mul_ps(z, z))), _mm512_set1_ps(1e-36));
	}

	inline __m512d vect_calc_dist(__m512d x, __m512d y, __m512d z) {
	    return _mm512_add_pd(_mm512_sqrt_pd(_mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(x, x), _mm512_mul_pd(y, y)), _mm512_mul_pd(z, z))), _mm512_set1_pd(1e-300));
	}

    inline void transpose_coord_vect(const float *src_vect, float *dest_arr) {
        int count_part = (sizeof(__m512)/sizeof(float))/4;
        __m512 srcs[4];

        srcs[0] = _mm512_load_ps(src_vect);
        srcs[1] = _mm512_load_ps(src_vect+16);
        srcs[2] = _mm512_load_ps(src_vect+24);
        srcs[3] = _mm512_setzero_ps();
        _MM512_TRANSPOSE4_PS(srcs[0], srcs[1], srcs[2], srcs[3]);
        for (ptrdiff_t ind = 0; ind < 4; ++ind) {
            for (int part_ind = 0; part_ind < count_part; ++part_ind) {
                _mm_maskstore_ps(dest_arr+(ind*count_part+part_ind)*3, mask_s, _mm512_extractf32x4_ps(srcs[ind], part_ind));
            }
        }

        // ptrdiff_t vector_dim = sizeof(__m512)/sizeof(float);
        // ALIGNED(64) long long indeces[vector_dim];

        // for (ptrdiff_t crd = 0; crd < 3; ++crd) {
        //     #pragma omp simd
        //     for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {
        //         indeces[v_s] = vector_dim*((crd*vector_dim+v_s)%3)+(crd*vector_dim+v_s)/3;
        //     }
        //     _mm512_store_ps(dest_arr+crd*vector_dim, _mm512_insertf32x8(_mm512_castps256_ps512(_mm512_i64gather_ps(_mm512_load_si512((__m512i*)(indeces)), src_vect, 1)), _mm512_i64gather_ps(_mm512_load_si512((__m512i*)(indeces+vector_dim/2)), src_vect, 1), 1));
        // }
    }

    inline void transpose_coord_vect(const double *src_vect, double *dest_arr) {
        int count_part = (sizeof(__m512d)/sizeof(double))/4;
        __m512d srcs[4];
        srcs[0] = _mm512_load_pd(src_vect);
        srcs[1] = _mm512_load_pd(src_vect+8);
        srcs[2] = _mm512_load_pd(src_vect+16);
        srcs[3] = _mm512_setzero_pd();
        _MM512_TRANSPOSE4_PD(srcs[0], srcs[1], srcs[2], srcs[3]);
        for (ptrdiff_t ind = 0; ind < 4; ++ind) {
            for (int part_ind = 0; part_ind < count_part; ++part_ind) {
                 _mm256_maskstore_pd(dest_arr+(ind*count_part+part_ind)*3, mask_d, _mm512_extractf64x4_pd(rows[ind], part_ind)); 
            }
        }

        // ptrdiff_t vector_dim = sizeof(__m512d)/sizeof(double);
        // ALIGNED(64) long long indeces[vector_dim];

        // for (ptrdiff_t crd = 0; crd < 3; ++crd) {
        //     #pragma omp simd
        //     for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {
        //         indeces[v_s] = vector_dim*((crd*vector_dim+v_s)%3)+(crd*vector_dim+v_s)/3;
        //     }
        //     _mm512_store_pd(dest_arr+crd*vector_dim, _mm512_i64gather_pd(_mm512_load_si512((__m512i*)(indeces)), src_vect, 1));
        // }   
    }

};

template <>
void AmplitudesCalculatorM512<float>::realize_calculate() {
    ptrdiff_t matrix_size = 6;
    ptrdiff_t vector_dim = sizeof(__m512)/sizeof(float);
    std::unique_ptr<__m512[], decltype(free)*> vect_rec_coord{static_cast<__m512*>(aligned_alloc(sizeof(__m512), sizeof(__m512)*(n_rec/vector_dim)*3)), free};

    // #pragma omp parallel
    {
        ALIGNED(64) long long indeces[vector_dim];
        // #pragma omp for collapse(2)
        for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
            for (ptrdiff_t i = 0; i < 3; ++i) {
                // #pragma omp simd
                for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {
                    indeces[v_s] = (r_ind+v_s)*3+i;
                }
                vect_rec_coord[(r_ind/vector_dim)*3+i] = _mm512_insertf32x8(_mm512_castps256_ps512(_mm512_i64gather_ps(_mm512_load_si512((__m512i*)(indeces)), rec_coords_, 1)), _mm512_i64gather_ps(_mm512_load_si512((__m512i*)(indeces+vector_dim/2)), rec_coords_, 1), 1);
                vect_rec_coord[(r_ind/vector_dim)*3+i] = _mm512_set_ps(rec_coords_[(r_ind+15)*3+i], rec_coords_[(r_ind+14)*3+i], rec_coords_[(r_ind+13)*3+i], rec_coords_[(r_ind+12)*3+i],
                                                                       rec_coords_[(r_ind+11)*3+i], rec_coords_[(r_ind+10)*3+i], rec_coords_[(r_ind+9)*3+i], rec_coords_[(r_ind+8)*3+i],
                                                                       rec_coords_[(r_ind+7)*3+i], rec_coords_[(r_ind+6)*3+i], rec_coords_[(r_ind+5)*3+i], rec_coords_[(r_ind+4)*3+i],
                                                                       rec_coords_[(r_ind+3)*3+i], rec_coords_[(r_ind+2)*3+i], rec_coords_[(r_ind+1)*3+i], rec_coords_[(r_ind)*3+i]);
            }
        }

        __m512 coord_vec[3];
        __m512 G_P_vect[matrix_size];
        // ALIGNED(64) float coords_transposed[vector_dim*3];
        // ALIGNED(64) float G_P[matrix_size*vector_dim];
        // #pragma omp for collapse(2)
        for (ptrdiff_t i = 0; i < sources_count; ++i) {
            for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm512_sub_ps(_mm512_set_ps(rec_coords_[(r_ind+15)*3+crd], rec_coords_[(r_ind+14)*3crd], rec_coords_[(r_ind+13)*3+crd], rec_coords_[(r_ind+12)*3+crd],
                                                               rec_coords_[(r_ind+11)*3+crd], rec_coords_[(r_ind+10)*3crd], rec_coords_[(r_ind+9)*3+crd], rec_coords_[(r_ind+8)*3+crd],
                                                               rec_coords_[(r_ind+7)*3+crd], rec_coords_[(r_ind+6)*3+crd], rec_coords_[(r_ind+5)*3+crd], rec_coords_[(r_ind+4)*3+crd],
                                                               rec_coords_[(r_ind+3)*3+crd], rec_coords_[(r_ind+2)*3+crd], rec_coords_[(r_ind+1)*3+crd], rec_coords_[(r_ind)*3+crd]), _mm512_set1_ps(sources_coords_[i*3+crd]));
                }

                __m512 dist = vect_calc_dist(coord_vec[0], coord_vec[1], coord_vec[2]);

                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm512_div_ps(coord_vec[crd], dist);
                    G_P_vect[crd] = _mm512_div_ps(_mm512_mul_ps(coord_vec[crd], coord_vec[crd]), dist);
                }

                G_P_vect[3] = _mm512_div_ps(_mm512_mul_ps(_mm512_mul_ps(coord_vec[1], coord_vec[2]), _mm512_set1_ps(2.)), dist);
                G_P_vect[4] = _mm512_div_ps(_mm512_mul_ps(_mm512_mul_ps(coord_vec[0], coord_vec[2]), _mm512_set1_ps(2.)), dist);
                G_P_vect[5] = _mm512_div_ps(_mm512_mul_ps(_mm512_mul_ps(coord_vec[0], coord_vec[1]), _mm512_set1_ps(2.)), dist);

                // transpose_coord_vect(coords, coords_transposed);

                // for (ptrdiff_t m = 0; m < matrix_size; ++m) {
                //     _mm512_store_ps(G_P+m*vector_dim, G_P_vect[m]);
                //     // #pragma omp simd
                //     for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {
                //         for (ptrdiff_t rec_comp = 0; rec_comp < 3; ++rec_comp) {
                //             amplitudes_[i*n_rec*3+(r_ind+v_s)*3+rec_comp] += G_P[m*vector_dim+v_s]*coords_transposed[v_s*3+rec_comp]*tensor_matrix_[m];
                //         }
                //     }
                // }

                __m512 tmp_dot = _mm512_setzero_ps();
                for (ptrdiff_t m = 0; m < matrix_size; ++m) {
                    tmp_dot = _mm512_add_ps(tmp_dot, _mm512_mul_ps(G_P[m], _mm512_set1_ps(tensor_matrix_[m])));
                }

                coord_vec[0] = _mm512_mul_ps(coord_vec[0], tmp_dot);
                coord_vec[1] = _mm512_mul_ps(coord_vec[1], tmp_dot);
                coord_vec[2] = _mm512_mul_ps(coord_vec[2], tmp_dot);

                __m512 tmp_coord_vect1 = coord_vec[0];
                __m512 tmp_coord_vect2 = coord_vec[1];
                __m512 extra_row = _mm512_mask_shuffle_ps(_mm512_permute_ps(coord_vec[0], 0x40), 0x7777, coord_vec[1], coord_vec[2], 0x2C);
                coord_vec[0] = _mm512_mask_shuffle_ps(coord_vec[0], 0x7777, coord_vec[1], coord_vec[2], 0x14);
                coord_vec[1] = _mm512_mask_shuffle_ps(coord_vec[1], 0x7777, coord_vec[2], tmp_coord_vect1, 0x1C);
                coord_vec[1] = _mm512_mask_shuffle_ps(coord_vec[2], 0x7777, tmp_coord_vect1, tmp_coord_vect2, 0x2C);
                _MM512_TRANSPOSE4_PS(coord_vec[0], coord_vec[1], coord_vec[2], extra_row);

                _mm512_storeu_ps(amplitudes_+i*n_rec*3+r_ind*3, coord_vec[0]);
                _mm512_storeu_ps(amplitudes_+i*n_rec*3+r_ind*3+12, coord_vec[1]);
                _mm512_storeu_ps(amplitudes_+i*n_rec*3+r_ind*3+24, coord_vec[2]);
                _mm512_mask_storeu_ps(amplitudes_+i*n_rec*3+r_ind*3+36, 0xFFF0, extra_row);
            }
        }     
    }
    non_vector_calculate_amplitudes(n_rec-(n_rec%vector_dim), sources_coords_, rec_coords_, tensor_matrix_, sources_count, n_rec, amplitudes_);
}

template<>
void AmplitudesCalculatorM512<double>::realize_calculate() {
    ptrdiff_t matrix_size = 6;
    ptrdiff_t vector_dim = sizeof(__m512d)/sizeof(double);

    #pragma omp parallel
    {
        __m512d coord_vec[3];
        __m512d G_P_vect[matrix_size];
        // ALIGNED(64) double coords_transposed[vector_dim*3];
        // ALIGNED(64) double G_P[matrix_size*vector_dim];
        #pragma omp for collapse(2)
        for (ptrdiff_t i = 0; i < sources_count; ++i) {
            for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm512_sub_pd(_mm512_set_pd(rec_coords_[(r_ind+7)*3+crd], rec_coords_[(r_ind+6)*3+crd], rec_coords_[(r_ind+5)*3+crd], rec_coords_[(r_ind+4)*3+crd],
                                                               rec_coords_[(r_ind+3)*3+crd], rec_coords_[(r_ind+2)*3+crd], rec_coords_[(r_ind+1)*3+crd], rec_coords_[(r_ind)*3+crd]), _mm512_set1_pd(sources_coords_[i*3+crd]));
                }

                __m512d dist = vect_calc_dist(coord_vec[0], coord_vec[1], coord_vec[2]);

                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm512_div_pd(coord_vec[crd], dist);
                    G_P_vect[crd] = _mm512_div_pd(_mm512_mul_pd(coord_vec[crd], coord_vec[crd]), dist);
                }

                G_P_vect[3] = _mm512_div_pd(_mm512_mul_pd(_mm512_mul_pd(coord_vec[1], coord_vec[2]), _mm512_set1_pd(2.)), dist);
                G_P_vect[4] = _mm512_div_pd(_mm512_mul_pd(_mm512_mul_pd(coord_vec[0], coord_vec[2]), _mm512_set1_pd(2.)), dist);
                G_P_vect[5] = _mm512_div_pd(_mm512_mul_pd(_mm512_mul_pd(coord_vec[1], coord_vec[0]), _mm512_set1_pd(2.)), dist);

                // transpose_coord_vect(coords, coords_transposed);

                // for (ptrdiff_t m = 0; m < matrix_size; ++m) {
                //     _mm512_store_pd(G_P+m*vector_dim, G_P_vect[m]);
                //     #pragma omp simd
                //     for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {
                //         for (ptrdiff_t rec_comp = 0; rec_comp < 3; ++rec_comp) {
                //             amplitudes_[i*n_rec*3+(r_ind+v_s)*3+rec_comp] += G_P[m*vector_dim+v_s]*coords_transposed[v_s*3+rec_comp]*tensor_matrix_[m];
                //         }
                //     }
                // }

                __m512d tmp_dot = _mm512_setzero_pd();
                for (ptrdiff_t m = 0; m < matrix_size; ++m) {
                    tmp_dot = _mm512_add_pd(tmp_dot, _mm512_mul_pd(G_P[m], _mm512_set1_pd(tensor_matrix_[m])));
                }

                coord_vec[0] = _mm512_mul_pd(coord_vec[0], tmp_dot);
                coord_vec[1] = _mm512_mul_pd(coord_vec[1], tmp_dot);
                coord_vec[2] = _mm512_mul_pd(coord_vec[2], tmp_dot);

                __m512d extra_row = _mm512_permute_pd(coord_vec[0], 0xFF);
                coord_vec[0] = _mm512_shuffle_pd(coord_vec[0], coord_vec[1], 0x55);
                coord_vec[1] = _mm512_shuffle_pd(coord_vec[1], coord_vec[2], 0x55);

                _MM512_TRANSPOSE4_PD(coord_vec[0], coord_vec[1], coord_vec[2], extra_row);

                _mm512_storeu_pd(amplitudes_+i*n_rec*3+r_ind*3, coord_vec[0]);
                _mm512_storeu_pd(amplitudes_+i*n_rec*3+r_ind*3+6, coord_vec[1]);
                _mm512_storeu_pd(amplitudes_+i*n_rec*3+r_ind*3+12, coord_vec[2]);
                _mm512_mask_store_pd(amplitudes_+i*n_rec*3+r_ind*3+18, 0xFC, extra_row);
            }
        }      
    }
    non_vector_calculate_amplitudes(n_rec-(n_rec%vector_dim), sources_coords_, rec_coords_, tensor_matrix_, sources_count, n_rec, amplitudes_);
}

#endif /*_AMPLITUDES_CALCULATOR_M512*/