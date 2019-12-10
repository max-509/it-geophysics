#ifndef _AMPLITUDES_CALCULATOR_M128
#define _AMPLITUDES_CALCULATOR_M128

#include "amplitudes_calculator.h"

#include <x86intrin.h>
#include <functional>
#include <cstdlib>
#include <cmath>
#include <memory>

#define ALIGNED(n) __attribute__((aligned(n)))

#define _MM_TRANSPOSE3_PD(row1, row2, row3) \
__m128d tmp1 = _mm_unpacklo_pd(row1, row2); \
__m128d tmp2 = _mm_move_sd(row1, row3); \
__m128d tmp3 = _mm_unpackhi_pd(row2, row3); \
(row1) = tmp1; \
(row2) = tmp2; \
(row3) = tmp3;

template <typename T>
class AmplitudesCalculatorM128 : public AmplitudesCalculator<T, AmplitudesCalculatorM128<T>> {
public:
	AmplitudesCalculatorM128(const T *sources_coords,
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

	friend AmplitudesCalculator<T, AmplitudesCalculatorM128<T>>;

private:
	const T *sources_coords_;
	const T *rec_coords_;
	const T *tensor_matrix_;
    ptrdiff_t sources_count;
    ptrdiff_t n_rec;
	T *amplitudes_;

	void realize_calculate();

	inline __m128 vect_calc_norm(__m128 x, __m128 y, __m128 z) {
	    return _mm_add_ps(_mm_sqrt_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(x, x), _mm_mul_ps(y, y)), _mm_mul_ps(z, z))), _mm_set1_ps(1e-36));
	}

	inline __m128d vect_calc_norm(__m128d x, __m128d y, __m128d z) {
	    return _mm_add_pd(_mm_sqrt_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y)), _mm_mul_pd(z, z))), _mm_set1_pd(1e-300));
	}

    inline void transpose_coord_vect(const float *src_vect, float *dest_arr) {
        __m128 row1 = _mm_load_ps(src_vect);
        __m128 row2 = _mm_load_ps(src_vect+4);
        __m128 row3 = _mm_load_ps(src_vect+8);
        __m128 row4 = _mm_setzero_ps();
        _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
        _mm_store_ps(dest_arr, row1);
        _mm_store_ps(dest_arr+3, row2);
        _mm_store_ps(dest_arr+6, row3);
        _mm_store_ps(dest_arr+9, row4);
    }

    inline void transpose_coord_vect(const double *src_vect, double *dest_arr) {
        __m128 row1 = _mm_load_pd(src_vect);
        __m128 row2 = _mm_load_pd(src_vect+2);
        __m128 row3 = _mm_load_pd(src_vect+4);
        __MM_TRANSPOSE3_PD(row1, row2, row3);
        _mm_store_pd(dest_arr, row1);
        _mm_store_pd(dest_arr+2, row2);
        _mm_store_pd(dest_arr+4, row3);
    }

};

template <>
void AmplitudesCalculatorM128<float>::realize_calculate() {
    ptrdiff_t matrix_size = 6;
    ptrdiff_t vector_dim = sizeof(__m128)/sizeof(float);
    std::unique_ptr<__m128[], decltype(free)*> vect_rec_coord{static_cast<__m128*>(aligned_alloc(sizeof(__m128), sizeof(__m128)*(n_rec/vector_dim)*3)), free};

    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
            for (ptrdiff_t i = 0; i < 3; ++i) {
                vect_rec_coord[(r_ind/vector_dim)*3+i] = _mm_set_ps(rec_coords_[(r_ind+3)*3+i], rec_coords_[(r_ind+2)*3+i], rec_coords_[(r_ind+1)*3+i], rec_coords_[(r_ind)*3+i]);
            }
        }

        __m128 coord_vec[3];
        __m128 G_P_vect[matrix_size];
        ALIGNED(16) float coords[vector_dim*3];
        ALIGNED(16) float coords_transposed[vector_dim*3];
        ALIGNED(16) float G_P[matrix_size*vector_dim];
        #pragma omp for collapse(2)
        for (ptrdiff_t i = 0; i < sources_count; ++i) {
            for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm_sub_ps(vect_rec_coord[(r_ind/vector_dim)*3+crd], _mm_set1_ps(sources_coords_[i*3+crd]));
                }

                __m128 dist = vect_calc_norm(coord_vec[0], coord_vec[1], coord_vec[2]);

                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm_div_ps(coord_vec[crd], dist);
                    _mm_storeu_ps(coords+crd*vector_dim, coord_vec[crd]);
                    G_P_vect[crd] = _mm_div_ps(_mm_mul_ps(coord_vec[crd], coord_vec[crd]), dist);
                }

                G_P_vect[3] = _mm_div_ps(_mm_mul_ps(coord_vec[0], coord_vec[1]), dist);
                G_P_vect[4] = _mm_div_ps(_mm_mul_ps(coord_vec[0], coord_vec[2]), dist);
                G_P_vect[5] = _mm_div_ps(_mm_mul_ps(coord_vec[1], coord_vec[2]), dist);

                transpose_coord_vect(coords, coords_transposed);

                for (ptrdiff_t m = 0; m < matrix_size; ++m) {
                    _mm_storeu_ps(G_P+m*vector_dim, G_P_vect[m]);
                    #pragma omp simd
                    for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {
                        for (ptrdiff_t rec_comp = 0; rec_comp < 3; ++rec_comp) {
                            amplitudes_[i*n_rec*3+(r_ind+v_s)*3+rec_comp] += G_P[m*vector_dim+v_s]*coords_transposed[v_s*3+rec_comp]*tensor_matrix_[m];
                        }
                    }
                }
            }
        }       
    }
    non_vector_calculate_amplitudes(n_rec-(n_rec%vector_dim), sources_coords_, rec_coords_, tensor_matrix_, sources_count, n_rec, amplitudes_);
}

template <>
void AmplitudesCalculatorM128<double>::realize_calculate() {
    ptrdiff_t matrix_size = 6;
    ptrdiff_t vector_dim = sizeof(__m128d)/sizeof(double);
    std::unique_ptr<__m128d[], decltype(free)*> vect_rec_coord{static_cast<__m128d*>(aligned_alloc(sizeof(__m128d), sizeof(__m128d)*(n_rec/vector_dim)*3)), free};

    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
            for (ptrdiff_t i = 0; i < 3; ++i) {
                vect_rec_coord[(r_ind/vector_dim)*3+i] = _mm_set_pd(rec_coords_[(r_ind+1)*3+i], rec_coords_[(r_ind)*3+i]);
            }
        }

        __m128d coord_vec[3];
        __m128d G_P_vect[matrix_size];
        ALIGNED(16) double coords[vector_dim*3];
        ALIGNED(16) double coords_transposed[vector_dim*3];
        ALIGNED(16) double G_P[matrix_size*vector_dim];
        #pragma omp for collapse(2)
        for (ptrdiff_t i = 0; i < sources_count; ++i) {
            for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm_sub_pd(vect_rec_coord[(r_ind/vector_dim)*3+crd], _mm_set1_pd(sources_coords_[i*3+crd]));
                }

                __m128d dist = vect_calc_norm(coord_vec[0], coord_vec[1], coord_vec[2]);

                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm_div_pd(coord_vec[crd], dist);
                    _mm_storeu_pd(coords+crd*vector_dim, coord_vec[crd]);
                    G_P_vect[crd] = _mm_div_pd(_mm_mul_pd(coord_vec[crd], coord_vec[crd]), dist);
                }

                G_P_vect[3] = _mm_div_pd(_mm_mul_pd(coord_vec[0], coord_vec[1]), dist);
                G_P_vect[4] = _mm_div_pd(_mm_mul_pd(coord_vec[0], coord_vec[2]), dist);
                G_P_vect[5] = _mm_div_pd(_mm_mul_pd(coord_vec[1], coord_vec[2]), dist);

                transpose_coord_vect(coords, coords_transposed);

                for (ptrdiff_t m = 0; m < matrix_size; ++m) {
                    _mm_storeu_pd(G_P+m*vector_dim, G_P_vect[m]);
                    #pragma omp simd
                    for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {
                        for (ptrdiff_t rec_comp = 0; rec_comp < 3; ++rec_comp) {
                            amplitudes_[i*n_rec*3+(r_ind+v_s)*3+rec_comp] += G_P[m*vector_dim+v_s]*coords_transposed[v_s*3+rec_comp]*tensor_matrix_[m];
                        }
                    }
                }
            }
        }         
    }
    non_vector_calculate_amplitudes(n_rec-(n_rec%vector_dim), sources_coords_, rec_coords_, tensor_matrix_, sources_count, n_rec, amplitudes_);
}

#endif /*_AMPLITUDES_CALCULATOR_M128*/