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

	void realize_calculate() {}

	inline __m256 vect_calc_norm(__m256 x, __m256 y, __m256 z) {
	    return _mm256_add_ps(_mm256_sqrt_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y)), _mm256_mul_ps(z, z))), _mm256_set1_ps(1e-36));
	}

	inline __m256d vect_calc_norm(__m256d x, __m256d y, __m256d z) {
	    return _mm256_add_pd(_mm256_sqrt_pd(_mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(x, x), _mm256_mul_pd(y, y)), _mm256_mul_pd(z, z))), _mm256_set1_pd(1e-300));
	}

    inline void transpose_coord_vect(const float *src_vect, float *dest_arr) {
        ptrdiff_t vector_dim = sizeof(__m256)/sizeof(float);
        ALIGNED(32) int64_t indeces[vector_dim];

        for (ptrdiff_t crd = 0; crd < 3; ++crd) {
            #pragma omp simd
            for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {
                indeces[v_s] = vector_dim*((crd*vector_dim+v_s)%3)+(crd*vector_dim+v_s)/3;
            }
            _mm256_store_ps(dest_arr+crd*vector_dim, _mm256_setr_m128(_mm256_i64gather_ps(src_vect, _mm256_load_si256(indeces), 1), _mm256_i64gather_ps(src_vect, _mm256_load_si256(indeces+vector_dim/2), 1)));
        }
    }

    inline void transpose_coord_vect(const double *src_vect, double *dest_vect) {
        ptrdiff_t vector_dim = sizeof(__m256d)/sizeof(double);
        ALIGNED(32) int64_t indeces[vector_dim];

        for (ptrdiff_t crd = 0; crd < 3; ++crd) {
            #pragma omp simd
            for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {
                indeces[v_s] = vector_dim*((crd*vector_dim+v_s)%3)+(crd*vector_dim+v_s)/3;
            }
            _mm256_store_pd(dest_arr+crd*vector_dim, _mm256_i64gather_pd(src_vect, _mm256_load_si256(indeces), 1));
        }   
    }

};

template <>
void AmplitudesCalculatorM256<float>::realize_calculate() {
    ptrdiff_t matrix_size = 6;
    ptrdiff_t vector_dim = sizeof(__m256)/sizeof(float);
    std::unique_ptr<__m256[], decltype(free)*> vect_rec_coord{static_cast<__m256*>(aligned_alloc(sizeof(__m256), sizeof(__m256)*(n_rec/vector_dim)*3)), free};

    #pragma omp parallel
    {
        ALIGNED(32) int64_t indeces[vector_dim];
        #pragma omp for collapse(2)
        for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
            for (ptrdiff_t i = 0; i < 3; ++i) {
                #pragma omp simd
                for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {
                    indeces[v_s] = (r_ind+v_s)*3+i;
                }
                vect_rec_coord[(r_ind/vector_dim)*3+i] = _mm256_setr_m128(_mm256_i32gather_ps(rec_coords_, _mm256_load_si256(indeces), 1), _mm256_i64gather_ps(rec_coords_, _mm256_load_si256(indeces+vector_dim/2), 1));
                // vect_rec_coord[(r_ind/vector_dim)*3+i] = _mm256_set_ps(rec_coords_[(r_ind+7)*3+i], rec_coords_[(r_ind+6)*3+i], rec_coords_[(r_ind+5)*3+i], rec_coords_[(r_ind+4)*3+i],
                //                                                        rec_coords_[(r_ind+3)*3+i], rec_coords_[(r_ind+2)*3+i], rec_coords_[(r_ind+1)*3+i], rec_coords_[(r_ind)*3+i]);
            }
        }

        __m256 coord_vec[3];
        __m256 G_P_vect[matrix_size];
        ALIGNED(32) float coords[vector_dim*3];
        ALIGNED(32) float coords_transposed[vector_dim*3];
        ALIGNED(32) float G_P[matrix_size*vector_dim];
        #pragma omp for collapse(2)
        for (ptrdiff_t i = 0; i < sources_count; ++i) {
            for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm256_sub_ps(vect_rec_coord[(r_ind/vector_dim)*3+crd], _mm256_set1_ps(sources_coords_[i*3+crd]));
                }

                __m256 dist = vect_calc_norm(coord_vec[0], coord_vec[1], coord_vec[2]);

                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm256_div_ps(coord_vec[crd], dist);
                    _mm256_store_ps(coords+crd*vector_dim, coord_vec[crd]);
                    G_P_vect[crd] = _mm256_div_ps(_mm256_mul_ps(coord_vec[crd], coord_vec[crd]), dist);
                }

                G_P_vect[3] = _mm256_div_ps(_mm256_mul_ps(coord_vec[0], coord_vec[1]), dist);
                G_P_vect[4] = _mm256_div_ps(_mm256_mul_ps(coord_vec[0], coord_vec[2]), dist);
                G_P_vect[5] = _mm256_div_ps(_mm256_mul_ps(coord_vec[1], coord_vec[2]), dist);

                transpose_coord_vect(coords, coords_transposed);

                for (ptrdiff_t m = 0; m < matrix_size; ++m) {
                    _mm256_store_ps(G_P+m*vector_dim, G_P_vect[m]);
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
void AmplitudesCalculatorM256<double>::realize_calculate() {
    ptrdiff_t matrix_size = 6;
    ptrdiff_t vector_dim = sizeof(__m256d)/sizeof(double);
    std::unique_ptr<__m256d[], decltype(free)*> vect_rec_coord{static_cast<__m256d*>(aligned_alloc(sizeof(__m256d), sizeof(__m256d)*(n_rec/vector_dim)*3)), free};

    #pragma omp parallel
    {
        ALIGNED(32) int64_t indeces[vector_dim];
        #pragma omp for collapse(2)
        for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
            for (ptrdiff_t i = 0; i < 3; ++i) {
                #pragma omp simd
                for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {
                    indeces[v_s] = (r_ind+v_s)*3+i;
                }
                vect_rec_coord[(r_ind/vector_dim)*3+i] = _mm256_i64gather_pd(rec_coords_, _mm256_load_si256(indeces), 1);
                // vect_rec_coord[(r_ind/vector_dim)*3+i] = _mm256_set_pd(rec_coords_[(r_ind+3)*3+i], rec_coords_[(r_ind+2)*3+i], rec_coords_[(r_ind+1)*3+i], rec_coords_[(r_ind)*3+i]);
            }
        }

        __m256d coord_vec[3];
        __m256d G_P_vect[matrix_size];
        ALIGNED(32) double coords[vector_dim*3];
        ALIGNED(32) double coords_transposed[vector_dim*3];
        ALIGNED(32) double G_P[matrix_size*vector_dim];
        #pragma omp for collapse(2)
        for (ptrdiff_t i = 0; i < sources_count; ++i) {
            for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm256_sub_pd(vect_rec_coord[(r_ind/vector_dim)*3+crd], _mm256_set1_pd(sources_coords_(i, crd)));
                }

                __m256d dist = vect_calc_dist(coord_vec[0], coord_vec[1], coord_vec[2]);

                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm256_div_pd(coord_vec[crd], dist);
                    _mm256_store_pd(coords+crd*vector_dim, coord_vec[crd]);
                    G_P_vect[crd] = _mm256_div_pd(_mm256_mul_pd(coord_vec[crd], coord_vec[crd]), dist);
                }

                G_P_vect[3] = _mm256_div_pd(_mm256_mul_pd(coord_vec[0], coord_vec[1]), dist);
                G_P_vect[4] = _mm256_div_pd(_mm256_mul_pd(coord_vec[0], coord_vec[2]), dist);
                G_P_vect[5] = _mm256_div_pd(_mm256_mul_pd(coord_vec[1], coord_vec[2]), dist);

                transpose_coord_vect(coords, coords_transposed);

                for (ptrdiff_t m = 0; m < matrix_size; ++m) {
                    _mm256_store_pd(G_P+m*vector_dim, G_P_vect[m]);
                    #pragma omp simd
                    for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {
                        #pragma omp simd
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

#endif /*_AMPLITUDES_CALCULATOR_M256*/