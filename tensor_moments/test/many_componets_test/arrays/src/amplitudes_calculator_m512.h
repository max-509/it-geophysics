#ifndef _AMPLITUDES_CALCULATOR_M512
#define _AMPLITUDES_CALCULATOR_M512

#include "amplitudes_calculator.h"

#include <x86intrin.h>
#include <functional>
#include <cstdlib>
#include <cmath>
#include <memory>

#define ALIGNED(n) __attribute__((aligned(n)))

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

	void realize_calculate();

	inline __m512 vect_calc_norm(__m512 x, __m512 y, __m512 z) {
	    return _mm512_add_ps(_mm512_sqrt_ps(_mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(x, x), _mm512_mul_ps(y, y)), _mm512_mul_ps(z, z))), _mm512_set1_ps(1e-36));
	}

	inline __m512d vect_calc_norm(__m512d x, __m512d y, __m512d z) {
	    return _mm512_add_pd(_mm512_sqrt_pd(_mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(x, x), _mm512_mul_pd(y, y)), _mm512_mul_pd(z, z))), _mm512_set1_pd(1e-300));
	}

    inline void transpose_coord_vect(const float *src_vect, float *dest_arr) {
        ptrdiff_t vector_dim = sizeof(__m512)/sizeof(float);
        ALIGNED(64) int64_t indeces[vector_dim];

        for (ptrdiff_t crd = 0; crd < 3; ++crd) {
            #pragma omp simd
            for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {
                indeces[v_s] = vector_dim*((crd*vector_dim+v_s)%3)+(crd*vector_dim+v_s)/3;
            }
            _mm256_store_ps(dest_arr+crd*vector_dim, _mm512_insertf32x8(_mm512_castps256_ps512(_mm512_i64gather_ps(_mm512_load_si512(indeces), src_vect, 1)), _mm512_i64gather_ps(_mm512_load_si512(indeces+vector_dim/2), src_vect, 1), 1));
        }
    }

    inline void transpose_coord_vect(const double *src_vect, double *dest_vect) {
        ptrdiff_t vector_dim = sizeof(__m512d)/sizeof(double);
        ALIGNED(64) int64_t indeces[vector_dim];

        for (ptrdiff_t crd = 0; crd < 3; ++crd) {
            #pragma omp simd
            for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {
                indeces[v_s] = vector_dim*((crd*vector_dim+v_s)%3)+(crd*vector_dim+v_s)/3;
            }
            _mm512_store_pd(dest_arr+crd*vector_dim, _mm512_i64gather_pd(_mm512_load_si512(indeces), src_vect, 1));
        }   
    }

};

template <>
void AmplitudesCalculatorM512<float>::realize_calculate() {
    ptrdiff_t matrix_size = 6;
    ptrdiff_t vector_dim = sizeof(__m512)/sizeof(float);
    std::unique_ptr<__m512[], decltype(free)*> vect_rec_coord{static_cast<__m512*>(aligned_alloc(sizeof(__m512), sizeof(__m512)*(n_rec/vector_dim)*3)), free};

    #pragma omp parallel
    {
        ALIGNED(64) int64_t indeces[vector_dim];
        #pragma omp for collapse(2)
        for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
            for (ptrdiff_t i = 0; i < 3; ++i) {
                #pragma omp simd
                for (v_s = 0; v_s < vector_dim; ++v_s) {
                    indeces[v_s] = (r_ind+v_s)*3+i;
                }
                vect_rec_coord[(r_ind/vector_dim)*3+i] = _mm512_insertf32x8(_mm512_castps256_ps512(_mm512_i64gather_ps(_mm512_load_si512(indeces), rec_coords_, 1)), _mm512_i64gather_ps(_mm512_load_si512(indeces+vector_dim/2), rec_coords_, 1), 1);
                // vect_rec_coord[(r_ind/vector_dim)*3+i] = _mm512_set_ps(rec_coords_[(r_ind+15)*3+i], rec_coords_[(r_ind+14)*3+i], rec_coords_[(r_ind+13)*3+i], rec_coords_[(r_ind+12)*3+i],
                //                                                        rec_coords_[(r_ind+11)*3+i], rec_coords_[(r_ind+10)*3+i], rec_coords_[(r_ind+9)*3+i], rec_coords_[(r_ind+8)*3+i],
                //                                                        rec_coords_[(r_ind+7)*3+i], rec_coords_[(r_ind+6)*3+i], rec_coords_[(r_ind+5)*3+i], rec_coords_[(r_ind+4)*3+i],
                //                                                        rec_coords_[(r_ind+3)*3+i], rec_coords_[(r_ind+2)*3+i], rec_coords_[(r_ind+1)*3+i], rec_coords_[(r_ind)*3+i]);
            }
        }

        __m512 coord_vec[3];
        __m512 G_P_vect[matrix_size];
        ALIGNED(64) float coords[vector_dim*3];
        ALIGNED(64) float coords_transposed[vector_dim*3];
        ALIGNED(64) float G_P[matrix_size*vector_dim];
        #pragma omp for collapse(2)
        for (ptrdiff_t i = 0; i < sources_count; ++i) {
            for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm512_sub_ps(vect_rec_coord[(r_ind/vector_dim)*3+crd], _mm512_set1_ps(sources_coords_[i*3+crd]));
                }

                __m512 dist = vect_calc_norm(coord_vec[0], coord_vec[1], coord_vec[2]);

                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm512_div_ps(coord_vec[crd], dist);
                    _mm512_store_ps(coords+crd*vector_dim, coord_vec[crd]);
                    G_P_vect[crd] = _mm512_div_ps(_mm512_mul_ps(coord_vec[crd], coord_vec[crd]), dist);
                }

                G_P_vect[3] = _mm512_div_ps(_mm512_mul_ps(coord_vec[0], coord_vec[1]), dist);
                G_P_vect[4] = _mm512_div_ps(_mm512_mul_ps(coord_vec[0], coord_vec[2]), dist);
                G_P_vect[5] = _mm512_div_ps(_mm512_mul_ps(coord_vec[1], coord_vec[2]), dist);

                transpose_coord_vect(coords, coords_transposed);

                for (ptrdiff_t m = 0; m < matrix_size; ++m) {
                    _mm512_store_ps(G_P+m*vector_dim, G_P_vect[m]);
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

template<>
void AmplitudesCalculatorM512<double>::realize_calculate() {
    ptrdiff_t matrix_size = 6;
    ptrdiff_t vector_dim = sizeof(__m512d)/sizeof(double);
    std::unique_ptr<__m512d[], decltype(free)*> vect_rec_coord{static_cast<__m512d*>(aligned_alloc(sizeof(__m512d), sizeof(__m512d)*(n_rec/vector_dim)*3)), free};

    #pragma omp parallel
    {
        ALIGNED(64) int64_t indeces[vector_dim];
        #pragma omp for collapse(2)
        for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
            for (ptrdiff_t i = 0; i < 3; ++i) {
                #pragma omp simd
                for (v_s = 0; v_s < vector_dim; ++v_s) {
                    indeces[v_s] = (r_ind+v_s)*3+i;
                }
                vect_calc_norm[(r_ind/vector_dim)*3+i] = _mm512_i64gather_pd(_mm512_load_si512(indeces), rec_coords_, 1);
                // vect_rec_coord[(r_ind/vector_dim)*3+i] = _mm512_set_pd(rec_coords_[(r_ind+7)*3+i], rec_coords_[(r_ind+6)*3+i], rec_coords_[(r_ind+5)*3+i], rec_coords_[(r_ind+4)*3+i],
                //                                                        rec_coords_[(r_ind+3)*3+i], rec_coords_[(r_ind+2)*3+i], rec_coords_[(r_ind+1)*3+i], rec_coords_[(r_ind)*3+i]);
            }
        }

        __m512d coord_vec[3];
        __m512d G_P_vect[matrix_size];
        ALIGNED(64) double coords[vector_dim*3];
        ALIGNED(64) double coords_transposed[vector_dim*3];
        ALIGNED(64) double G_P[matrix_size*vector_dim];
        #pragma omp for collapse(2)
        for (ptrdiff_t i = 0; i < sources_count; ++i) {
            for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm512_sub_pd(vect_rec_coord[(r_ind/vector_dim)*3+crd], _mm512_set1_pd(sources_coords_[i*3+crd]));
                }

                __m512d dist = vect_calc_norm(coord_vec[0], coord_vec[1], coord_vec[2]);

                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm512_div_pd(coord_vec[crd], dist);
                    _mm512_store_pd(coords+crd*vector_dim, coord_vec[crd]);
                    G_P_vect[crd] = _mm512_div_pd(_mm512_mul_pd(coord_vec[crd], coord_vec[crd]), dist);
                }

                G_P_vect[3] = _mm512_div_pd(_mm512_mul_pd(coord_vec[0], coord_vec[1]), dist);
                G_P_vect[4] = _mm512_div_pd(_mm512_mul_pd(coord_vec[0], coord_vec[2]), dist);
                G_P_vect[5] = _mm512_div_pd(_mm512_mul_pd(coord_vec[1], coord_vec[2]), dist);

                transpose_coord_vect(coords, coords_transposed);

                for (ptrdiff_t m = 0; m < matrix_size; ++m) {
                    _mm512_store_pd(G_P+m*vector_dim, G_P_vect[m]);
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

#endif /*_AMPLITUDES_CALCULATOR_M512*/