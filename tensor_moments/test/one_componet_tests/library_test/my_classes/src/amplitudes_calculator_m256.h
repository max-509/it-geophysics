#ifndef _AMPLITUDES_CALCULATOR_M256
#define _AMPLITUDES_CALCULATOR_M256

#include "amplitudes_calculator.h"
#include "array2D.h"

#include <x86intrin.h>
#include <functional>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <omp.h>

template <typename T>
class AmplitudesCalculatorM256 : public AmplitudesCalculator<T, AmplitudesCalculatorM256<T>> {
public:
	AmplitudesCalculatorM256(const Array2D<T> &sources_coords,
						 	  const Array2D<T> &rec_coords,
						 	  const T *tensor_matrix,
						 	  Array2D<T> &amplitudes) : 
		sources_coords_(sources_coords),
		rec_coords_(rec_coords),
		tensor_matrix_(tensor_matrix),
		amplitudes_(amplitudes) 
	{ }

	friend AmplitudesCalculator<T, AmplitudesCalculatorM256<T>>;

private:
	const Array2D<T> &sources_coords_;
	const Array2D<T> &rec_coords_;
	const T *tensor_matrix_;
	Array2D<T> &amplitudes_;

	void realize_calculate() {}

	inline __m256 vect_calc_norm(__m256 x, __m256 y, __m256 z) {
	    return _mm256_add_ps(_mm256_sqrt_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y)), _mm256_mul_ps(z, z))), _mm256_set1_ps(1e-36));
	}

	inline __m256d vect_calc_norm(__m256d x, __m256d y, __m256d z) {
	    return _mm256_add_pd(_mm256_sqrt_pd(_mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(x, x), _mm256_mul_pd(y, y)), _mm256_mul_pd(z, z))), _mm256_set1_pd(1e-300));
	}

};

template <>
void AmplitudesCalculatorM256<float>::realize_calculate() {
	ptrdiff_t n_rec = rec_coords_.get_y_dim();
    ptrdiff_t sources_count = sources_coords_.get_y_dim();
    ptrdiff_t matrix_size = 6;
    ptrdiff_t vector_dim = sizeof(__m256)/sizeof(float);
    ptrdiff_t ampl_dim = amplitudes_.get_x_dim();

    #pragma omp parallel
    {
        __m256 coord_vec[3];
        __m256 G_P_vect[matrix_size];
        #pragma omp for collapse(2)
        for (ptrdiff_t i = 0; i < sources_count; ++i) {
            for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm256_sub_ps(_mm256_set_ps(rec_coords_(r_ind+7, crd), rec_coords_(r_ind+6, crd), rec_coords_(r_ind+5, crd), rec_coords_(r_ind+4, crd),
                                                               rec_coords_(r_ind+3, crd), rec_coords_(r_ind+2, crd), rec_coords_(r_ind+1, crd), rec_coords_(r_ind+0, crd)
                                                               ), _mm256_set1_ps(sources_coords_(i, crd)));
                }

                __m256 dist = vect_calc_norm(coord_vec[0], coord_vec[1], coord_vec[2]);

                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm256_div_ps(coord_vec[crd], dist);
                    G_P_vect[crd] = _mm256_div_ps(_mm256_mul_ps(_mm256_set1_ps(tensor_matrix_[crd]), _mm256_mul_ps(coord_vec[2], _mm256_mul_ps(coord_vec[crd], coord_vec[crd]))), dist);
                }

                G_P_vect[3] = _mm256_div_ps(_mm256_mul_ps(_mm256_set1_ps(tensor_matrix_[3]), _mm256_mul_ps(coord_vec[2], _mm256_mul_ps(coord_vec[1], coord_vec[2]))), dist);
                G_P_vect[4] = _mm256_div_ps(_mm256_mul_ps(_mm256_set1_ps(tensor_matrix_[4]), _mm256_mul_ps(coord_vec[2], _mm256_mul_ps(coord_vec[0], coord_vec[2]))), dist);
                G_P_vect[5] = _mm256_div_ps(_mm256_mul_ps(_mm256_set1_ps(tensor_matrix_[5]), _mm256_mul_ps(coord_vec[2], _mm256_mul_ps(coord_vec[0], coord_vec[1]))), dist);

                for (ptrdiff_t m = 0; m < matrix_size; ++m) {
                    _mm256_storeu_ps(&amplitudes_(i, r_ind), _mm256_add_ps(_mm256_loadu_ps(&amplitudes_(i, r_ind)), G_P_vect[m]));
                }
            }
        }
    }
    non_vector_calculate_amplitudes(n_rec-(n_rec%vector_dim), sources_coords_, rec_coords_, tensor_matrix_, amplitudes_);
}

template <>
void AmplitudesCalculatorM256<double>::realize_calculate() {
	ptrdiff_t n_rec = rec_coords_.get_y_dim();
    ptrdiff_t sources_count = sources_coords_.get_y_dim();
    ptrdiff_t matrix_size = 6;
    ptrdiff_t vector_dim = sizeof(__m256d)/sizeof(double);
    ptrdiff_t ampl_dim = amplitudes_.get_x_dim(); 

    #pragma omp parallel
    {
        __m256d coord_vec[3];
        __m256d G_P_vect[matrix_size];
        #pragma omp for collapse(2)
        for (ptrdiff_t i = 0; i < sources_count; ++i) {
            for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm256_sub_pd(_mm256_set_pd(rec_coords_(r_ind+3, crd), rec_coords_(r_ind+2, crd), rec_coords_(r_ind+1, crd), rec_coords_(r_ind+0, crd)), _mm256_set1_pd(sources_coords_(i, crd)));
                }

                __m256d dist = vect_calc_norm(coord_vec[0], coord_vec[1], coord_vec[2]);

                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm256_div_pd(coord_vec[crd], dist);
                    G_P_vect[crd] = _mm256_div_pd(_mm256_mul_pd(_mm256_set1_pd(tensor_matrix_[crd]), _mm256_mul_pd(coord_vec[2], _mm256_mul_pd(coord_vec[crd], coord_vec[crd]))), dist);
                }

                G_P_vect[3] = _mm256_div_pd(_mm256_mul_pd(_mm256_set1_pd(tensor_matrix_[3]), _mm256_mul_pd(coord_vec[2], _mm256_mul_pd(coord_vec[1], coord_vec[2]))), dist);
                G_P_vect[4] = _mm256_div_pd(_mm256_mul_pd(_mm256_set1_pd(tensor_matrix_[4]), _mm256_mul_pd(coord_vec[2], _mm256_mul_pd(coord_vec[0], coord_vec[2]))), dist);
                G_P_vect[5] = _mm256_div_pd(_mm256_mul_pd(_mm256_set1_pd(tensor_matrix_[5]), _mm256_mul_pd(coord_vec[2], _mm256_mul_pd(coord_vec[0], coord_vec[1]))), dist);

                for (ptrdiff_t m = 0; m < matrix_size; ++m) {
                    _mm256_storeu_pd(&amplitudes_(i, r_ind), _mm256_add_pd(_mm256_loadu_pd(&amplitudes_(i, r_ind)), G_P_vect[m]));
                }
            }
        }         
    }
    non_vector_calculate_amplitudes(n_rec-(n_rec%vector_dim), sources_coords_, rec_coords_, tensor_matrix_, amplitudes_);
}

#endif /*_AMPLITUDES_CALCULATOR_M256*/