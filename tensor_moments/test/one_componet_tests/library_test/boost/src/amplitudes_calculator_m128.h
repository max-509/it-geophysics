#ifndef _AMPLITUDES_CALCULATOR_M128
#define _AMPLITUDES_CALCULATOR_M128

// #define BOOST_MULTI_ARRAY_NO_GENERATORS
#define BOOST_DISABLE_ASSERTS

#include "amplitudes_calculator.h"

#include <boost/multi_array.hpp>
#include <boost/array.hpp>
#include <x86intrin.h>
#include <functional>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <omp.h>

using Array1D_ind = typename boost::array<size_t, 1>;
using Array2D_ind = typename boost::array<size_t, 2>;

template <typename T>
class AmplitudesCalculatorM128 : public AmplitudesCalculator<T, AmplitudesCalculatorM128<T>> {
public:
    using Array1D = boost::multi_array_ref<T, 1>;
    using Array2D = boost::multi_array_ref<T, 2>;
    
	AmplitudesCalculatorM128(const Array2D &sources_coords,
						 	  const Array2D &rec_coords,
						 	  const Array1D &tensor_matrix,
						 	  Array2D &amplitudes) : 
		sources_coords_(sources_coords),
		rec_coords_(rec_coords),
		tensor_matrix_(tensor_matrix),
		amplitudes_(amplitudes) 
	{ }

	friend AmplitudesCalculator<T, AmplitudesCalculatorM128<T>>;

private:
	const Array2D &sources_coords_;
	const Array2D &rec_coords_;
	const Array1D &tensor_matrix_;
	Array2D &amplitudes_;

	void realize_calculate();

	inline __m128 vect_calc_norm(__m128 x, __m128 y, __m128 z) {
	    return _mm_add_ps(_mm_sqrt_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(x, x), _mm_mul_ps(y, y)), _mm_mul_ps(z, z))), _mm_set1_ps(1e-36));
	}

	inline __m128d vect_calc_norm(__m128d x, __m128d y, __m128d z) {
	    return _mm_add_pd(_mm_sqrt_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y)), _mm_mul_pd(z, z))), _mm_set1_pd(1e-300));
	}

};

template <>
void AmplitudesCalculatorM128<float>::realize_calculate() {
	size_t n_rec = rec_coords_.shape()[0];
    size_t sources_count = sources_coords_.shape()[0];
    size_t matrix_size = tensor_matrix_.shape()[0];
    size_t vector_dim = sizeof(__m128)/sizeof(float);

    #pragma omp parallel
    {
       __m128 coord_vec[3];
        __m128 G_P_vect[matrix_size];
        #pragma omp for collapse(2)
        for (size_t i = 0; i < sources_count; ++i) {
            for (size_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (size_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm_sub_ps(_mm_set_ps(rec_coords_(Array2D_ind{{r_ind+3, crd}}), rec_coords_(Array2D_ind{{r_ind+2, crd}}), 
                                                            rec_coords_(Array2D_ind{{r_ind+1, crd}}), rec_coords_(Array2D_ind{{r_ind, crd}})), _mm_set1_ps(sources_coords_(Array2D_ind{{i, crd}})));
                }

                __m128 dist = vect_calc_norm(coord_vec[0], coord_vec[1], coord_vec[2]);

                for (size_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm_div_ps(coord_vec[crd], dist);
                    G_P_vect[crd] = _mm_div_ps(_mm_mul_ps(_mm_set1_ps(tensor_matrix_[crd]), _mm_mul_ps(coord_vec[2], _mm_mul_ps(coord_vec[crd], coord_vec[crd]))), dist);
                }

                G_P_vect[3] = _mm_div_ps(_mm_mul_ps(_mm_set1_ps(tensor_matrix_[3]), _mm_mul_ps(coord_vec[2], _mm_mul_ps(coord_vec[1], coord_vec[2]))), dist);
                G_P_vect[4] = _mm_div_ps(_mm_mul_ps(_mm_set1_ps(tensor_matrix_[4]), _mm_mul_ps(coord_vec[2], _mm_mul_ps(coord_vec[0], coord_vec[2]))), dist);
                G_P_vect[5] = _mm_div_ps(_mm_mul_ps(_mm_set1_ps(tensor_matrix_[5]), _mm_mul_ps(coord_vec[2], _mm_mul_ps(coord_vec[0], coord_vec[1]))), dist);

                for (size_t m = 0; m < matrix_size; ++m) {
                    _mm_storeu_ps(&amplitudes_(Array2D_ind{{i, r_ind}}), _mm_add_ps(_mm_loadu_ps(&amplitudes_(Array2D_ind{{i, r_ind}})), G_P_vect[m]));
                }
            }
        }       
    }
    non_vector_calculate_amplitudes(n_rec-(n_rec%vector_dim), sources_coords_, rec_coords_, tensor_matrix_, amplitudes_);
}

template <>
void AmplitudesCalculatorM128<double>::realize_calculate() {
	size_t n_rec = rec_coords_.shape()[0];
    size_t sources_count = sources_coords_.shape()[0];
    size_t matrix_size = tensor_matrix_.shape()[0];
    size_t vector_dim = sizeof(__m128d)/sizeof(double);

    #pragma omp parallel
    {
        __m128d coord_vec[3];
        __m128d G_P_vect[matrix_size];
        double coords[vector_dim*3];
        double G_P[matrix_size*vector_dim];
        #pragma omp for collapse(2)
        for (size_t i = 0; i < sources_count; ++i) {
            for (size_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (size_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm_sub_pd(_mm_set_pd(rec_coords_(Array2D_ind{{r_ind+1, crd}}), rec_coords_(Array2D_ind{{r_ind+1, crd}})), _mm_set1_pd(sources_coords_(Array2D_ind{{i, crd}})));
                }

                __m128d dist = vect_calc_norm(coord_vec[0], coord_vec[1], coord_vec[2]);

                for (size_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm_div_pd(coord_vec[crd], dist);
                    G_P_vect[crd] = _mm_div_pd(_mm_mul_pd(_mm_set1_pd(tensor_matrix_[crd]), _mm_mul_pd(coord_vec[2], _mm_mul_pd(coord_vec[crd], coord_vec[crd]))), dist);
                }

                G_P_vect[3] = _mm_div_pd(_mm_mul_pd(_mm_set1_pd(tensor_matrix_[3]), _mm_mul_pd(coord_vec[2], _mm_mul_pd(coord_vec[1], coord_vec[2]))), dist);
                G_P_vect[4] = _mm_div_pd(_mm_mul_pd(_mm_set1_pd(tensor_matrix_[4]), _mm_mul_pd(coord_vec[2], _mm_mul_pd(coord_vec[0], coord_vec[2]))), dist);
                G_P_vect[5] = _mm_div_pd(_mm_mul_pd(_mm_set1_pd(tensor_matrix_[5]), _mm_mul_pd(coord_vec[2], _mm_mul_pd(coord_vec[0], coord_vec[1]))), dist);

                for (size_t m = 0; m < matrix_size; ++m) {
                    _mm_storeu_pd(&amplitudes_(Array2D_ind{{i, r_ind}}), _mm_add_pd(_mm_loadu_pd(&amplitudes_(Array2D_ind{{i, r_ind}})), G_P_vect[m]));
                }
            }
        }         
    }
    non_vector_calculate_amplitudes(n_rec-(n_rec%vector_dim), sources_coords_, rec_coords_, tensor_matrix_, amplitudes_);
}

#endif /*_AMPLITUDES_CALCULATOR_M128*/