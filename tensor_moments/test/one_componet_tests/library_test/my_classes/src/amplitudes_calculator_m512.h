#ifndef _AMPLITUDES_CALCULATOR_M512
#define _AMPLITUDES_CALCULATOR_M512

#include "amplitudes_calculator.h"
#include "array2D.h"

#include <x86intrin.h>
#include <functional>
#include <cstdlib>
#include <cmath>
#include <memory>

template <typename T>
class AmplitudesCalculatorM512 : public AmplitudesCalculator<T, AmplitudesCalculatorM512<T>> {
public:
	AmplitudesCalculatorM512(const Array2D<T> &sources_coords,
						 	  const Array2D<T> &rec_coords,
						 	  const T *tensor_matrix,
						 	  Array2D<T> &amplitudes) : 
		sources_coords_(sources_coords),
		rec_coords_(rec_coords),
		tensor_matrix_(tensor_matrix),
		amplitudes_(amplitudes) 
	{ }

	friend AmplitudesCalculator<T, AmplitudesCalculatorM512<T>>;

private:
	const Array2D<T> &sources_coords_;
	const Array2D<T> &rec_coords_;
	const T *tensor_matrix_;
	Array2D<T> &amplitudes_;

	void realize_calculate();

	inline __m512 vect_calc_norm(__m512 x, __m512 y, __m512 z) {
	    return _mm512_add_ps(_mm512_sqrt_ps(_mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(x, x), _mm512_mul_ps(y, y)), _mm512_mul_ps(z, z))), _mm512_set1_ps(1e-36));
	}

	inline __m512d vect_calc_norm(__m512d x, __m512d y, __m512d z) {
	    return _mm512_add_pd(_mm512_sqrt_pd(_mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(x, x), _mm512_mul_pd(y, y)), _mm512_mul_pd(z, z))), _mm512_set1_pd(1e-300));
	}

};

template <>
void AmplitudesCalculatorM512<float>::realize_calculate() {
	ptrdiff_t n_rec = rec_coords_.get_y_dim();
    ptrdiff_t sources_count = sources_coords_.get_y_dim();
    ptrdiff_t matrix_size = 6;
    ptrdiff_t vector_dim = sizeof(__m512)/sizeof(float);
    ptrdiff_t ampl_dim = amplitudes_.get_x_dim();
    std::unique_ptr<__m512[], decltype(free)*> vect_rec_coord{reinterpret_cast<__m512*>(aligned_alloc(sizeof(__m512), sizeof(__m512)*(n_rec/vector_dim)*3)), free};

    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
            for (ptrdiff_t i = 0; i < 3; ++i) {
                vect_rec_coord[(r_ind/vector_dim)*3+i] = _mm512_set_ps(rec_coords_(r_ind+15, i), rec_coords_(r_ind+14, i), rec_coords_(r_ind+13, i), rec_coords_(r_ind+12, i),
                                                                       rec_coords_(r_ind+11, i), rec_coords_(r_ind+10, i), rec_coords_(r_ind+9, i), rec_coords_(r_ind+8, i),
                                                                       rec_coords_(r_ind+7, i), rec_coords_(r_ind+6, i), rec_coords_(r_ind+5, i), rec_coords_(r_ind+4, i),
                                                                       rec_coords_(r_ind+3, i), rec_coords_(r_ind+2, i), rec_coords_(r_ind+1, i), rec_coords_(r_ind+0, i));
            }
        }

        __m512 coord_vec[3];
        __m512 G_P_vect[matrix_size];
        #pragma omp for collapse(2)
        for (ptrdiff_t i = 0; i < sources_count; ++i) {
            for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm512_sub_ps(vect_rec_coord[(r_ind/vector_dim)*3+crd], _mm512_set1_ps(sources_coords_(i, crd)));
                }

                __m512 dist = vect_calc_norm(coord_vec[0], coord_vec[1], coord_vec[2]);

                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm512_div_ps(coord_vec[crd], dist);
                    G_P_vect[crd] = _mm512_div_ps(_mm512_mul_ps(_mm512_set1_ps(tensor_matrix_[crd]), _mm512_mul_ps(coord_vec[2], _mm512_mul_ps(coord_vec[crd], coord_vec[crd]), dist)));
                }

                G_P_vect[3] = _mm512_div_ps(_mm512_mul_ps(_mm512_set1_ps(tensor_matrix_[3]), _mm512_mul_ps(coord_vec[2], _mm512_mul_ps(coord_vec[1], coord_vec[2]), dist)));
                G_P_vect[4] = _mm512_div_ps(_mm512_mul_ps(_mm512_set1_ps(tensor_matrix_[4]), _mm512_mul_ps(coord_vec[2], _mm512_mul_ps(coord_vec[0], coord_vec[2]), dist)));
                G_P_vect[5] = _mm512_div_ps(_mm512_mul_ps(_mm512_set1_ps(tensor_matrix_[5]), _mm512_mul_ps(coord_vec[2], _mm512_mul_ps(coord_vec[0], coord_vec[1]), dist)));

                for (ptrdiff_t m = 0; m < matrix_size; ++m) {
                    _mm512_storeu_ps(&amplitudes_(i, r_ind), _mm512_add_ps(_mm512_loadu_ps(&amplitudes_(i, r_ind)), G_P_vect[m]));
                }
            }
        }     
    }
    non_vector_calculate_amplitudes(n_rec-(n_rec%vector_dim), sources_coords_, rec_coords_, tensor_matrix_, amplitudes_);
}

template<>
void AmplitudesCalculatorM512<double>::realize_calculate() {
	ptrdiff_t n_rec = rec_coords_.get_y_dim();
    ptrdiff_t sources_count = sources_coords_.get_y_dim();
    ptrdiff_t matrix_size = 6;
    ptrdiff_t vector_dim = sizeof(__m512d)/sizeof(double);
    ptrdiff_t ampl_dim = amplitudes_.get_x_dim();
    std::unique_ptr<__m512d[], decltype(free)*> vect_rec_coord{reinterpret_cast<__m512d*>(aligned_alloc(sizeof(__m512d), sizeof(__m512d)*(n_rec/vector_dim)*3)), free};

    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
            for (ptrdiff_t i = 0; i < 3; ++i) {
                vect_rec_coord[(r_ind/vector_dim)*3+i] = _mm512_set_pd(rec_coords_(r_ind+7, i), rec_coords_(r_ind+6, i), rec_coords_(r_ind+5, i), rec_coords_(r_ind+4, i),
                                                                       rec_coords_(r_ind+3, i), rec_coords_(r_ind+2, i), rec_coords_(r_ind+1, i), rec_coords_(r_ind+0, i));
            }
        }

        __m512d coord_vec[3];
        __m512d G_P_vect[matrix_size];
        #pragma omp for collapse(2)
        for (ptrdiff_t i = 0; i < sources_count; ++i) {
            for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm512_sub_pd(vect_rec_coord[(r_ind/vector_dim)*3+crd], _mm512_set1_pd(sources_coords_(i, crd)));
                }

                __m512d dist = vect_calc_norm(coord_vec[0], coord_vec[1], coord_vec[2]);

                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vec[crd] = _mm512_div_pd(coord_vec[crd], dist);
                    G_P_vect[crd] = _mm512_div_pd(_mm512_mul_pd(_mm512_set1_pd(tensor_matrix_[crd]), _mm5_mul_pd(coord_vec[2], _mm512_mul_pd(coord_vec[crd], coord_vec[crd]))), dist);
                }

                G_P_vect[3] = _mm512_div_pd(_mm512_mul_pd(_mm512_set1_pd(tensor_matrix_[3]), _mm512_mul_pd(coord_vec[2], _mm512_mul_pd(coord_vec[1], coord_vec[2]))), dist);
                G_P_vect[4] = _mm512_div_pd(_mm512_mul_pd(_mm512_set1_pd(tensor_matrix_[4]), _mm512_mul_pd(coord_vec[2], _mm512_mul_pd(coord_vec[0], coord_vec[2]))), dist);
                G_P_vect[5] = _mm512_div_pd(_mm512_mul_pd(_mm512_set1_pd(tensor_matrix_[5]), _mm512_mul_pd(coord_vec[2], _mm512_mul_pd(coord_vec[0], coord_vec[1]))), dist);

                for (ptrdiff_t m = 0; m < matrix_size; ++m) {
                    _mm512_storeu_pd(&amplitudes_(i, r_ind), _mm512_add_pd(_mm512_loadu_pd(&amplitudes_(i, r_ind)), G_P_vect[m]));
                }
            }
        }      
    }
    non_vector_calculate_amplitudes(n_rec-(n_rec%vector_dim), sources_coords_, rec_coords_, tensor_matrix_, amplitudes_);
}

#endif /*_AMPLITUDES_CALCULATOR_M512*/