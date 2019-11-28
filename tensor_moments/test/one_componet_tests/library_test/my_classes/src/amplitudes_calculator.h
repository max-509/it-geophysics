#ifndef _AMPLITUDES_CALCULATOR
#define _AMPLITUDES_CALCULATOR

#include "array2D.h"

#include <cstddef>
#include <cmath>

template <typename T, typename Realization>
class AmplitudesCalculator {
public:

	void calculate() {
		static_cast<Realization*>(this)->realize_calculate(); 
	}

	friend Realization;

private:

	void non_vector_calculate_amplitudes(ptrdiff_t ind_first_rec, const Array2D<T> &sources_coords, const Array2D<T> &rec_coords, const T *tensor_matrix, Array2D<T> &amplitudes) {
	    ptrdiff_t n_rec = rec_coords.get_y_dim();
	    ptrdiff_t sources_count = sources_coords.get_y_dim();
	    ptrdiff_t matrix_size = 6;

	    #pragma omp parallel
	    {
	        T coord_vect[3];
	        T G_P[matrix_size];
	        #pragma omp for collapse(2)
	        for (ptrdiff_t i = 0; i < sources_count; ++i) {
	            for (ptrdiff_t r_ind = ind_first_rec; r_ind < n_rec; ++r_ind) {
	                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
	                    coord_vect[crd] = rec_coords(r_ind, crd)-sources_coords(i, crd);
	                }
	                T dist = calc_norm(coord_vect[0], coord_vect[1], coord_vect[2]);
	                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
	                    coord_vect[crd] /= dist;    
	                    G_P[crd] = coord_vect[crd]*coord_vect[crd]/dist;
	                }

	                G_P[0] *= coord_vect[2];
	                G_P[1] *= coord_vect[2];
	                G_P[2] *= coord_vect[2];
	                G_P[3] = 2*coord_vect[2]*coord_vect[1]*coord_vect[2]/dist;
	                G_P[4] = 2*coord_vect[2]*coord_vect[0]*coord_vect[2]/dist;
	                G_P[5] = 2*coord_vect[2]*coord_vect[0]*coord_vect[1]/dist;
	                

	                for (ptrdiff_t m = 0; m < matrix_size; ++m) {
                        amplitudes(i, r_ind) += (G_P[m])*tensor_matrix[m];
	                }
	            }
	        }         
	    }
	    T max_ampl = amplitudes(0, 0);
	    #pragma omp parallel for simd collapse(2) reduction(max:max_ampl)
	    for (ptrdiff_t i = 0; i < amplitudes.get_y_dim(); ++i) {
	        for (ptrdiff_t j = 0; j < amplitudes.get_x_dim(); ++j) {
                max_ampl = (max_ampl > amplitudes(i, j)) ? max_ampl : amplitudes(i, j);
	        } 
	    }
	    max_ampl += 1e-30;

	    #pragma omp parallel for simd collapse(2)
	    for (ptrdiff_t i = 0; i < amplitudes.get_y_dim(); ++i) {
	        for (ptrdiff_t j = 0; j < amplitudes.get_x_dim(); ++j) {
                amplitudes(i, j) /= max_ampl;
	        } 
	    }


	}

	#pragma omp declare simd
	inline T calc_norm(T x, T y, T z) {
	    return sqrt(x*x+y*y+z*z)+1e-30;
	}
};

#endif /*_AMPLITUDES_CALCULATOR*/