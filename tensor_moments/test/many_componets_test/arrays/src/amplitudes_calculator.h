#ifndef _AMPLITUDES_CALCULATOR
#define _AMPLITUDES_CALCULATOR

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

	void non_vector_calculate_amplitudes(ptrdiff_t ind_first_rec, const T *sources_coords, const T *rec_coords, 
										 const T *tensor_matrix, ptrdiff_t sources_count, ptrdiff_t n_rec, T *amplitudes) {
	    ptrdiff_t matrix_size = 6;

	    #pragma omp parallel
	    {
	        T coord_vect[3];
	        T G_P[matrix_size];
	        #pragma omp for collapse(2)
	        for (ptrdiff_t i = 0; i < sources_count; ++i) {
	            for (ptrdiff_t r_ind = ind_first_rec; r_ind < n_rec; ++r_ind) {
	                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
	                    coord_vect[crd] = rec_coords[r_ind*3+crd]-sources_coords[i*3+crd];
	                }
	                T dist = calc_norm(coord_vect[0], coord_vect[1], coord_vect[2]);
	                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
	                    coord_vect[crd] /= dist;    
	                    G_P[crd] = coord_vect[crd]*coord_vect[crd]/dist;
	                }

	                G_P[3] = 2*coord_vect[1]*coord_vect[2]/dist;
	                G_P[4] = 2*coord_vect[0]*coord_vect[2]/dist;
	                G_P[5] = 2*coord_vect[0]*coord_vect[1]/dist;
	                
                	#pragma omp simd
	                for (ptrdiff_t rec_comp = 0; rec_comp < 3; ++rec_comp) {
	                	for (ptrdiff_t m = 0; m < matrix_size; ++m) {
                        	amplitudes[i*n_rec*3+r_ind*3+rec_comp] += (G_P[m]*coord_vect[rec_comp])*tensor_matrix[m];
	                	}
	                }
	            }
	        }         
	    }
	}

	#pragma omp declare simd
	inline T calc_norm(T x, T y, T z) {
	    return sqrt(x*x+y*y+z*z)+1e-30;
	}
};

#endif /*_AMPLITUDES_CALCULATOR*/