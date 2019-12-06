#ifndef _AMPLITUDES_CALCULATOR
#define _AMPLITUDES_CALCULATOR

#define BOOST_DISABLE_ASSERTS

#include <boost/multi_array.hpp>
#include <boost/array.hpp>
#include <cstddef>
#include <cmath>

template <typename T, typename Realization>
class AmplitudesCalculator {
public:

	using Array1D = typename boost::multi_array_ref<T, 1>;
	using Array2D = typename boost::multi_array_ref<T, 2>;
	using Array1D_ind = boost::array<size_t, 1>;
	using Array2D_ind = boost::array<size_t, 2>;

	void calculate() {
		static_cast<Realization*>(this)->realize_calculate(); 
	}

	friend Realization;

private:

	void non_vector_calculate_amplitudes(size_t ind_first_rec, const Array2D &sources_coords, const Array2D &rec_coords, const Array1D &tensor_matrix, Array2D &amplitudes) {
	    size_t n_rec = rec_coords.shape()[0];
	    size_t sources_count = sources_coords.shape()[0];
	    size_t matrix_size = tensor_matrix.shape()[0];

	    #pragma omp parallel
	    {
	        T coord_vect[3];
	        T G_P[matrix_size];
	        #pragma omp for collapse(2)
	        for (size_t i = 0; i < sources_count; ++i) {
	            for (size_t r_ind = ind_first_rec; r_ind < n_rec; ++r_ind) {
	                for (size_t crd = 0; crd < 3; ++crd) {
	                    coord_vect[crd] = rec_coords(Array2D_ind{{r_ind, crd}})-sources_coords(Array2D_ind{{i, crd}});
	                }
	                T dist = calc_norm(coord_vect[0], coord_vect[1], coord_vect[2]);
	                for (size_t crd = 0; crd < 3; ++crd) {
	                    coord_vect[crd] /= dist;    
	                    G_P[crd] = coord_vect[crd]*coord_vect[crd]/dist;
	                }

	                G_P[0] *= coord_vect[2];
	                G_P[1] *= coord_vect[2];
	                G_P[2] *= coord_vect[2];
	                G_P[3] = 2*coord_vect[2]*coord_vect[1]*coord_vect[2]/dist;
	                G_P[4] = 2*coord_vect[2]*coord_vect[0]*coord_vect[2]/dist;
	                G_P[5] = 2*coord_vect[2]*coord_vect[0]*coord_vect[1]/dist;
	                
	                #pragma omp simd
	                for (size_t m = 0; m < matrix_size; ++m) {
                        amplitudes(Array2D_ind{{i, r_ind}}) += (G_P[m])*tensor_matrix[m];
	                }
	            }
	        }         
	    }

	}

	#pragma omp declare simd
	inline T calc_norm(T x, T y, T z) {
	    return std::sqrt(x*x+y*y+z*z)+1e-30;
	}
};

#endif /*_AMPLITUDES_CALCULATOR*/