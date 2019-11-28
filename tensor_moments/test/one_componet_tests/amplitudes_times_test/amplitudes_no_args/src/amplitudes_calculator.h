#ifndef _AMPLITUDES_CALCULATOR
#define _AMPLITUDES_CALCULATOR

#define EIGEN_NO_DEBUG

#include <Eigen/Dense>
#include <cstddef>
#include <cmath>

using namespace Eigen;

template <typename T>
using Tensor_t = Matrix<T, 6, 1>;
template <typename T>
using MapTensor_t = Map<const Tensor_t<T>, Aligned>;

template <typename T>
using Array2D = Matrix<T, Dynamic, Dynamic, RowMajor>;
template <typename T>
using MapArray2D = Map<const Array2D<T>, Aligned>;

template <typename T>
void calculate_amplitudes(const MapArray2D<T> &sources_coords, const MapArray2D<T> &rec_coords, const MapTensor_t<T> &tensor_matrix, Array2D<T> &amplitudes) {
    ptrdiff_t n_rec = rec_coords.rows();
    ptrdiff_t sources_count = sources_coords.rows();

    #pragma omp parallel
    {
        Matrix<T, 3, 1> coord_vect;
        Matrix<T, 6, 1> G_P;
        #pragma omp for collapse(2)
        for (ptrdiff_t i = 0; i < sources_count; ++i) {
            for (ptrdiff_t r_ind = 0; r_ind < n_rec; ++r_ind) {
                coord_vect.noalias() = rec_coords.row(r_ind)-sources_coords.row(i);
                T dist = coord_vect.norm() + 1e-30;
                coord_vect /= dist;
                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    G_P(crd) = coord_vect(crd)*coord_vect(crd);
                }

                G_P(3) = 2*coord_vect(1)*coord_vect(2);
                G_P(4) = 2*coord_vect(0)*coord_vect(2);
                G_P(5) = 2*coord_vect(0)*coord_vect(1);

                G_P *= (coord_vect(2)/dist);

                amplitudes(i, r_ind) = G_P.dot(tensor_matrix);
                
            }
        }         
    }

    // amplitudes /= (amplitudes.template lpNorm<Infinity>() + 1e-30);
}

#endif /*_AMPLITUDES_CALCULATOR*/