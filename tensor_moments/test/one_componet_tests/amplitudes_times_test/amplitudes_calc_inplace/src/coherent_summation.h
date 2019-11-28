#ifndef _COH_SUMM
#define _COH_SUMM

#define EIGEN_NO_DEBUG

#include <Eigen/Dense>
#include <cmath>
#include <omp.h>
#include <algorithm>
#include <cstddef>

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
using ResArrType = Map<Array2D<T>, Aligned>;

template <typename T, typename R>
void compute(const MapArray2D<T> &data, const MapArray2D<T> &rec_coords, const MapArray2D<T> &sources_coords, 
             const MapArray2D<R> &sources_times, const MapTensor_t<T> &tensor_matrix, ResArrType<T> &result_data) {
    ptrdiff_t n_rec = data.rows();
    ptrdiff_t n_samples = data.cols();
    ptrdiff_t sources_count = sources_coords.rows();

    #pragma omp parallel
    {
        Matrix<T, 3, 1> coord_vect;
        Matrix<T, 6, 1> G_P;
        T ampl_tmp = 0.;
        #pragma omp for schedule(dynamic)
            for (ptrdiff_t i = 0; i < sources_count; ++i) {
                for (ptrdiff_t r_ind = bl_r; r_ind < std::min(bl_r+rec_block_size, n_rec); ++r_ind) {
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

                    ampl_tmp = G_P.dot(tensor_matrix);
                    ptrdiff_t ind = sources_times(i, r_ind);

                    result_data.row(i).head(n_samples-ind).noalias() += ampl_tmp*data.row(r_ind).tail(n_samples-ind);
                }
            }
    }

}

#endif /*_COH_SUMM*/