#ifndef _COH_SUMM
#define _COH_SUMM

#define EIGEN_NO_DEBUG

#include "amplitudes_calculator.h"

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

    Array2D<T> amplitudes{sources_count, n_rec*3};
    calculate_amplitudes(sources_coords, rec_coords, tensor_matrix, amplitudes);

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (ptrdiff_t i = 0; i < sources_count; ++i) {
            for (ptrdiff_t r_ind = 0; r_ind < n_rec; ++r_ind) {
                ptrdiff_t ind = sources_times(i, r_ind);
                for (ptrdiff_t comp_i = 0; comp_i < 3; ++comp_i) {
                    result_data.row(i).head(n_samples-ind).noalias() += amplitudes(i, r_ind*3+comp_i)*data.row(r_ind*3+comp_i).tail(n_samples-ind);
                }
            }
        }
    }

}

#endif /*_COH_SUMM*/