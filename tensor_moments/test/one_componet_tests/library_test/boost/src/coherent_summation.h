#ifndef _COH_SUMM
#define _COH_SUMM

#define BOOST_DISABLE_ASSERTS
#define BOOST_UBLAS_NDEBUG
#define ARMA_NO_DEBUG

#include <boost/multi_array.hpp>
#include <boost/array.hpp>
#include <cmath>
#include <omp.h>
#include <algorithm>
#include <cstddef>

/*Selection of SIMD instructions*/
#ifdef __AVX512F__
#include "amplitudes_calculator_m512.h"
template <typename T>
using AmplitudesComputerType = AmplitudesCalculatorM512<T>;
#elif __AVX__
#include "amplitudes_calculator_m256.h"
template <typename T>
using AmplitudesComputerType = AmplitudesCalculatorM256<T>;
#elif __SSE2__
#include "amplitudes_calculator_m128.h"
template <typename T>
using AmplitudesComputerType = AmplitudesCalculatorM128<T>;
#else
#include "amplitudes_calculator_non_vectors.h"
template <typename T>
using AmplitudesComputerType = AmplitudesCalculatorNonVectors<T>;
#endif /*End selection of SIMD instructions*/

using Array1D_ind = boost::array<size_t, 1>;
using Array2D_ind = boost::array<size_t, 2>;

template <typename T>
using Array1D = boost::multi_array_ref<T, 1>;
template <typename T>
using Array2D = boost::multi_array_ref<T, 2>;

template <typename T, typename R>
void compute(const Array2D<T> &data, const Array2D<T> &rec_coords, const Array2D<T> &sources_coords, 
             const Array2D<R> &sources_times, const Array1D<T> &tensor_matrix, T *result_data) {
	size_t rec_block_size = 60;
    size_t samples_block_size = 400;
    size_t n_samples = data.shape()[1];
    size_t n_rec = data.shape()[0];
    size_t sources_count = sources_coords.shape()[0];

    T *amplitudes_buf = new T[sources_count*n_rec]();
    Array2D<T> amplitudes{amplitudes_buf, boost::extents[sources_count][n_rec], boost::c_storage_order()};
    AmplitudesComputerType<T> computer(sources_coords, rec_coords, tensor_matrix, amplitudes);
    computer.calculate();

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (size_t bl_t = 0; bl_t < n_samples; bl_t += samples_block_size) {
            for (size_t bl_r = 0; bl_r < n_rec; bl_r += rec_block_size) {
                for (size_t i = 0; i < sources_count; ++i) {
                    for (size_t r_ind = bl_r; r_ind < std::min(bl_r+rec_block_size, n_rec); ++r_ind) {
                        size_t ind = sources_times(Array2D_ind{{i, r_ind}});
                        for (size_t t = bl_t; t < std::min(bl_t+samples_block_size, n_samples-ind); ++t) {
                            result_data[i*n_samples+t] += data(Array2D_ind{{r_ind, ind+t}})*amplitudes(Array2D_ind{{i, r_ind}});
                        }
                    }
                }
            }
        }
    }
    delete [] amplitudes_buf;
}

#endif /*_COH_SUMM*/