#ifndef _COH_SUMM
#define _COH_SUMM

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

#include <cmath>
#include <omp.h>
#include <algorithm>
#include <cstddef>

template <typename T, typename R>
void compute(const T *data, const T *rec_coords, const T *sources_coords, 
             const R *sources_times, const T *tensor_matrix, ptrdiff_t n_samples, 
             ptrdiff_t sources_count, ptrdiff_t n_rec, T *result_data) {

    T *amplitudes = new T[sources_count*n_rec]();
    AmplitudesComputerType<T> computer(sources_coords, rec_coords, tensor_matrix, sources_count, n_rec, amplitudes);
    computer.calculate();

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (ptrdiff_t i = 0; i < sources_count; ++i) {
            for (ptrdiff_t r_ind = 0; r_ind < n_rec; ++r_ind) {
                ptrdiff_t ind = sources_times[i*n_rec+r_ind];
                for (ptrdiff_t t = 0; t < n_samples-ind; ++t) {
                    result_data[i*n_samples+t] += data[r_ind*n_samples+ind+t]*amplitudes[i*n_rec+r_ind];
                }
            }
        }
    }
    delete [] amplitudes;
}


#endif /*_COH_SUMM*/