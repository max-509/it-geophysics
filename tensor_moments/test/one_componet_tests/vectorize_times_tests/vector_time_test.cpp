#ifdef VECT_NO
#include "src/amplitudes_calculator_non_vectors.h"
template <typename T>
using AmplitudesComputerType = AmplitudesCalculatorNonVectors<T>;
#endif

#ifdef VECT_128
#include "amplitudes_calculator_m128.h"
template <typename T>
using AmplitudesComputerType = AmplitudesCalculatorM128<T>;
#endif

#ifdef VECT_256
#include "amplitudes_calculator_m256.h"
template <typename T>
using AmplitudesComputerType = AmplitudesCalculatorM256<T>;
#endif

#ifdef VECT_512
#include "amplitudes_calculator_m512.h"
template <typename T>
using AmplitudesComputerType = AmplitudesCalculatorM512<T>;
#endif

#include <omp.h>
#include <iostream>
#include <cstdint>
#include <cstddef>
#include <cstdlib>

int main(int argc, char const *argv[]) {
	ptrdiff_t n_rec = 5000;
	ptrdiff_t n_sources = 50000;
	ptrdiff_t matrix_dim = 3;

	double *sources = new double[n_sources*3];
	double *rec = new double[n_rec*3];
	double *tensor_matrix = new double[6];
	double *amplitudes = new double[n_sources*n_rec*3];

	double t1 = omp_get_wtime();
	AmplitudesComputerType<double> computer(sources, rec, tensor_matrix, n_sources, n_rec, amplitudes);
	double t2 = omp_get_wtime();

	printf("Anti opt: %d\n", amplitudes[0]);
	printf("Srcs: %d, Recs: %d, Time: %f\n", n_sources, n_rec, t2-t1);

	delete [] sources;
	delete [] rec;
	delete [] tensor_matrix;
	delete [] amplitudes;
	return 0;
}