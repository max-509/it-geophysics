#ifdef VECT_NO
#include "src/amplitudes_calculator_non_vectors.h"
template <typename T>
using AmplitudesComputerType = AmplitudesCalculatorNonVectors<T>;
#endif

#ifdef VECT_128
#include "src/amplitudes_calculator_m128.h"
template <typename T>
using AmplitudesComputerType = AmplitudesCalculatorM128<T>;
#endif

#ifdef VECT_256
#include "src/amplitudes_calculator_m256.h"
template <typename T>
using AmplitudesComputerType = AmplitudesCalculatorM256<T>;
#endif

#ifdef VECT_512
#include "src/amplitudes_calculator_m512.h"
template <typename T>
using AmplitudesComputerType = AmplitudesCalculatorM512<T>;
#endif

#include <omp.h>
#include <iostream>
#include <cstdint>
#include <cstddef>
#include <cstdlib>

int main(int argc, char const *argv[]) {
	ptrdiff_t n_rec = atol(argv[1]);
	ptrdiff_t n_sources = atol(argv[2]);

	float *sources = new float[n_sources*3];
	float *rec = new float[n_rec*3];
	float *tensor_matrix = new float[6];
 	float *amplitudes = new float[n_sources*n_rec*3];

	double t1 = omp_get_wtime();
	AmplitudesComputerType<float> computer(sources, rec, tensor_matrix, n_sources, n_rec, amplitudes);
	computer.calculate();
	double t2 = omp_get_wtime();

	fprintf(stderr, "Anti opt: %d\n", amplitudes[0]);
	printf("Recs: %d, Srcs: %d, Time: %.2f\n", n_rec, n_sources, t2-t1);

	delete [] sources;
	delete [] rec;
	delete [] tensor_matrix;
	delete [] amplitudes;
	return 0;
}