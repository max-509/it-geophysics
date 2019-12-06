#ifdef WITH_BLOCKS
#include "with_blocks/src/coherent_summation.h"
#endif

#ifdef WITHOUT_BLOCKS
#include "without_blocks/src/coherent_summation.h"
#endif

#include <omp.h>
#include <iostream>
#include <cstdint>
#include <cstddef>
#include <cstdlib>

using namespace Eigen;

template <typename T>
using Array1D = Matrix<T, 6, 1>;
template <typename T>
using MapArray1D = Map<const Array1D<T>, Aligned>;

template <typename T>
using Array2D = Matrix<T, Dynamic, Dynamic, RowMajor>;
template <typename T>
using MapArray2D = Map<const Array2D<T>, Aligned>;

int main(int argc, char const *argv[]) {
	ptrdiff_t n_rec = atol(argv[1]);
	ptrdiff_t sources_count = atol(argv[2]);
	ptrdiff_t n_samples = atol(argv[3]);
	
	double *sources_coords_data = new double[sources_count*3];
	double *rec_coords_data = new double[n_rec*3];
	double *tensor_matrix_data = new double[6];
	double *rec_samples_data = new double[n_rec*n_samples];
	int64_t *sources_times_data = new int64_t[sources_count*n_rec];
	double *data = new double[sources_count*n_samples];

	MapArray2D<double> rec_samples_data2D{rec_samples_data, n_rec, n_samples};
	MapArray2D<double> rec_coords_data2D{rec_coords_data, n_rec, 3};
	MapArray2D<double> sources_coords_data2D{sources_coords_data, sources_count, 3};
	MapArray2D<int64_t> sources_times_data2D{sources_times_data, sources_count, n_rec};
	MapArray1D<double> tensor_matrix_data1D{tensor_matrix_data};	
	Map<Matrix<double, Dynamic, Dynamic, RowMajor>, Aligned> data2D(data, sources_count, n_samples);

	double t1 = omp_get_wtime();
	compute(rec_samples_data2D, rec_coords_data2D, sources_coords_data2D, sources_times_data2D, tensor_matrix_data1D, data2D);
	double t2 = omp_get_wtime();

	fprintf(stderr, "anti_opt: %f\n", data[0]);
	printf("Recs: %d, Srcs: %d, Smpls: %d, Time: %.2f\n", n_rec, sources_count, n_samples, t2-t1);

	delete [] sources_coords_data;
	delete [] rec_coords_data;
	delete [] tensor_matrix_data;
	delete [] rec_samples_data;
	delete [] sources_times_data;
	delete [] data;
	
	return 0;
}