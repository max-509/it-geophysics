#include "src/array2D.h"
#include "src/coherent_summation.h"
#include <boost/multi_array.hpp>

#include <omp.h>
#include <iostream>
#include <cstdint>
#include <cstddef>
#include <cstdlib>

int main(int argc, char const *argv[]) {
	ptrdiff_t n_rec = 2000;
	ptrdiff_t sources_count = 3000;
	ptrdiff_t n_samples = 10000;

	double *sources_coords_data = new double[sources_count*3];
	double *rec_coords_data = new double[n_rec*3];
	double *tensor_matrix_data = new double[6];
	double *rec_samples_data = new double[n_rec*n_samples];
	int64_t *sources_times_data = new int64_t[sources_count*n_rec];
	double *data = new double[sources_count*n_samples];

	Array2D<double> rec_samples_data3D{rec_samples_data, n_rec, n_samples};
	Array2D<double> rec_coords_data2D{rec_coords_data, n_rec, 3};
	Array2D<double> sources_coords_data2D{sources_coords_data, sources_count, 3};
	Array2D<int64_t> sources_times_data2D{sources_times_data, sources_count, n_rec};

	double t1 = omp_get_wtime();
	compute(rec_samples_data3D, rec_coords_data2D, sources_coords_data2D, sources_times_data2D, tensor_matrix_data, data);
	double t2 = omp_get_wtime();

	printf("anti_opt: %f\n", data[0]);
	printf("anti_opt: %f\n", data[10]);
	printf("anti_opt: %f\n", data[20]);
	printf("N_smps: %d, Srcs: %d, Recs: %d, Time: %f\n", n_samples, sources_count, n_rec, t2-t1);

	delete [] sources_coords_data;
	delete [] rec_coords_data;
	delete [] tensor_matrix_data;
	delete [] rec_samples_data;
	delete [] sources_times_data;
	delete [] data;

	return 0;
}