#include "src/coherent_summation.h"
#include <boost/multi_array.hpp>

#include <omp.h>
#include <iostream>
#include <cstdint>
#include <cstddef>
#include <cstdlib>

template <typename T>
using Array1D = boost::multi_array_ref<T, 1>;
template <typename T>
using Array2D = boost::multi_array_ref<T, 2>;
template <typename T>
using Array3D = boost::multi_array_ref<T, 3>;

int main(int argc, char const *argv[]) {
	size_t n_rec = atol(argv[1]);
	size_t sources_count = atol(argv[2]);
	size_t n_samples = atol(argv[3]);

	double *sources_coords_data = new double[sources_count*3];
	double *rec_coords_data = new double[n_rec*3];
	double *tensor_matrix_data = new double[6];
	double *rec_samples_data = new double[n_rec*n_samples];
	int64_t *sources_times_data = new int64_t[sources_count*n_rec];
	double *data = new double[sources_count*n_samples];

	Array2D<double> rec_samples_data2D{rec_samples_data, boost::extents[n_rec][n_samples], boost::c_storage_order()};
	Array2D<double> rec_coords_data2D{rec_coords_data, boost::extents[n_rec][3], boost::c_storage_order()};
	Array2D<double> sources_coords_data2D{sources_coords_data, boost::extents[sources_count][3], boost::c_storage_order()};
	Array2D<int64_t> sources_times_data2D{sources_times_data, boost::extents[sources_count][n_rec], boost::c_storage_order()};
	Array1D<double> tensor_matrix_data1D{tensor_matrix_data, boost::extents[6], boost::c_storage_order()};

	double t1 = omp_get_wtime();
	compute(rec_samples_data2D, rec_coords_data2D, sources_coords_data2D, sources_times_data2D, tensor_matrix_data1D, data);
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