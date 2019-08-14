#include <fstream>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <omp.h>

float calc_radius(float dx, float dy, float dz) {
	return sqrt(dx*dx+dy*dy+dz*dz);
}

int main(int argc, char const *argv[]) {

	std::ifstream data_file, receivers_file;
	data_file.open("../Data_noise_free.bin", std::ios::binary);
	if (!data_file.is_open()) {
		std::cerr << "Can't open Data_noise_free.bin" << std::endl;
		return 1;
	}
	receivers_file.open("../Receivers_Array.bin", std::ios::binary);
	if (!receivers_file.is_open()) {
		std::cerr << "Can't open Receivers_Array.bin" << std::endl;
		return 1;
	}

	size_t rec_count = 2000;
	size_t times = 10000;

	std::unique_ptr<float[]> rec_times{new float[rec_count*times]()};
	data_file.read(reinterpret_cast<char*>(rec_times.get()), rec_count*times*sizeof(float));

	std::unique_ptr<float[]> rec_coords{new float[rec_count*3]()};
	receivers_file.read(reinterpret_cast<char*>(rec_coords.get()), rec_count*3*sizeof(float));

	float dt = 2e-2;
	size_t nx = 10;
	size_t ny = 10;
	size_t nz = 10;

	float vv = 3000;

	long long x0 = -1000;
	long long x1 = 1000;
	long long y0 = -1000;
	long long y1 = 1000;
	long long z0 = 500;
	long long z1 = 2500;

	std::unique_ptr<float[]> area_discr{new float[times*nx*ny*nz]()};

	float dx, dy, dz;
	if (1 < nx) dx = ((float)(x1-x0))/(nx-1);
	if (1 < ny) dy = ((float)(y1-y0))/(ny-1);
	if (1 < nz) dz = ((float)(z1-z0))/(nz-1);

	double t1, t2;

	size_t count_blocks_by_times = 1;
	size_t count_blocks_by_rec = 1;
	if (3 == argc) {
		count_blocks_by_times = atoi(argv[1]);
		count_blocks_by_rec = atoi(argv[2]);
	}
	
	size_t block_size_by_times;
	size_t rest_block_by_times = times%count_blocks_by_times;
	size_t sum_block_size_by_times = 0;

	size_t block_size_by_rec;
	size_t rest_block_by_rec = rec_count%count_blocks_by_rec;
	size_t sum_block_size_by_rec = 0;


	t1 = omp_get_wtime();
	//algorithm
	//******************************************************//
	float r, t, res;
	size_t ind;
	for (size_t c = 0; c < count_blocks_by_times; ++c) {
		if (rest_block_by_times > 0) {
			block_size_by_times = times/count_blocks_by_times + 1;
			--rest_block_by_times;
		} else {
			block_size_by_times = times/count_blocks_by_times;
		}
		for (size_t i = 0; i < nz; ++i) {
			for (size_t j = 0; j < nx; ++j) {
				for (size_t k = 0; k < ny; ++k) {
					for (size_t m = 0; m < rec_count; ++m) {
						r = calc_radius((x0+j*dx)-rec_coords[m*3],
									    (y0+k*dy)-rec_coords[m*3+1],
									    (z0+i*dz)-rec_coords[m*3+2]);
						t = r/vv;
						ind = (size_t)(t/dt);
						for (size_t l = sum_block_size_by_times; l < sum_block_size_by_times+=block_size_by_times; ++l) {
							if (l+ind < times) {
								area_discr[i*nx*ny*times+j*ny*times+k*times+l] += rec_times[m*times+ind+l];
							} else break;
						}
					}
				}
			}
		}
	}
	//******************************************************//
	t2 = omp_get_wtime();

	return 0;
}