#include <fstream>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <omp.h>

typedef float v4sf __attribute__ ((vector_size(16)));

union f4 {
	v4sf v;
	float f[4];
};

void v4sf_addmov(float* v1, const float* v2) {

	f4 temp = {v1[0], v1[1], v1[2], v1[3]};
	temp.v += *((v4sf*)v2);
	__builtin_ia32_movntps(v1, temp.v);
}

float calc_radius(float dx, float dy, float dz) {
	return sqrt(dx*dx+dy*dy+dz*dz);
}

int main(int argc, char const *argv[]) {
	omp_set_num_threads(1);
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
	size_t nz = 100;

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

	size_t count_blocks_by_rec = 1;
	if (2 == argc) {
		count_blocks_by_rec = atoi(argv[1]);
	}
	size_t block_size_by_rec;
	size_t rest_block_by_rec = rec_count%count_blocks_by_rec;
	size_t sum_block_size_by_rec = 0;

	t1 = omp_get_wtime();
	//algorithm
	//******************************************************//
	#pragma omp parallel
	{
		float r, t, res;
		size_t ind;
		#pragma omp for schedule(guided)
		for (size_t i = 0; i < nz; ++i) {
			for (size_t c = 0; c < count_blocks_by_rec; ++c) {
				if (rest_block_by_rec > 0) {
					block_size_by_rec = rec_count/count_blocks_by_rec + 1;
					--rest_block_by_rec;
				} else {
					block_size_by_rec = rec_count/count_blocks_by_rec;
				}	
				for (size_t j = 0; j < nx; ++j) {
					for (size_t k = 0; k < ny; ++k) {
						for (size_t m = sum_block_size_by_rec; m < sum_block_size_by_rec+block_size_by_rec; ++m) {
							r = calc_radius((x0+j*dx)-rec_coords[m*3],
										    (y0+k*dy)-rec_coords[m*3+1],
										    (z0+i*dz)-rec_coords[m*3+2]);
							t = r/vv;
							ind = (size_t)(t/dt);
							size_t l = 0;
							for (l = 0; l < times/4; ++l) {
								if ((l+1)*4+ind < times) {
									v4sf_addmov(area_discr.get()+i*nx*ny*times+j*ny*times+k*times+l*4, rec_times.get()+m*times+ind+l*4);
								} else break;
							}
								std::cerr << times/4 << std::endl;
							for (l = l*4; l < times; ++l) {
								if (l+ind < times) {
									area_discr[i*nx*ny*times+j*ny*times+k*times+l] += rec_times[m*times+ind+l];
								} else break;
							}
						}
					}
				}
			}
			sum_block_size_by_rec += block_size_by_rec;
		}
	}
	//******************************************************//
	t2 = omp_get_wtime();

	std::ofstream time_file;
	time_file.open("./time_file", std::ios::out | std::ios::app);
	if (!time_file.is_open()) {
		std::cout << "Count blocks: " << count_blocks_by_rec << ", Time: " << t2-t1 << std::endl;
	} else {
		time_file << "Receiver block" << ": Count blocks: " << count_blocks_by_rec << ", Time: " << t2-t1 << std::endl;
	}

	return 0;
}