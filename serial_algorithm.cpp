#include <iostream>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <omp.h>

float calc_radius(float dx, float dy, float dz) {
	return sqrt(dx*dx+dy*dy+dz*dz);
}

int main(int argc, char const *argv[]) {
	FILE* data_file = fopen("./Data_noise_free.bin", "r");
	if (NULL == data_file) {
		perror("data_file");
		exit(0);
	}
	FILE* receivers_file = fopen("./Receivers_Array.bin", "r");
	if (NULL == receivers_file) {
		perror("receivers_file");
		exit(0);
	}

	size_t rec_count = 8000;
	size_t times = 10000;

	float* rec_coords = new float[rec_count*3];
	float* rec_times = new float[rec_count*times];

	fread(rec_coords, sizeof(float), rec_count*3, receivers_file);
	fread(rec_times, sizeof(float), rec_count*times, data_file);

	// for (size_t i = 0; i < rec_count; ++i) {
	// 	// for (size_t j = 0; j < times; ++j) {
	// 		// std::cerr << rec_times[i][j] << " ";
	// 	// }
	// 	// std::cerr << std::endl;
	// 	std::cerr << rec_coords[i][0] << " " << rec_coords[i][1] << " "
	// 	<< rec_coords[i][2] << std::endl;
	// }


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

	float* area_discr = new float[times*nx*ny*nz];

	float dx, dy, dz;
	if (1 < nx) dx = (float)(x1-x0)/nx-1;
	if (1 < ny) dy = (float)(y1-y0)/ny-1;
	if (1 < nz) dz = (float)(z1-z0)/nz-1;

	double t1, t2;

	//algorithm
	//******************************************************//
	t1 = omp_get_wtime();
	float r, t, res;
	size_t ind;
	for (size_t i = 0; i < times; ++i) {
		for (size_t j = 0; j < nx; ++j) {
			for (size_t k = 0; k < ny; ++k) {
				for (size_t l = 0; l < nz; ++l) {
					res = 0;
					for (size_t m = 0; m < rec_count; ++m) {
						r = calc_radius((x0+j*dx)-rec_coords[m*3],
									    (y0+k*dy)-rec_coords[m*3+1],
									    (z0+l*dz)-rec_coords[m*3+2]);
						t = r/vv;
						ind = (size_t)(t/dt);
						if (ind < times) {
							res += rec_times[i+m*times+ind];
						}
					}
					area_discr[i*nx*ny*nz+j*ny*nz+k*nz+l] = res;
				}
			}
		}
	}
	t2 = omp_get_wtime();
	//******************************************************//

	std::cout << "Time: " << t2-t1 << std::endl;

	delete [] area_discr;
	delete [] rec_coords;
	delete [] rec_times;
	return 0;
}