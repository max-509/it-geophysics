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
		perror("Data_noise_free.bin");
		exit(0);
	}
	FILE* receivers_file = fopen("./Receivers_Array.bin", "r");
	if (NULL == receivers_file) {
		perror("Receivers_Array.bin");
		exit(0);
	}

	size_t rec_count = 2000;
	size_t times = 10000;

	float* rec_coords = new float[rec_count*3];
	float* rec_times = new float[rec_count*times];

	fread(rec_coords, sizeof(float), rec_count*3, receivers_file);
	fread(rec_times, sizeof(float), rec_count*times, data_file);

	fclose(receivers_file);
	fclose(data_file);

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
	if (1 < nx) dx = ((float)(x1-x0))/(nx-1);
	if (1 < ny) dy = ((float)(y1-y0))/(ny-1);
	if (1 < nz) dz = ((float)(z1-z0))/(nz-1);

	double t1, t2;

	t1 = omp_get_wtime();
	//algorithm
	//******************************************************//
	float r, t, res;
	size_t ind;
	for (size_t l = 0; l < times; ++l) {
		for (size_t i = 0; i < nz; ++i) {
			for (size_t j = 0; j < nx; ++j) {
				for (size_t k = 0; k < ny; ++k) {
					res = 0;
					for (size_t m = 0; m < rec_count; ++m) {
						r = calc_radius((x0+j*dx)-rec_coords[m*3],
									    (y0+k*dy)-rec_coords[m*3+1],
									    (z0+i*dz)-rec_coords[m*3+2]);
						t = r/vv;
						ind = (size_t)(t/dt);
						if (l+ind < times) {
							res += rec_times[m*times+ind+l];
						}
					}
					area_discr[l*nx*ny*nz+i*nz*nx+j*ny+k] = res;
				}
			}
		}
	}
	//******************************************************//
	t2 = omp_get_wtime();

	std::cout << "Time: " << t2-t1 << std::endl;

	delete [] area_discr;
	delete [] rec_coords;
	delete [] rec_times;

	return 0;
}