#include <fstream>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstddef>
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

	ptrdiff_t rec_count = 2000;
	ptrdiff_t times = 20000;

	std::unique_ptr<float[]> rec_times{new float[rec_count*times]()};
	data_file.read(reinterpret_cast<char*>(rec_times.get()), rec_count*times*sizeof(float));

	std::unique_ptr<float[]> rec_coords{new float[rec_count*3]()};
	receivers_file.read(reinterpret_cast<char*>(rec_coords.get()), rec_count*3*sizeof(float));

	float dt = 2e-3;
	ptrdiff_t nx = 10;
	ptrdiff_t ny = 10;
	ptrdiff_t nz = 10;

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

	std::unique_ptr<size_t[]> ind_arr{new size_t[nx*ny*nz*rec_count]};
	std::unique_ptr<size_t[]> min_ind_arr{new size_t[nx*ny*nz]};
	float r, t, res;
	t1 = omp_get_wtime();
	//algorithm
	//******************************************************//
	for (ptrdiff_t i = 0; i < nz; ++i) {
		for (ptrdiff_t j = 0; j < nx; ++j) {
			for (ptrdiff_t k = 0; k < ny; ++k) {
				r = calc_radius((x0+j*dx)-rec_coords[0],
							    (y0+k*dy)-rec_coords[1],
							    (z0+i*dz)-rec_coords[2]);
				t = r/vv;
				ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count] = round(t/dt) + 1;
				min_ind_arr[i*nx*ny+j*ny+k] = ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count];
				for (ptrdiff_t m = 1; m < rec_count; ++m) {
					r = calc_radius((x0+j*dx)-rec_coords[m*3],
								    (y0+k*dy)-rec_coords[m*3+1],
								    (z0+i*dz)-rec_coords[m*3+2]);
					t = r/vv;
					ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count+m] = round(t/dt) + 1;
					min_ind_arr[i*nx*ny+j*ny+k] = std::min(min_ind_arr[i*nx*ny+j*ny+k], 
														   ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count+m]);
				}
			}
		}
	}
	for (ptrdiff_t i = 0; i < nz; ++i) {
		for (ptrdiff_t j = 0; j < nx; ++j) {
			for (ptrdiff_t k = 0; k < ny; ++k) {
				ptrdiff_t min_ind = min_ind_arr[i*nx*ny+j*ny+k];
				for (ptrdiff_t l = 0; l < times; ++l) {
					for (ptrdiff_t m = 0; m < rec_count; ++m) {
						ptrdiff_t ind = ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count+m] - min_ind;
						if (l+ind < times) {
							area_discr[i*nx*ny*times+j*ny*times+k*times+l] += rec_times[m*times+ind+l];
						}					
					}
				}
			}
		}
	}
	//0) спросить про репозиторий 
	//1) текущая версия с внешним циклом по times (сравнить)
	//2) разбить на блоки по times
	//3) замеры времени от размера блока times
	//4) разбить на блоки по rec_count
	//5) циклы по блокам times и rec_count вынести наверх
	//Ещё сделать смарт-птр'ы и не сишные файлы, проверить mmap для файлов
	//Идея: сделать цикл по times самым глубоким, чтобы прерывать, если ind+t>=times
	//******************************************************//
	t2 = omp_get_wtime();

	std::cout << "Source time: " << t2-t1 << std::endl;

	return 0;
}