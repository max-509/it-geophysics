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

	// for (size_t i = 0; i < rec_count; ++i) {
	// 	// for (size_t j = 0; j < times; ++j) {
	// 		// std::cerr << rec_times[i][j] << " ";
	// 	// }
	// 	// std::cerr << std::endl;
	// 	std::cerr << rec_coords[i*3] << " " << rec_coords[i*3+1] << " "
	// 	<< rec_coords[i*3+2] << std::endl;
	// }


	// std::cerr << rec_times[0] << " " << rec_times[9999];
	// std::cerr << std::endl;

	// std::cerr << rec_times[1999*10000] << " " << rec_times[1999*10000+9999];
	// std::cerr << std::endl;

	// std::cerr << rec_coords[0] << " " << rec_coords[1] << " "
	// << rec_coords[2] << std::endl;

	// std::cerr << rec_coords[1999*3] << " " << rec_coords[1999*3+1] << " "
	// << rec_coords[1999*3+2] << std::endl;

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

	t1 = omp_get_wtime();
	//algorithm
	//******************************************************//
	float r, t, res;
	size_t ind;
	for (size_t i = 0; i < nz; ++i) {
		for (size_t j = 0; j < nx; ++j) {
			for (size_t k = 0; k < ny; ++k) {
				for (size_t l = 0; l < times; ++l) {
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
					area_discr[i*nx*ny*times+j*ny*times+k*times+l] = res;
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

	// std::ifstream results_file;
	// results_file.open("../Summation_Results.bin", std::ios::binary);
	// if (!results_file.is_open()) {
	// 	std::cerr << "Can't open Summation_Results.bin" << std::endl;
	// 	return 1;
	// }

	// std::unique_ptr<float[]> real_results{new float[nx*ny*nz*times]};
	// results_file.read(reinterpret_cast<char*>(real_results.get()), nx*ny*nz*times*sizeof(float));

	// float result = 0;
	// float temp1 = 0, temp2 = 0;
	// for (size_t i = 0; i < nz; ++i) {
	// 	for (size_t j = 0; j < nx; ++j) {
	// 		for (size_t k = 0; k < ny; ++k) {
	// 			for (size_t l = 0; l < times; ++l) {
	// 				temp1 += (real_results[i*nx*ny*times+j*ny*times+k*times+l]-area_discr[i*nx*ny*times+j*ny*times+k*times+l])*
	// 						 (real_results[i*nx*ny*times+j*ny*times+k*times+l]-area_discr[i*nx*ny*times+j*ny*times+k*times+l]);
	// 				std::cout << real_results[i*nx*ny*times+j*ny*times+k*times+l] << " " << area_discr[i*nx*ny*times+j*ny*times+k*times+l] << std::endl;
	// 				temp2 += real_results[i*nx*ny*times+j*ny*times+k*times+l]*real_results[i*nx*ny*times+j*ny*times+k*times+l];
	// 			}
	// 		}
	// 	}
	// }
	// result = sqrt(temp1)/sqrt(temp2);

	std::ofstream time_file;
	time_file.open("./time_file", std::ios::out | std::ios::app);
	if (!time_file.is_open()) {
		std::cout << "Time: " << t2-t1 << std::endl;
	} else {
		time_file << "Source" << ": Time: " << t2-t1 << std::endl;
	}

	// std::cout << "Result == " << result << std::endl;

	return 0;
}