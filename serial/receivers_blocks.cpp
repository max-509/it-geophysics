#include <cstddef>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <omp.h>

//Test rec_blocks, where numbers - rec_block_size
//r17-26 : 2000, 500, 100, 50, 40, 20, 10, 5, 2, 1

float calc_radius(float dx, float dy, float dz) {
    return sqrt(dx*dx+dy*dy+dz*dz);
}

int main(int argc, char const *argv[]) {
	omp_set_num_threads(4);
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

	float dt = 2e-3;
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

	size_t rec_block_size = rec_count;	
	if (argc >= 2) {
        rec_block_size = atoi(argv[1]);
    }

	t1 = omp_get_wtime();
	//algorithm
	//******************************************************//
	std::unique_ptr<size_t[]> ind_arr{new size_t[nx*ny*nz*rec_count]()};
	std::unique_ptr<size_t[]> min_ind_arr{new size_t[nx*ny*nz]()};
	#pragma omp parallel 
	{
		#pragma omp for schedule(dynamic)
	    for (size_t i = 0; i < nz; ++i) {
	        for (size_t c_r = 0; c_r < rec_count; c_r += rec_block_size) {
		        for (size_t j = 0; j < nx; ++j) {
		            for (size_t k = 0; k < ny; ++k) {
		            	if (min_ind_arr[i*nx*ny+j*ny+k] == 0) {
		            		for (size_t m = 0; m < rec_count; ++m) {
		            			ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count+m] = round(calc_radius((x0+j*dx)-rec_coords[m*3],
		            																				  		(y0+k*dy)-rec_coords[m*3+1],
		            																				  		(z0+i*dz)-rec_coords[m*3+2])
		            																						/(vv*dt)) + 1;
		            			if (0 != m) {
		            				min_ind_arr[i*nx*ny+j*ny+k] = std::min(min_ind_arr[i*nx*ny+j*nx+k], 
		            											  		   ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count+m]); 
		            			} else {
		            				min_ind_arr[i*nx*ny+j*ny+k] = ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count+m];
		            			}
		            		}
		            	}
		                for (size_t m = c_r; m < std::min(c_r+rec_block_size, rec_count); ++m) {
		                    for (size_t l = 0; l < times-ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count+m]+min_ind_arr[i*nx*ny+j*ny+k]; ++l) {
		                    	// for (size_t m = 0; m < rec_count; ++m) {
		                    	// 	std::cout << rec_times[m*times+ind_arr[m]+l-min_ind] << std::endl;
		                    	// }
		                    	// return 0;
		                        area_discr[i*nx*ny*times+j*ny*times+k*times+l] += rec_times[m*times+ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count+m]
		                        															+l-min_ind_arr[i*nx*ny+j*ny+k]];
		                    }
		                }
		            }
		        }
		    }
    	}	
	}
    //******************************************************//
    t2 = omp_get_wtime();

 //    std::ifstream results_file;
	// results_file.open("../Summation_Results2.bin", std::ios::binary);
	// if (!results_file.is_open()) {
	// 	std::cerr << "Can't open Summation_Results.bin" << std::endl;
	// 	return 1;
	// }

	// std::ofstream file_test;
	// file_test.open("./Results.bin", std::ios::binary | std::ios::out);

	// for (size_t i = 0; i < nz; ++i) {
	// 	for (size_t j = 0; j < nx; ++j) {
	// 		for (size_t k = 0; k < ny; ++k) {
	// 			for (size_t l = 0; l < times; ++l) {
	// 				file_test.write(reinterpret_cast<char*>(area_discr.get()+i*nx*ny*times+j*ny*times+k*times+l), sizeof(float));
	// 				// temp1 += (real_results[i*nx*ny*times+j*ny*times+k*times+l]-area_discr[i*nx*ny*times+j*ny*times+k*times+l])*
	// 				// 		 (real_results[i*nx*ny*times+j*ny*times+k*times+l]-area_discr[i*nx*ny*times+j*ny*times+k*times+l]);
	// 				// // std::cout << real_results[i*nx*ny*times+j*ny*times+k*times+l] << " " << area_discr[i*nx*ny*times+j*ny*times+k*times+l] << std::endl;
	// 				// temp2 += real_results[i*nx*ny*times+j*ny*times+k*times+l]*real_results[i*nx*ny*times+j*ny*times+k*times+l];
	// 			}
	// 			// return 0;
	// 		}
	// 	}
	// }

	// std::unique_ptr<float[]> real_results{new float[nx*ny*nz*times]};
	// results_file.read(reinterpret_cast<char*>(real_results.get()), nx*ny*nz*times*sizeof(float));

	// // std::cout << area_discr[0] << " " << area_discr[1] << " " << area_discr[10*10*10*10000-1] << std::endl;

	// float result = 0;
	// float temp1 = 0, temp2 = 0;
	// for (size_t i = 0; i < nz; ++i) {
	// 	for (size_t j = 0; j < nx; ++j) {
	// 		for (size_t k = 0; k < ny; ++k) {
	// 			for (size_t l = 0; l < times; ++l) {
	// 				temp1 += (real_results[i*nx*ny*times+j*ny*times+k*times+l]-area_discr[i*nx*ny*times+j*ny*times+k*times+l])*
	// 						 (real_results[i*nx*ny*times+j*ny*times+k*times+l]-area_discr[i*nx*ny*times+j*ny*times+k*times+l]);
	// 				// std::cout << real_results[i*nx*ny*times+j*ny*times+k*times+l] << " " << area_discr[i*nx*ny*times+j*ny*times+k*times+l] << std::endl;
	// 				temp2 += real_results[i*nx*ny*times+j*ny*times+k*times+l]*real_results[i*nx*ny*times+j*ny*times+k*times+l];
	// 			}
	// 			// return 0;
	// 		}
	// 	}
	// }
	// result = sqrt(temp1)/sqrt(temp2);

	// std::cout << "Result == " << result << std::endl;


    // std::ofstream time_file;
    // time_file.open("./time_file", std::ios::out | std::ios::app);
    // if (!time_file.is_open()) {
    //     std::cout << "Rec block size: " << rec_block_size << ", Time: " << t2-t1 << std::endl;
    // } else {
    //     time_file << "Receiver block" << ": Rec block size: " << rec_block_size << ", Time: " << t2-t1 << std::endl;
    // }

    return 0;
}