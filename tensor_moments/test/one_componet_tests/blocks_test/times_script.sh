#!/bin/sh
#first arg - n_rec, second arg - n_srcs, third arg - n_samples
mkdir results_times
export OMP_NUM_THREADS=4

#Eigen

g++ -O3 -pthread -march=native -mavx2 -std=c++11 -fopenmp Eigen/times_tests.cpp -DWITH_BLOCKS
./a.out 2000 4000 10000 >> results_times/eigen_with_blocks.txt
./a.out 2000 8000 10000 >> results_times/eigen_with_blocks.txt
./a.out 2000 12000 10000 >> results_times/eigen_with_blocks.txt
./a.out 2000 16000 10000 >> results_times/eigen_with_blocks.txt
./a.out 2000 19000 10000 >> results_times/eigen_with_blocks.txt

./a.out 4000 4000 10000 >> results_times/eigen_with_blocks.txt
./a.out 4000 8000 10000 >> results_times/eigen_with_blocks.txt
./a.out 4000 12000 10000 >> results_times/eigen_with_blocks.txt
./a.out 4000 16000 10000 >> results_times/eigen_with_blocks.txt
./a.out 4000 19000 10000 >> results_times/eigen_with_blocks.txt

./a.out 2000 4000 10000 >> results_times/eigen_with_blocks.txt
./a.out 2000 4000 20000 >> results_times/eigen_with_blocks.txt
./a.out 2000 4000 30000 >> results_times/eigen_with_blocks.txt
./a.out 2000 4000 40000 >> results_times/eigen_with_blocks.txt
./a.out 2000 4000 50000 >> results_times/eigen_with_blocks.txt
./a.out 2000 4000 60000 >> results_times/eigen_with_blocks.txt
./a.out 2000 4000 70000 >> results_times/eigen_with_blocks.txt
./a.out 2000 4000 80000 >> results_times/eigen_with_blocks.txt
./a.out 2000 4000 90000 >> results_times/eigen_with_blocks.txt
./a.out 2000 4000 100000 >> results_times/eigen_with_blocks.txt

g++ -O3 -pthread -march=native -mavx2 -std=c++11 -fopenmp Eigen/times_tests.cpp -DWITHOUT_BLOCKS
./a.out 2000 4000 10000 >> results_times/eigen_without_blocks.txt
./a.out 2000 8000 10000 >> results_times/eigen_without_blocks.txt
./a.out 2000 12000 10000 >> results_times/eigen_without_blocks.txt
./a.out 2000 16000 10000 >> results_times/eigen_without_blocks.txt
./a.out 2000 19000 10000 >> results_times/eigen_without_blocks.txt

./a.out 4000 4000 10000 >> results_times/eigen_without_blocks.txt
./a.out 4000 8000 10000 >> results_times/eigen_without_blocks.txt
./a.out 4000 12000 10000 >> results_times/eigen_without_blocks.txt
./a.out 4000 16000 10000 >> results_times/eigen_without_blocks.txt
./a.out 4000 19000 10000 >> results_times/eigen_without_blocks.txt

./a.out 2000 4000 10000 >> results_times/eigen_without_blocks.txt
./a.out 2000 4000 20000 >> results_times/eigen_without_blocks.txt
./a.out 2000 4000 30000 >> results_times/eigen_without_blocks.txt
./a.out 2000 4000 40000 >> results_times/eigen_without_blocks.txt
./a.out 2000 4000 50000 >> results_times/eigen_without_blocks.txt
./a.out 2000 4000 60000 >> results_times/eigen_without_blocks.txt
./a.out 2000 4000 70000 >> results_times/eigen_without_blocks.txt
./a.out 2000 4000 80000 >> results_times/eigen_without_blocks.txt
./a.out 2000 4000 90000 >> results_times/eigen_without_blocks.txt
./a.out 2000 4000 100000 >> results_times/eigen_without_blocks.txt

#########################

#Arrays

g++ -O3 -pthread -march=native -mavx2 -std=c++11 -fopenmp arrays/times_tests.cpp -DWITH_BLOCKS
./a.out 2000 4000 10000 >> results_times/arrays_with_blocks.txt
./a.out 2000 8000 10000 >> results_times/arrays_with_blocks.txt
./a.out 2000 12000 10000 >> results_times/arrays_with_blocks.txt
./a.out 2000 16000 10000 >> results_times/arrays_with_blocks.txt
./a.out 2000 19000 10000 >> results_times/arrays_with_blocks.txt

./a.out 4000 4000 10000 >> results_times/arrays_with_blocks.txt
./a.out 4000 8000 10000 >> results_times/arrays_with_blocks.txt
./a.out 4000 12000 10000 >> results_times/arrays_with_blocks.txt
./a.out 4000 16000 10000 >> results_times/arrays_with_blocks.txt
./a.out 4000 19000 10000 >> results_times/arrays_with_blocks.txt

./a.out 2000 4000 10000 >> results_times/arrays_with_blocks.txt
./a.out 2000 4000 20000 >> results_times/arrays_with_blocks.txt
./a.out 2000 4000 30000 >> results_times/arrays_with_blocks.txt
./a.out 2000 4000 40000 >> results_times/arrays_with_blocks.txt
./a.out 2000 4000 50000 >> results_times/arrays_with_blocks.txt
./a.out 2000 4000 60000 >> results_times/arrays_with_blocks.txt
./a.out 2000 4000 70000 >> results_times/arrays_with_blocks.txt
./a.out 2000 4000 80000 >> results_times/arrays_with_blocks.txt
./a.out 2000 4000 90000 >> results_times/arrays_with_blocks.txt
./a.out 2000 4000 100000 >> results_times/arrays_with_blocks.txt

g++ -O3 -pthread -march=native -mavx2 -std=c++11 -fopenmp arrays/times_tests.cpp -DWITHOUT_BLOCKS
./a.out 2000 4000 10000 >> results_times/arrays_without_blocks.txt
./a.out 2000 8000 10000 >> results_times/arrays_without_blocks.txt
./a.out 2000 12000 10000 >> results_times/arrays_without_blocks.txt
./a.out 2000 16000 10000 >> results_times/arrays_without_blocks.txt
./a.out 2000 19000 10000 >> results_times/arrays_without_blocks.txt

./a.out 4000 4000 10000 >> results_times/arrays_without_blocks.txt
./a.out 4000 8000 10000 >> results_times/arrays_without_blocks.txt
./a.out 4000 12000 10000 >> results_times/arrays_without_blocks.txt
./a.out 4000 16000 10000 >> results_times/arrays_without_blocks.txt
./a.out 4000 19000 10000 >> results_times/arrays_without_blocks.txt

./a.out 2000 4000 10000 >> results_times/arrays_without_blocks.txt
./a.out 2000 4000 20000 >> results_times/arrays_without_blocks.txt
./a.out 2000 4000 30000 >> results_times/arrays_without_blocks.txt
./a.out 2000 4000 40000 >> results_times/arrays_without_blocks.txt
./a.out 2000 4000 50000 >> results_times/arrays_without_blocks.txt
./a.out 2000 4000 60000 >> results_times/arrays_without_blocks.txt
./a.out 2000 4000 70000 >> results_times/arrays_without_blocks.txt
./a.out 2000 4000 80000 >> results_times/arrays_without_blocks.txt
./a.out 2000 4000 90000 >> results_times/arrays_without_blocks.txt
./a.out 2000 4000 100000 >> results_times/arrays_without_blocks.txt