#!/bin/sh
#first arg - n_rec, second arg - n_srcs, third arg - n_samples
mkdir results_times
export OMP_NUM_THREADS=4

#My classes

g++ -O3 -pthread -march=native -mavx2 -std=c++11 -fopenmp my_classes/times_tests.cpp
# ./a.out 2000 4000 10000 >> results_times/my_classes_times.txt
# ./a.out 2000 8000 10000 >> results_times/my_classes_times.txt
# ./a.out 2000 12000 10000 >> results_times/my_classes_times.txt
# ./a.out 2000 16000 10000 >> results_times/my_classes_times.txt
# ./a.out 2000 19000 10000 >> results_times/my_classes_times.txt

# ./a.out 4000 4000 10000 >> results_times/my_classes_times.txt
# ./a.out 4000 8000 10000 >> results_times/my_classes_times.txt
# ./a.out 4000 12000 10000 >> results_times/my_classes_times.txt
# ./a.out 4000 16000 10000 >> results_times/my_classes_times.txt
# ./a.out 4000 19000 10000 >> results_times/my_classes_times.txt

# ./a.out 2000 4000 10000 >> results_times/my_classes_times.txt
# ./a.out 2000 4000 20000 >> results_times/my_classes_times.txt
# ./a.out 2000 4000 30000 >> results_times/my_classes_times.txt
# ./a.out 2000 4000 40000 >> results_times/my_classes_times.txt
# ./a.out 2000 4000 50000 >> results_times/my_classes_times.txt
# ./a.out 2000 4000 60000 >> results_times/my_classes_times.txt
# ./a.out 2000 4000 70000 >> results_times/my_classes_times.txt
# ./a.out 2000 4000 80000 >> results_times/my_classes_times.txt
# ./a.out 2000 4000 90000 >> results_times/my_classes_times.txt
# ./a.out 2000 4000 100000 >> results_times/my_classes_times.txt

#########################

#Boost

g++ -O3 -pthread -march=native -mavx2 -std=c++11 -fopenmp boost/times_tests.cpp
./a.out 2000 4000 10000 >> results_times/boost_times.txt
./a.out 2000 8000 10000 >> results_times/boost_times.txt
./a.out 2000 12000 10000 >> results_times/boost_times.txt
./a.out 2000 16000 10000 >> results_times/boost_times.txt
./a.out 2000 19000 10000 >> results_times/boost_times.txt

./a.out 4000 4000 10000 >> results_times/boost_times.txt
./a.out 4000 8000 10000 >> results_times/boost_times.txt
./a.out 4000 12000 10000 >> results_times/boost_times.txt
./a.out 4000 16000 10000 >> results_times/boost_times.txt
./a.out 4000 19000 10000 >> results_times/boost_times.txt

./a.out 2000 4000 10000 >> results_times/boost_times.txt
./a.out 2000 4000 20000 >> results_times/boost_times.txt
./a.out 2000 4000 30000 >> results_times/boost_times.txt
./a.out 2000 4000 40000 >> results_times/boost_times.txt
./a.out 2000 4000 50000 >> results_times/boost_times.txt
./a.out 2000 4000 60000 >> results_times/boost_times.txt
./a.out 2000 4000 70000 >> results_times/boost_times.txt
./a.out 2000 4000 80000 >> results_times/boost_times.txt
./a.out 2000 4000 90000 >> results_times/boost_times.txt
./a.out 2000 4000 100000 >> results_times/boost_times.txt

#########################

#Eigen

g++ -O3 -pthread -march=native -mavx2 -std=c++11 -fopenmp eigen/times_tests.cpp
./a.out 2000 4000 10000 >> results_times/eigen_times.txt
./a.out 2000 8000 10000 >> results_times/eigen_times.txt
./a.out 2000 12000 10000 >> results_times/eigen_times.txt
./a.out 2000 16000 10000 >> results_times/eigen_times.txt
./a.out 2000 19000 10000 >> results_times/eigen_times.txt

./a.out 4000 4000 10000 >> results_times/eigen_times.txt
./a.out 4000 8000 10000 >> results_times/eigen_times.txt
./a.out 4000 12000 10000 >> results_times/eigen_times.txt
./a.out 4000 16000 10000 >> results_times/eigen_times.txt
./a.out 4000 19000 10000 >> results_times/eigen_times.txt

./a.out 2000 4000 10000 >> results_times/eigen_times.txt
./a.out 2000 4000 20000 >> results_times/eigen_times.txt
./a.out 2000 4000 30000 >> results_times/eigen_times.txt
./a.out 2000 4000 40000 >> results_times/eigen_times.txt
./a.out 2000 4000 50000 >> results_times/eigen_times.txt
./a.out 2000 4000 60000 >> results_times/eigen_times.txt
./a.out 2000 4000 70000 >> results_times/eigen_times.txt
./a.out 2000 4000 80000 >> results_times/eigen_times.txt
./a.out 2000 4000 90000 >> results_times/eigen_times.txt
./a.out 2000 4000 100000 >> results_times/eigen_times.txt

#########################

#Arrays

g++ -O3 -pthread -march=native -mavx2 -std=c++11 -fopenmp arrays/times_tests.cpp
./a.out 2000 4000 10000 >> results_times/arrays_times.txt
./a.out 2000 8000 10000 >> results_times/arrays_times.txt
./a.out 2000 12000 10000 >> results_times/arrays_times.txt
./a.out 2000 16000 10000 >> results_times/arrays_times.txt
./a.out 2000 19000 10000 >> results_times/arrays_times.txt

./a.out 4000 4000 10000 >> results_times/arrays_times.txt
./a.out 4000 8000 10000 >> results_times/arrays_times.txt
./a.out 4000 12000 10000 >> results_times/arrays_times.txt
./a.out 4000 16000 10000 >> results_times/arrays_times.txt
./a.out 4000 19000 10000 >> results_times/arrays_times.txt

./a.out 2000 4000 10000 >> results_times/arrays_times.txt
./a.out 2000 4000 20000 >> results_times/arrays_times.txt
./a.out 2000 4000 30000 >> results_times/arrays_times.txt
./a.out 2000 4000 40000 >> results_times/arrays_times.txt
./a.out 2000 4000 50000 >> results_times/arrays_times.txt
./a.out 2000 4000 60000 >> results_times/arrays_times.txt
./a.out 2000 4000 70000 >> results_times/arrays_times.txt
./a.out 2000 4000 80000 >> results_times/arrays_times.txt
./a.out 2000 4000 90000 >> results_times/arrays_times.txt
./a.out 2000 4000 100000 >> results_times/arrays_times.txt
