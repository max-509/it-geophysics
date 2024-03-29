#include "coherent_summation.h"

#include <cstdint>
#include <numpy/arrayobject.h>
#include <Python.h>

static PyObject *CompError;

static PyObject* compute_coherent_summation(PyObject* self, PyObject* args) {
	PyObject *arg1 = nullptr, *arg2 = nullptr, *arg3 = nullptr;

	if (!PyArg_ParseTuple(args, "OOO", &arg1, &arg2, &arg3)) {
		return nullptr;
	}

	PyObject *rec_samples = nullptr, *rec_coords = nullptr, *sources_times = nullptr;
	rec_samples = PyArray_FROM_OTF(arg1, NPY_FLOAT32, NPY_ARRAY_ALIGNED);
	if (rec_samples == nullptr) {
		return nullptr;
	}
	rec_coords = PyArray_FROM_OTF(arg2, NPY_FLOAT32, NPY_ARRAY_ALIGNED);
	if (rec_coords == nullptr) {
		return nullptr;
	}
	sources_times = PyArray_FROM_OTF(arg3, NPY_INT32, NPY_ARRAY_ALIGNED);
	if (sources_times == nullptr) {
		return nullptr;
	}

	// const npy_intp n_rec = PyArray_DIMS(rec_samples)[0];
	const npy_intp n_samples = PyArray_DIMS(rec_samples)[1];

	const npy_intp n_rec = PyArray_DIMS(rec_coords)[0];
	if (PyArray_DIMS(rec_coords)[1] != 3) {
		PyErr_SetString(CompError, "Incorrect receiver coordinates data");
		return nullptr;
	}

	if (n_rec != PyArray_DIMS(rec_samples)[0]) {
		PyErr_SetString(CompError, "The number of receivers in the array of samples and in the array of coordinates does not match");
		return nullptr;
	}

	const npy_intp n_xyz = PyArray_DIMS(sources_times)[0];
	if (PyArray_DIMS(sources_times)[1] != n_rec) {
		PyErr_SetString(CompError, "Incorrect area coordinates data");
		return nullptr;
	} 

	float *rec_samples_data = (float*)PyArray_DATA(rec_samples);
	float *rec_coords_data = (float*)PyArray_DATA(rec_coords);
	int32_t *sources_times_data = (int32_t*)PyArray_DATA(sources_times);

	npy_intp result_dims[2] = {n_xyz, n_samples};
	PyObject *result_arr = PyArray_ZEROS(2, result_dims, NPY_FLOAT32, 0);

	if (result_arr == nullptr) {
		return nullptr;
	}

	float* result_arr_data = (float*)PyArray_DATA(result_arr);

	compute(rec_samples_data, rec_coords_data, sources_times_data, n_samples, n_rec, n_xyz, result_arr_data);

	return result_arr;

}

static PyMethodDef methods[] = 
{
	{"computeCoherentSummation", compute_coherent_summation, METH_VARARGS, "Coherent Summation"},
	{nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef module = 
{
	PyModuleDef_HEAD_INIT,
	"coherentSummation",
	"Effective parallel coherent summation with vectorization",
	0,
	methods
};

PyMODINIT_FUNC 
PyInit_CoherentSumModule() {

	import_array();

	PyObject* m = nullptr;
	m = PyModule_Create(&module);
	if (m == nullptr) return nullptr;
	CompError = PyErr_NewException("compute.error", nullptr, nullptr);
	Py_INCREF(CompError);
	PyModule_AddObject(m, "error", CompError);

	return m;
}