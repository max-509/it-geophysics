#ifndef _ARRAY3D
#define _ARRAY3D

#include <cstddef>

template <typename T>
class Array3D {
public:
	Array3D(T* p_array, ptrdiff_t z_dim, ptrdiff_t y_dim, ptrdiff_t x_dim) 
		: _p_array(p_array), _z_dim(z_dim), _y_dim(y_dim), _x_dim(x_dim)  
	{ }

	Array3D(const Array3D<T>&) = delete;
	Array3D& operator=(const Array3D<T>&) = delete;

	const T& operator()(ptrdiff_t z, ptrdiff_t y, ptrdiff_t x) const {
		return _p_array[z*_y_dim*_x_dim+y*_x_dim+x];
	}

	T& operator()(ptrdiff_t z, ptrdiff_t y, ptrdiff_t x) {
		return _p_array[z*_y_dim*_x_dim+y*_x_dim+x];
	}

	ptrdiff_t get_z_dim() const { return _z_dim; }
	ptrdiff_t get_y_dim() const { return _y_dim; }
	ptrdiff_t get_x_dim() const { return _x_dim; }

private:
	T* _p_array;
	ptrdiff_t _z_dim;
	ptrdiff_t _y_dim;
	ptrdiff_t _x_dim;
};

#endif /*_ARRAY3D*/