#ifndef __TRANSFORMATION_PROCESSES
#define __TRANSFORMATION_PROCESSES

# include <gdal/gdal_priv.h>
# include "../data/read.h"
# include "../memory/memory.h"

typedef const float * aff_mat;

/*
NOTE This function should encapsulate the geometric transformation.
*/
void Transform(const Dataset*, const Dataset*, const float, const float, Memory&);
void WriteXYZMatrix(float * buffer, float * height, int xsize, int ysize);
aff_mat GetAffineMatrix(float, float);

#endif
