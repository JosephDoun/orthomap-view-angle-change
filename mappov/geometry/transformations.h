#ifndef __TRANSFORMATION_PROCESSES
#define __TRANSFORMATION_PROCESSES

# include <gdal/gdal_priv.h>
# include "../data/read.h"


typedef const float * aff_mat;

void Transform(Dataset*, Dataset*, float, float);
aff_mat GetAffineMatrix(float, float);


#endif
