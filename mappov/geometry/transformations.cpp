
#include <openblas/cblas.h>
#include "transformations.h"
#include "../memory/memory.h"
#include "../data/read.h"


typedef const float * aff_mat;

/*
This function should encapsulate the complete action of geometric transformation.
*/
void Transform(const Dataset * lcmap, const Dataset * dsm,
               const float zenith, const float azimuth, Memory& m)
{
    /* Transformation matrix. */
    aff_mat a_mat { GetAffineMatrix(zenith, azimuth) };

    for (uint16_t i {0}; i < lcmap->n_tiles; i++)
    {
        /* Repeat transformation over each tile. 
           1. Read lcmap tile.
           2. Read dsm tile.
           3. Construct affine transformation matrix.
           4. Construct indices matrix of dsm w/ z.
           5. Transform indices matrix.
           6. Resample lcmap.
        */
        printf("Looped in transform.\n");
        float * lcover = (*lcmap)[i];
        printf("Got cover tile.\n");
        float * elev   = ( *dsm )[i];
        printf("Got elevation tile.\n");
        // indices.
    
    }

    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //             lcmap->t_size, 1, 1, 1.0);
}


/* Use the zenith and azimuth angles to construct
   the affine transformation matrix. */
aff_mat GetAffineMatrix(float zenith, float azimuth)
{
    // TODO math.
    static const float Affine [3 * 3] { 1., 0., 0.,
                                        0., 1., 0.,
                                        0., 0., 1. };
    return Affine;
}


/* Fill xyz grid location matrix of dimensions XxYx3 */
void WriteXYZMatrix(float * buffer, float * height, int xsize, int ysize)
{    
    /* Row wise iteration. */
    for (int y {0}; y < ysize; y++)
    {
        /* Column wise iteration. */
        for (int x {0}; x < ysize; x++)
        {
            /* 
            Total buffer array length N = 3 * xsize * ysize.
            */
            *buffer = /* X grid coord. */   x;
            ++buffer;
            *buffer = /* Y grid coord. */   y;
            ++buffer;
            *buffer = height[/* x, y position in dsm */ x /* TODO */];
            ++buffer;
        }
    }
}
