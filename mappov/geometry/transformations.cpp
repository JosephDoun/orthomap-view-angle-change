
#include <openblas/cblas.h>
#include "transformations.h"
#include "../memory/memory.h"
#include "../data/read.h"


typedef const float * aff_mat;

/*
This function should encapsulate the complete action of geometric transformation.
*/
void Transform(Dataset * lcmap, Dataset * dsm, float zenith, float azimuth)
{
    printf("Entered transform.\n");
    printf("tile num: %d\n", lcmap->n_tiles);

    aff_mat a_mat { GetAffineMatrix(zenith, azimuth) };
    printf("Got affine matrix of size %d.\n", sizeof(a_mat));

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
    printf("Constructed affine matrix.\n");
    return Affine;
}


/* Return a location index matrix with dimensions Nx3 */
float * constructIdxMatrix() { return NULL; }
