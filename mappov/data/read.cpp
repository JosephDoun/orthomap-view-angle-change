#include "read.h"


Dataset::Dataset(std::string path, Memory &m) :
/* Single tile default. */
n_tiles{1},
memory{m}
{
	/* GDALDataset. */
	ds = ReadDataset(path);

	/* Set default tile size. */
	t_size_x = ds->GetRasterXSize();
	t_size_y = ds->GetRasterYSize();
}


Dataset::Dataset(std::string /* File path. */  path,
				 uint16_t 	 /* tile width */  tsx,
				 uint16_t 	 /* tile height */ tsy,
				 Memory 	 /* Memory inst */ &m) :
/* Memory. */
memory{m}
{
	/* GDALDataset. */
	ds = ReadDataset(path);
	t_size_x = MIN(tsx, ds->GetRasterXSize());
	t_size_y = MIN(tsy, ds->GetRasterYSize());

	/* Tiles fitting horizontally times tiles fitting vertically. */
	n_tiles = (ds->GetRasterXSize() / t_size_x) *
			  (ds->GetRasterYSize() / t_size_y);
}


// Read dataset at path and handle errors.
GDALDatasetUniquePtr Dataset::ReadDataset(std::string p)
{
	/* Get GDALDatasetH internal handle. */
	auto load = GDALOpen(p.c_str(), GA_ReadOnly);
	
	if (load == NULL)
	{
		throw std::invalid_argument(p.append(" is invalid."));
	}
	
	/* Return specialized unique ptr. */
	return GDALDatasetUniquePtr(GDALDataset::FromHandle(load));
}


/* Datatype containing x, y offset coordinates. */
typedef struct _t_coords { uint16_t x; uint16_t y; } t_coords;


/* Return the pxl coordinates of a tile. */
inline t_coords Dataset::tile_coords(uint16_t index)
{
	t_coords offsets {};

	/* X offset is i'th tile of row times tile size.
	  (Reset if edge passed) */
	offsets.x = index % (ds->GetRasterXSize() / t_size_x) * t_size_x;

	/* Y offset is i'th tile of column times tile size.
	  (Increment if edge passed) */
	offsets.y = index / (ds->GetRasterXSize() / t_size_x) * t_size_y;

	return offsets;
}


/* Tile fetching operator. */
float * Dataset::operator[](uint16_t index)
{
	/* Get preallocated memory block. */
	void * buffer = memory.Allocate();

	t_coords c = tile_coords(index);

	CPLErr ErrNO = ds->RasterIO(GF_Read,
								/* nXOff */ 					c.x,
								/* nYOff */ 					c.y,
								/* nXSize */ 					t_size_x,
								/* nYSize */ 					t_size_y,
			 					/* pData */ 					buffer,
								/* nBufXSize (int) */ 			t_size_x,
			 					/* nBufYSize (int) */ 			t_size_y,
			 					/* eBufType (GDALDataType) */ 	GDT_Float32,
			 					/* nBandCount (int) */ 			ds->GetRasterCount(),
								/* panBandMap (int *) */ 		0,
			 					/* nPixelSpace */ 				0,
								/* nLineSpace */ 				0,
								/* nBandSpace */ 				0,
			 					/* GDALRasterIOExtraArg */ 		nullptr);

	printf("CPLErr %d\n", ErrNO);

	return (float *) buffer;
}
