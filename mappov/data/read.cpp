#include "read.h"
#include <gdal/gdal_priv.h>


Dataset::Dataset(uint16_t tile_size) : t_size{tile_size} {};

// Read dataset at path and handle errors.
Dataset * ReadData(std::string p)
{
	auto load = GDALOpen(p.c_str(), GA_ReadOnly);
	if (load == NULL) throw std::invalid_argument(p + " is invalid.");
	return (Dataset *) load;
}

struct t_coords { uint16_t x; uint16_t y; };

/* Return the pxl coordinates of a tile. */
t_coords & Dataset::tile_coords(uint16_t index)
{
	t_coords offsets {};

	/* X offset is i'th tile of row times tile size.
	  (Reset if edge passed) */
	offsets.x = index % (GetRasterXSize() / t_size) * t_size;

	/* Y offset is i'th tile of column times tile size.
	  (Increment if edge passed) */
	offsets.y = index / (GetRasterXSize() / t_size) * t_size;

	return offsets;
}
