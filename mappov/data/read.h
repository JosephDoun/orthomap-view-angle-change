# ifndef __READ_DATA
# define __READ_DATA

# include <gdal/gdal_priv.h>
# include "../memory/memory.h"


struct _t_coords;
typedef _t_coords t_coords;


struct Dataset
{
	private:
	/* GDALDataset */
	GDALDatasetUniquePtr ds;
	Memory mem;
	/* fetch-tile dimensions. */
	int t_size_x;
	int t_size_y;
	/* Number of tiles in dataset. */
	int n_tiles;
	/* Function returning tile coordinates in pixels. */
	t_coords tile_coords(uint16_t);

	public:
	Dataset(std::string);
	Dataset(std::string, uint16_t, uint16_t);
	
	float * operator[](uint16_t);

	/* Read a raster from file into a dataset. */
	static GDALDatasetUniquePtr ReadDataset(std::string /* File path. */ p);

	/* Friend functions. */
	friend void Transform(Dataset*, Dataset*, float, float);

};


# endif
