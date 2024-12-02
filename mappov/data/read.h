# ifndef __READ
# define __READ
# include <gdal/gdal_priv.h>


struct _t_coords;
typedef _t_coords t_coords;


struct Dataset: public GDALDataset
{
	private:
	uint16_t t_size;
	uint16_t n_tiles;
	t_coords tile_coords(uint16_t);

	public:
	/* Tile size setter function. */
	Dataset * SetTSize(uint16_t);

	/* Read a raster from file into a dataset. */
	static Dataset * ReadDataset(std::string /* File path. */ p);

	friend void Transform(Dataset*, Dataset*, float, float);

};


# endif
