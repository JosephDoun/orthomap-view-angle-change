# ifndef __READ
# define __READ
# include <gdal/gdal_priv.h>


typedef struct t_coords;


struct Dataset: public GDALDataset
{
	private:
	uint16_t t_size;
	t_coords & tile_coords(uint16_t);

	public:
	Dataset(uint16_t tile_s);

};


Dataset * ReadData(std::string p);


# endif
