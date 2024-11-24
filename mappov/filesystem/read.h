# ifndef __READ
# define __READ
# include <gdal/gdal_priv.h>


	
class DatasetReader
{

	public:
		
	DatasetReader();
	~DatasetReader();

	// Read dataset and handle errors
	GDALDataset * read(std::string p);

	private:
	std::vector<GDALDataset*> data;
};


# endif

