# ifndef __READ
# define __READ
# include <gdal/gdal_priv.h>


namespace io {
	
	class DatasetReader {
		
		public:
		
		DatasetReader();
		~DatasetReader();

		// Read dataset and handle errors
		GDALDataset * read(const char *p); 
	};

} // namespace

# endif

