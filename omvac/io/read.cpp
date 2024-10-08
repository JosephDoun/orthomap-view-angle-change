#include "read.h"
#include <gdal/gdal_priv.h>


using namespace io;


DatasetReader::DatasetReader() {}
DatasetReader::~DatasetReader() {}


// Read dataset at path and handle errors.
GDALDataset * DatasetReader::read(const char *p)
{
	GDALDataset * ds = (GDALDataset *) GDALOpen(p, GA_ReadOnly);
	return ds;
}


