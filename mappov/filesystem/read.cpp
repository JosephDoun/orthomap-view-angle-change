#include "read.h"
#include <gdal/gdal_priv.h>


DatasetReader::DatasetReader() {}
DatasetReader::~DatasetReader() {}


// Read dataset at path and handle errors.
GDALDataset * DatasetReader::read(std::string p)
{
	data.push_back((GDALDataset *) GDALOpen(p.c_str(), GA_ReadOnly));
	return data.back();
}

