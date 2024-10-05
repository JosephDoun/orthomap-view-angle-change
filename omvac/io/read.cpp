#include "read.h"


data::DatasetReader::DatasetReader() {}
data::DatasetReader::~DatasetReader() {}

GDALDataset * data::DatasetReader::read(const char *p)
{
	GDALDataset * ds = (GDALDataset *) GDALOpen(p, GA_ReadOnly);
	return ds;
}

