#include "read.h"


using namespace io;


DatasetReader::DatasetReader() {}
DatasetReader::~DatasetReader() {}

GDALDataset * DatasetReader::read(const char *p)
{
	GDALDataset * ds = (GDALDataset *) GDALOpen(p, GA_ReadOnly);
	return ds;
}


