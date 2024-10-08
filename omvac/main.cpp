#include "version.h"
#include "io/read.h"
#include <cstdio>


// Entry point.
int main(int argc, const char * argv[])
{
	// Parse argv.
	
	// Instantiate reader.
	io::DatasetReader gdal_reader;
	
	printf("%d.%d.%d\n",
			OMVAC_VERSION_MAJOR,
			OMVAC_VERSION_MINOR,
			OMVAC_VERSION_PATCH);
	return 0;
}

