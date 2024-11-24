#include "version.h"
#include "filesystem/read.h"
#include "config/args.h"

#include <cstdio>


// Entry point.
int main(int argc, const char * argv[])
{	
	
	Args cliargs{argc, argv};

	// Instantiate reader.
	DatasetReader reader;

	GDALDataset * map = reader.read(cliargs.lcmap);
	GDALDataset * dem = reader.read(cliargs.dem);
	
	printf("%d.%d.%d\n",
			__MAPPOV_VERSION_MAJOR,
			__MAPPOV_VERSION_MINOR,
			__MAPPOV_VERSION_PATCH);

	return 0;
}

