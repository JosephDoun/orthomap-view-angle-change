#include "version.h"
#include "filesystem/read.h"
#include "config/args.h"
#include "memory/memory.h"

#include <cstdio>


// Application entry point.
int __main(int argc, const char * argv[])
{	
	
	Args cliargs{argc, argv};
	Memory mempool{2048, 16};

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

