#include "version.h"
#include "data/read.h"
#include "config/args.h"
#include "memory/memory.h"

#include <cstdio>


// Application entry point.
int __main(int argc, const char * argv[])
{	
	
	Args CLIArgs{argc, argv};
	Memory MemPool{2048, 16};

	// Instantiate reader.
	Dataset* lcmap = ReadData(CLIArgs.lcmap);
	Dataset* dsm = ReadData(CLIArgs.dsm);
	
	printf("%d.%d.%d\n",
			__MAPPOV_VERSION_MAJOR,
			__MAPPOV_VERSION_MINOR,
			__MAPPOV_VERSION_PATCH);

	return 0;
}

