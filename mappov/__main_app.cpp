#include "version.h"
#include "data/read.h"
#include "config/args.h"
#include "memory/memory.h"

#include <cstdio>


/* Application entry point. */
int __main(int argc, const char * argv[])
{
	/* Register available drivers. */
	GDALAllRegister();

	Args CLIArgs{argc, argv};
	Memory MemPool{2048, 16};

	/* Load datasets. */
	Dataset* lcmap = Dataset::ReadDataset(CLIArgs.lcmap)->SetTSize(100);
	Dataset* dsm   = Dataset::ReadDataset(CLIArgs.dsm)->SetTSize(100);

	/* Send datasets to transformation process. */

	/* Debug logging. */
	printf("%d\n", lcmap->GetRasterCount());
	
	printf("%d.%d.%d\n",
			__MAPPOV_VERSION_MAJOR,
			__MAPPOV_VERSION_MINOR,
			__MAPPOV_VERSION_PATCH);

	return 0;
}

