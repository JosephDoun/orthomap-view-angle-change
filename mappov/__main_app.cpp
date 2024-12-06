#include "data/read.h"
#include "config/args.h"
#include "memory/memory.h"
#include "geometry/transformations.h"
#include <cstdio>


/* Global memory instance. */
Memory main_mem;


/* Application entry point. */
int __main(int argc, const char * argv[])
{
	/* Register available drivers. */
	GDALAllRegister();

	Args CLIArgs{argc, argv};
		
	/* Load datasets. */
	std::unique_ptr<Dataset> lcmap(new Dataset{CLIArgs.lcmap});
	std::unique_ptr<Dataset>   dsm(new Dataset{CLIArgs.dsm});

	/* Send datasets to transformation process. */

	/* Debug logging. */
	printf("%d\n", lcmap->GetRasterCount());
	
	printf("%d.%d.%d\n",
			__MAPPOV_VERSION_MAJOR,
			__MAPPOV_VERSION_MINOR,
			__MAPPOV_VERSION_PATCH);

	return EXIT_SUCCESS;
}

