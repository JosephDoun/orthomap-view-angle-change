#include "data/read.h"
#include "config/args.h"
#include "memory/memory.h"
#include "geometry/transformations.h"
#include <cstdio>


/* Application entry point. */
int __main(int argc, const char * argv[])
{
	/* Register available drivers. */
	GDALAllRegister();

	Args CLIArgs{argc, argv};
	Memory main_mem;

	/* Setup necessary memory allocations. */
	main_mem.Setup(100 * 100 * GDALGetDataTypeSizeBytes(GDT_Float32), 10);

	/* Load datasets. */
	std::unique_ptr<Dataset> lcmap(new Dataset{CLIArgs.lcmap, 5, 5, main_mem});
	std::unique_ptr<Dataset>   dsm(new Dataset{CLIArgs.dsm, 5, 5, main_mem});

	/* Send datasets to transformation process. */
	Transform(lcmap.get(), dsm.get(), CLIArgs.zenith, CLIArgs.azimuth);

	/* Debug logging. */
	printf("%d\n", GDALGetDataTypeSizeBytes(GDT_Float32));
	
	// printf("%d, %d\n", lcmap->GetRasterXSize(), dsm->GetRasterYSize());

	return EXIT_SUCCESS;
}

