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

	/* Tile dimensions for processing.
	   Should be set via command line. */
	uint16_t t_x_size{170}, t_y_size{150};

	/*
	Setup necessary memory allocations multiple of 32 bytes.
	*/
	main_mem.Setup(t_x_size * t_y_size * 32, 3);

	/* Load datasets. */
	std::unique_ptr<const Dataset>
	lcmap(new Dataset{CLIArgs.lcmap, t_x_size, t_y_size, main_mem});
	
	std::unique_ptr<const Dataset>
	dsm(new Dataset{CLIArgs.dsm, t_x_size, t_y_size, main_mem});

	/* Send datasets to transformation process. */
	Transform(lcmap.get(), dsm.get(), CLIArgs.zenith, CLIArgs.azimuth, main_mem);

	/* Debug logging. */
	printf("%d\n", GDALGetDataTypeSizeBytes(GDT_Float32));
	
	// printf("%d, %d\n", lcmap->GetRasterXSize(), dsm->GetRasterYSize());

	return EXIT_SUCCESS;
}

