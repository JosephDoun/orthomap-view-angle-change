#include "__main_app.cpp"


// Entry point.
int main(int argc, const char * argv[])
{	
	try
	{
		return __main(argc, argv);
	}
	catch (const std::exception &e)
	{
		printf("%s\n", e.what());
		return EXIT_FAILURE;
	}
	catch (...)
	{
		printf("Error.\n");
		return EXIT_FAILURE;
	}

	return 0;
}

