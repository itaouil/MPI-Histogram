//
// Routines for use in the coursework. These routines MUST be called as provided;
// DO NOT MODIFY THEM OR REPLACE WITH YOUR OWN VERSION.
//


//
// Reads in a image file in PGM format (ASCII, greyscale). Returns with a pointer to
// the pixel array (which will need to be deallocated), or NULL if failed.
// Also pass a padding boundary; the image will be sized upwards to a multiple of this
// value, with values of -1 inserted for padding bytes.
//
int* readImage( char *fname, int *max, int *size, int paddingBoundary )
{
	// Try to open the image file.
	FILE *file = fopen( fname, "r" );
	if( !file )
	{
		printf( "Could not open image '%s' file for reading.\n", fname );
		return NULL;
	}

	// The first line should be "P2"; provide some basic error checking.
	char header[10];
	fscanf( file, "%s", header );
	if( header[0]!='P' || header[1]!='2' )
	{
		printf( "File not greyscale ASCII pgm.\n" );
		return NULL;
	}

	// The next two numbers are the image dimensions.
	int X, Y;
	fscanf( file, "%i %i", &X, &Y );

	// Get the actual number of pixels, and the image size, rounded up if necessary.
	int numPixels = X*Y;
	*size = paddingBoundary*( (numPixels+paddingBoundary-1)/paddingBoundary );

	// The next number is the grey scale, i.e. the value corresponding to white.
	fscanf( file, "%i", max );

	// Allocate memory for the pixel values. This will need to be allocated by the calling routine.
	int *image = (int*) malloc( numPixels*sizeof(int) );
	if( !image )
	{
		printf( "Could not allocate memory for the image.\n" );
		return NULL;
	}

	// Read in all of the pixels, with -1 for any padding values.
	int i;
	for( i=0; i<*size; i++ )
		if( i < numPixels )
			fscanf( file, "%i", &image[i] );
		else
			image[i] = -1;

	// Close the file.
	fclose(file);

	// Return with the pointer to the pixels.
	return image;
}


// Short function to display a memory allocate fail message, ends MPI, and returns -1.
// This may help generate more readable error messages compared to if MPI_Finalize()
// was not called. It also tidy's up the code a little.
int allocateFail( char *type, int rank )
{
	printf( "Could not allocate memory for %s on rank %d.\n", type, rank );
	MPI_Finalize();
	return EXIT_FAILURE;
}


// Saves the histogram to file.
void saveHist( int *hist, int maxValue )
{
	FILE *file = fopen( "hist.out", "w" );
	if( !file )
	{
		printf( "Could not open the file for output.\n" );
		return;
	}

	int i;
	for( i=0; i<maxValue+1; i++ ) fprintf( file, "%d %d\n", i, hist[i] );

	fclose( file );
}
