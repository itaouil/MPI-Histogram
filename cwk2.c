//
// Starting code for the MPI coursework.
//
// Compile with:
//
// mpicc -Wall -o cwk1 cwk1.c
//
// or use the provided makefile.
//

//
// Includes
//
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Some extra routines for this coursework. DO NOT MODIFY OR REPLACE THESE ROUTINES,
// as this file will be replaced with a different version for assessment.
#include "cwk2_extras.h"

//
// Main
//
int main( int argc, char **argv )
{
	// Initialise counter index
	int i;

	// Initialise MPI and get the rank and no. of processes.
	int rank, numProcs;
	MPI_Init( &argc, &argv );
	MPI_Comm_size( MPI_COMM_WORLD, &numProcs );
	MPI_Comm_rank( MPI_COMM_WORLD, &rank     );

	// MPI status
	MPI_Status status;

	// Read in the image file to rank 0.
	int *image = NULL, maxValue, pixelsPerProc, dataSize;
	if( rank==0 )
	{
		// Read in the file and extract the maximum grey scale value and the data size (including padding bytes).
		// Defined in cwk2_extras.h; do not change, although feel free to inspect.
		image = readImage( "image.pgm", &maxValue, &dataSize, numProcs );
		if( image==NULL )
		{
			MPI_Finalize();
			return EXIT_FAILURE;
		}

		// The image size has already been rounded up to a multiple of numProcs by "readImage()".
		pixelsPerProc = dataSize / numProcs;

		printf( "Rank 0: Read in PGM image array of size %d (%d per process), with max value %d.\n", dataSize, pixelsPerProc, maxValue );
	}

	// Allocate memory for the final histogram on rank 0.
	int *combinedHist = NULL;
 	if( rank==0 )
 	{
 		combinedHist = (int*) malloc( (maxValue+1)*sizeof(int) );
 		if( !combinedHist ) return allocateFail( "global histogram", rank );

 		for( i=0; i<maxValue+1; i++ ) combinedHist[i] = 0;
 	}

	// Broadcast pixelsPerProc and maxValue to all the processes
	MPI_Bcast(&pixelsPerProc, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&maxValue, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// Allocate localImage and localHist for each process
	int *localImage = (int*) malloc(sizeof(int) * pixelsPerProc);
	int *localHist = (int*) malloc(sizeof(int) * (maxValue +  1));

	// Distribute image's pixels
	// to the available processes
	MPI_Scatter(image,
	    pixelsPerProc,
	    MPI_INT,
	    localImage,
	    pixelsPerProc,
	    MPI_INT,
	    0,
	    MPI_COMM_WORLD
	);

	// Copy image's portion received
	// to the localHist
	for(i=0; i<pixelsPerProc; i++)
	{
		localHist[localImage[i]] += 1;
	}

	// // Send back localHist to combinedHist
	// MPI_Reduce(
	// 	localHist,
	// 	combinedHist,
	// 	maxValue+1,
	// 	MPI_INT,
	// 	MPI_SUM,
	// 	0,
    //     MPI_COMM_WORLD
	// );

	// Point to point communication states,
	// the initial level tree and the number
	// of active processes (involved in the computation)
	// at each time.
	int lev=1;
	int activeProcs = numProcs;

	// Declare remoteHist which
	// stores point to point data
	int *remoteHist = NULL;

	while(1<<lev<=numProcs)
	{

		// Send localHist if rank is qualified for sending
		if (rank >= activeProcs/2 && rank < numProcs) {
			MPI_Send(
			    localHist,
			    maxValue+1,
			    MPI_INT,
			    rank-(activeProcs/2),
			    0,
			    MPI_COMM_WORLD
			);
		}

		// Receive remoteHist if rank is qualified for receiving
		else if (rank < activeProcs/2) {
			// Allocate space for remoteHist (only allocated for receivers)
			remoteHist = (int*) malloc(sizeof(int) * (maxValue +  1));

			MPI_Recv(
			    remoteHist,
			    maxValue+1,
			    MPI_INT,
			    rank+(activeProcs/2),
			    0,
			    MPI_COMM_WORLD,
				MPI_STATUS_IGNORE
			);

			// Compute new localHist
			for(i=0; i<maxValue+1; i++)
			{
				localHist[i] += remoteHist[i];
			}
		}

		// Update activeProcs
		activeProcs /= 2;

		// Update level
		lev++;
	}

	// Do not free rank0 localHist
	// as it still needed for final
	// computation
	if (rank != 0) {
		free(localHist);
	}

	free(remoteHist);
	free(localImage);

	//
	// Constructs the histogram in serial on rank 0. Can be used as part of a check that your parallel version works.
	//
	if( rank==0 )
	{
		// Copy over localHist to combinedHist
		for(i=0; i<maxValue+1; i++)
		{
			combinedHist[i] += localHist[i];
		}

		free(localHist);

		// Allocate memory for the check histogram, and then initialise it to zero.
		int *checkHist = (int*) malloc( (maxValue+1)*sizeof(int) );
		if( !checkHist ) return allocateFail( "histogram for checking", rank );
		for( i=0; i<maxValue+1; i++ ) checkHist[i] = 0;

		// Construct the histogram using the global data only.
		for( i=0; i<dataSize; i++ )
			if( image[i]>=0 ) checkHist[image[i]]++;

		// Display the histgram.
		for( i=0; i<maxValue+1; i++ )
			printf( "Greyscale value %i:\tCount %i\t(check: %i)\n", i, combinedHist[i], checkHist[i] );

		free( checkHist );
	}

	//
	// Clear up and quit.
	//
	if( rank==0 )
	{
		saveHist( combinedHist, maxValue );		// Defined in cwk2_extras.h; do not change or replace the call.
		free( image );
		free( combinedHist );
	}

	MPI_Finalize();
	return EXIT_SUCCESS;
}
