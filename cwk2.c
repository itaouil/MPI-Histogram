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

	// Broadcast pixelsPerProc and maxValue data
	// to all the processes
	MPI_Bcast(&pixelsPerProc, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&maxValue, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// Initialise local processes' variables
	int *localHist = NULL;
	int *localImage = NULL;

	// Allocate localImage per process
	localImage = (int*) malloc(sizeof(int) * pixelsPerProc);

	// Allocate localHist per process
	localHist = (int*) malloc(sizeof(int) * (maxValue +  1));

	// Scatter image portions to processes
	MPI_Scatter(
	    image,
	    pixelsPerProc,
	    MPI_INT,
	    localImage,
	    pixelsPerProc,
	    MPI_INT,
	    0,
	    MPI_COMM_WORLD
	);

	// Increase localHist counter
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

	// Intialise the starting
	// point of the binary communication
	// the jump that the sending process
	// does to send the data and the prevSplit
	// which serves as boundary checking for the
	// sender processes.
	int lev=1;
	int jump=numProcs/2;
	int prevSplit = numProcs;

	// Intialise space for remoreHist (sent one)
	int *remoteHist = NULL;

	// Allocate localHist per process
	remoteHist = (int*) malloc(sizeof(int) * (maxValue +  1));

	while(1<<lev<=numProcs)
	{

		// Send localHist if rank is qualified for sending
		if (rank >= numProcs/(lev*2) && rank < prevSplit) {
			MPI_Send(
			    localHist,
			    maxValue+1,
			    MPI_INT,
			    rank-jump,
			    0,
			    MPI_COMM_WORLD
			);

		}

		// Receive remoteHist if rank is qualified for receiving
		else if (rank < numProcs/(lev*2)) {
			MPI_Recv(
			    remoteHist,
			    maxValue+1,
			    MPI_INT,
			    rank+jump,
			    0,
			    MPI_COMM_WORLD,
				MPI_STATUS_IGNORE
			);

			// Compute new localHist
			for(i=0; i<maxValue; i++)
			{
				localHist[i] += remoteHist[i];
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);

		// Update jump
		jump /= 2;

		// Update prevSplit
		prevSplit = numProcs/(lev*2);

		// Update level
		lev++;
	}

	//
	// Constructs the histogram in serial on rank 0. Can be used as part of a check that your parallel version works.
	//
	if( rank==0 )
	{
		// Copy over localHist to combinedHist
		combinedHist = localHist;

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
