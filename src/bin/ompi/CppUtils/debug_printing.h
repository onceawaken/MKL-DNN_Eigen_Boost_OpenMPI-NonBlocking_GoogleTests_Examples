#pragma once

#ifndef DEBUG_PRINTING_H
#define DEBUG_PRINTING_H

#define PPR( format, ... ) fprintf (stdout, format, ##__VA_ARGS__)

#if DEBUG > 0
#define DPPR( format, ... ) fprintf (stderr, format, ##__VA_ARGS__)
#define MDPPR( format, ... ) { fprintf(stderr, ">>> MASTER : "); fprintf (stderr, format, ##__VA_ARGS__); }
#define SDPPR( myrank, format, ... ) { fprintf(stdout, "::: SLAVE (%d): ", myrank); fprintf (stdout, format, ##__VA_ARGS__);}
#else
#define DPPR( format, ... ) {__VA_ARGS__;}
#define MDPPR( format, ... ) {__VA_ARGS__;}
#define SDPPR( myrank, format, ... ) {__VA_ARGS__;}

#endif

#if DEBUG > 1
#define DPPR1( format, ... ) fprintf (stderr, format, ##__VA_ARGS__)
#define MDPPR1( format, ... ) { fprintf(stderr, ">>> Master : "); fprintf (stderr, format, ##__VA_ARGS__);}
#define SDPPR1( myrank, format, ... ) { fprintf(stdout, "::: Slave (%d): ", myrank); fprintf (stdout, format, ##__VA_ARGS__);}
#else
#define DPPR1( format, ... ) {__VA_ARGS__;}
#define MDPPR1( format, ... ) {__VA_ARGS__;}
#define SDPPR1( myrank, format, ... ) {__VA_ARGS__;}
#endif

#if DEBUG > 2
#define DPPR2( format, ... ) fprintf (stderr, format, ##__VA_ARGS__)
#define MDPPR2( format, ... ) { fprintf(stderr, ">>> master : --- "); fprintf (stderr, format, ##__VA_ARGS__);}
#define SDPPR2( myrank, format, ... ) { fprintf(stdout, "::: slave (%d): ... ", myrank); fprintf (stdout, format, ##__VA_ARGS__);}
#else
#define DPPR2( format, ... ) {__VA_ARGS__;}
#define MDPPR2( format, ... ) {__VA_ARGS__;}
#define SDPPR2( myrank, format, ... ) {__VA_ARGS__;}
#endif

#if DEBUG > 3
#define DPPR3( format, ... ) fprintf (stderr, format, ##__VA_ARGS__)
#define MDPPR3( format, ... ) { fprintf(stderr, ">>> >>> master : --- --- "); fprintf (stderr, format, ##__VA_ARGS__);}
#define SDPPR3( myrank, format, ... ) { fprintf(stdout, "::: ::: slave (%d): ... ... ", myrank); fprintf (stdout, format, ##__VA_ARGS__);}
#else
#define DPPR3( format, ... )
#define MDPPR3( format, ... )
#define SDPPR3( myrank, format, ... )
#endif

#if DEBUG > 4
#define DPPR4( format, ... ) fprintf (stderr, format, ##__VA_ARGS__)
#define MDPPR4( format, ... ) { fprintf(stderr, "    > master : --- --- "); fprintf (stderr, format, ##__VA_ARGS__);}
#define SDPPR4( myrank, format, ... ) { fprintf(stdout, "    : slave (%d): ... ... ", myrank); fprintf (stdout, format, ##__VA_ARGS__);}
#else
#define DPPR4( format, ... )
#define MDPPR4( format, ... )
#define SDPPR4( myrank, format, ... )
#endif



#endif //MYMPICUDATEST_DEBUG_PRINTING_H
