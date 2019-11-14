#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <complex.h>
#include <string.h>

#include <time.h>

#include <fftw3.h>

#include "cfftlog.h"
#include "utils.h"
#include "utils_complex.h"

int main(int argc, char const *argv[])
{
	config my_config;
	double ell = 1.;
	my_config.nu = 1.01;
	my_config.c_window_width = 0.25;
	my_config.derivative = 0;
	my_config.N_pad = 0;

	char filename[] = "f1_chi.txt";
	FILE *IN = fopen(filename, "r");

	// double *ell, *fl;
	long Nk = 1000;
	double k[Nk], fk[Nk];

	long linenum = 0;
	while(!feof(IN) && (linenum<Nk)) {
		fscanf(IN, "%lg %lg", &k[linenum], &fk[linenum]);
		linenum++;
	}
	printf("linenum, %ld\n", linenum);
	double dlnk = log(k[1]/k[0]);
	int i,j;

	double *r, *result;
	r = malloc(Nk * sizeof(double));
	result = malloc(Nk * sizeof(double));
	for(i=0; i<Nk; i++) {
		result[i] = 0.;
	}
	printf("k:%lg, %lg, %lg\n", k[0], k[1],k[Nk-2]);
	clock_t start = clock();
	cfftlog(k, fk, Nk, &my_config, ell, r, result);
	printf("r:%lg, %lg\n", r[0], r[1]);
	printf("result:%lg, %lg\n", result[0], result[1]);

	clock_t end = clock();
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("time:%f\n", seconds);

	char outfilename[] = "test_1.txt";
	FILE *OUT = fopen(outfilename, "w");
	
	for(i=0; i<Nk; i++) {
		fprintf(OUT, "%lg %lg", r[i], result[i]);
		fprintf(OUT, "\n");
	}
	fclose(OUT);
	fclose(IN);
	free(r);
	free(result);

	return 0;
}