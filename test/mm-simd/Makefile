TARGET=mm

mm-simd:
	$(CC) -lm -mavx -mavx2 mm-smid.c -o mm-simd -g

mm-tile: mm-tile.c
	$(CC) -lm -mavx -mavx2 mm-tile.c -o mm-tile -g

mm: mm.c
	$(CC) -lm -mavx -mavx2 mm.c -o mm -g


CLEAN_EXTRA=viz


