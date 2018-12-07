# compilo con make -f punto1.mk

*.pdf : *.dat plots.py
	python plots.py

*.dat : ejercicio1_c
	./ejercicio1_c

punto1_c : punto1.c
	gcc -fopenmp ejercicio1.c -lm -o ejercicio1_c

clean :
	rm *.dat  *_c