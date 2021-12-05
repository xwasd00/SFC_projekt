CC=g++
CFLAGS=-std=c++17 -Wall -pedantic
LD=
SRC=*.cpp
PROJ=main
LOGIN=xsovam00

.PHONY:$(PROJ)

$(PROJ):
	$(CC) $(CFLAGS) $(SRC) $(LD) -o $(PROJ)

run:$(PROJ)
	#./$(PROJ) --save "w.test" --topology "B8-Spiral-topology.txt" --train "B8-Spiral.dta" --test "B8-Spiral-test.dta" -i 30000 -m 0.7 -e 0.1 -d3
	./$(PROJ) --save "w.test" --topology "xor-topology.txt" --train "xor.txt" --test "xor-test.txt" -e 0.2

clean:
	rm -f $(PROJ)

zip:
	zip $(LOGIN).zip *.cpp *.hpp xor* B8* sfc.pdf Makefile
