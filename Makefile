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
	./$(PROJ) --save "w.test" --topology "xor-topology.txt" --train "xor.txt" --test "xor-test.txt" -e 0.2 -d 3

clean:
	rm -f $(PROJ)
