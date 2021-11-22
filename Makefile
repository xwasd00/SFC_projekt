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
	./$(PROJ)

clean:
	rm -f $(PROJ)
