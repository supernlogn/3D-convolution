# Type where your fftw3.h was installed
INCS = -I/usr/include/

INCS += -Isrc/

LIBS = -lfftw3
# if we want to be used as library
FLAGS = -fPIC 
# to get all the warnings
FLAGS += -Wall

CC = gcc

LIBRARY_SRC = src/convolution.c
EXAMPLE_SRC = conv_example.c

all: example.bin conv3d.so

conv3d.so: $(LIBRARY_SRC)
	$(CC) -shared $(FLAGS) $(LIBS) $(LIBRARY_SRC) -o $@

example.bin: $(EXAMPLE_SRC)
	$(CC) $(FLAGS) $(LIBRARY_SRC) $(EXAMPLE_SRC) -o example.bin $(LIBS)

run_example: example.bin
	./example.bin

clean:
	rm -f example.bin conv3d.so
