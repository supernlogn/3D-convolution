# 3D Convolution

## Introduction
Searched all over and did'n find it. So here it is:
A library for doing n-Dimensional convolution in C/C++ without using any
special structures or frameworks. It uses fourier transform in order to
be fast and simple to understand. So it has only one minimum requirement the **fftw3** library.
It is also no `malloc` free, so stdlib.h or another library with memory allocation is
required.

## Requirements
You need to have installed the fftw3 pack for developers. 
If not its OK! A guide bellow is provided:

## How to build
Debian-oids:
1. Type on a terminal ``` sudo apt-get install libfftw3-dev```
2. Download this library. Just type: ```git clone https://github.com/supernlogn/3D-convolution.git```
3. Last but not the least: on the makefile change this line: https://github.com/supernlogn/3D-convolution/blob/master/Makefile#L2
4. Type on a terminal ``` make ```

Windows-oids:
To be honest: no tries in windows were done. 
But the steps would be something like this:

1. Go to this link: http://www.fftw.org/install/windows.html
2. Download this library. Just type: ```git clone https://github.com/supernlogn/3D-convolution.git```
3. Note: The makefile is for unix systems... (sorry) 
   You have to use an IDE or write your own makefile
