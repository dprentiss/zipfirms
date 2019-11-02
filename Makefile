SHELL = /bin/sh

CC = gcc
NVCC = nvcc
SRC = src
BIN = bin

.PHONY: all

.PHONY: zipfirms
zipfirms:
	$(RM) $(BIN)/$@
	mkdir -p $(BIN)
	$(NVCC) $(SRC)/zipfirms.cu -o $(BIN)/$@
