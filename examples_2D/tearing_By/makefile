CC = nvcc
CFLAGS = -O3 -arch=sm_89 -rdc=true -Xcompiler "/wd 4819"
SRCDIR = ../../lib_ResistiveMHD_2D_GPU_single
SOURCES = $(wildcard $(SRCDIR)/*.cu)
OBJS = $(patsubst $(SRCDIR)/%.cu, $(SRCDIR)/%.o, $(SOURCES))
MAIN = main_tearing_By.o
EXECUTABLE = program.exe

.PHONY: all clean

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS) $(MAIN)
	$(CC) $(CFLAGS) $(OBJS) $(MAIN) -o $(EXECUTABLE)

$(SRCDIR)/%.o: $(SRCDIR)/%.cu
	$(CC) $(CFLAGS) -c $< -o $@

$(MAIN): main_tearing_By.cu
	$(CC) $(CFLAGS) -c $< -o $@

run: $(EXECUTABLE)
	$(EXECUTABLE)

clean:
	rm -f $(OBJS) $(MAIN) $(EXECUTABLE)