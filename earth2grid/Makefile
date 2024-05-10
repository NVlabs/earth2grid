# The compiler.
# CXX = $(CUDA_HOME)/bin/nvcc

# Lib(s) include address
LIBSINC ?= .

# Lib(s) link name
LIBS ?=

# Optimization and Debugging
OPTFLAGS ?= -O3 -g
#OPTFLAGS ?= -O0 -g

# Source files
SRC_FILES = regridding/test_interpolation.cpp third_party/healpy_bare/healpix_bare.c

# Object Files
# OBJ_FILES_CXX := $(SRC_FILES:.cpp=.o)
# OBJ_FILES := $(OBJ_FILES_CXX:.c=.o)
OBJ_FILES = test_interpolation.o healpix_bare.o

# CXX flags
CXXFLAGS = $(OPTFLAGS) -std=c++20 -Wall
CXXFLAGS += $(foreach lib,$(LIBSINC), \
	       -I $(lib) )

# C flags
CFLAGS = $(OPTFLAGS) -Wall
CFLAGS += $(foreach lib,$(LIBSINC), \
	       -I $(lib) )


LINKFLAGS = $(CXXFLAGS)
LINKFLAGS += $(foreach lib,$(LIBS), \
	       -l$(lib) )

DEFAULT: interpolation

all = interpolation

interpolation: $(OBJ_FILES)

# Include the dependencies that were created by %.d rule.
#
DEPS_CXX := $(SRC_FILES:.cpp=.d)
DEPS := $(DEPS_CXX:.c=.d)
ifneq ($(MAKECMDGOALS),clean)
-include $(DEPS)
endif
#

# Prepare file holding dependencies, to be included in this file.
#

%.d: %.cpp %.c Makefile
	@set -e; rm -f $@; \
	$(CXX) -DMAKE_DEPEND -M $(CXXFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

test_interpolation.o: healpix_bare.o regridding/test_interpolation.cpp Makefile
	$(CXX) $(CXXFLAGS) -c regridding/test_interpolation.cpp

healpix_bare.o: third_party/healpy_bare/healpix_bare.c Makefile
	$(CC) $(CFLAGS) -c third_party/healpy_bare/healpix_bare.c

$(all):%:
	$(CXX) $(LINKFLAGS) -o $@ $^

clean:
	rm -f $(OBJ_FILES) *.o *.d interpolation \
	*.d.[0-9][0-9][0-9][0-9][0-9] *.d.[0-9][0-9][0-9][0-9] \
	*.d.[0-9][0-9][0-9] *.d.[0-9][0-9][0-9][0-9][0-9][0-9] \
	*~
