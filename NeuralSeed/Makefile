# Project Name
TARGET = neuralseed

# APP_TYPE = BOOT_SRAM

# Sources
CPP_SOURCES = neuralseed.cpp

# Library Locations
LIBDAISY_DIR = ../libDaisy
DAISYSP_DIR = ../DaisySP

# Core location, and generic Makefile.
SYSTEM_FILES_DIR = $(LIBDAISY_DIR)/core
include $(SYSTEM_FILES_DIR)/Makefile

# Include terrarium.h
C_INCLUDES += -I../Terrarium -I../RTNeural
CPPFLAGS += -DRTNEURAL_DEFAULT_ALIGNMENT=8 -DRTNEURAL_NO_DEBUG=1
