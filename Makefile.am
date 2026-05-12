# -*- Makefile -*-
#
#

AM_CPPFLAGS = -I$(top_builddir) -I$(top_srcdir) -I$(top_srcdir)/include

ACLOCAL_AMFLAGS = -I acinclude

dist_bin_SCRIPTS = \
  hgcclib.py \
  hgccutils.py \
  hgcompile.py \
  hglink.py \
  configlib.py

bin_SCRIPTS = \
  hgccvars.py \
  hgcc \
  hg++

AM_LDFLAGS = -ldl

EXTRA_DIST = clang hgcc_include support

# MV2 (and future --mpi=<impl>) link-time support sources consumed by hgcc at
# link time when --mpi=mvapich2 is passed. Installed to $(datadir)/hgcc/mv2.
mv2supportdir = $(datadir)/hgcc/mv2
dist_mv2support_DATA = \
  support/mv2/mv2_mpi_wrappers.c \
  support/mv2/mv2_ld_glibc_stub.c \
  support/mv2/gen_register_shim.py

if HAVE_CLANG

bin_PROGRAMS = ssthg_clang

ssthg_clang_SOURCES = \
  clang/util.cc \
  clang/main.cc \
  clang/astVisitorGlobalVars.cc \
  clang/pragmas.cc \
  clang/computePragma.cc \
  clang/computeVisitor.cc \
  clang/replacePragma.cc \
  clang/astConsumers.cc \
  clang/frontendActions.cc \
  clang/astVisitor.cc 

if USE_REPLACEMENT_HEADERS
	nobase_library_include_HEADERS += \
  replacements/time.h 
#   replacements/sys/types.h 
#  	replacements/linux/mmtimer.h \
#  	replacements/linux/limits.h \
#  replacements/clear_symbol_macros.h \
#  replacements/return_symbol_macros.h \
  #replacements/sstmac_pthread_return.h \
#  replacements/sys/sendfile.h \
#  replacements/dlfcn.h \
  #replacements/future \
  #replacements/fstream \
  #replacements/iostream \
  #replacements/ios \
 # replacements/ostream \
 # replacements/istream \
  #replacements/condition_variable \
  #replacements/list \
  #replacements/map \
  #replacements/unordered_map \
  #replacements/unordered_set \
  #replacements/memory \
  #replacements/mutex \
  #replacements/thread \
  #replacements/sched.h \
  #replacements/pthread.h \
  #replacements/unistd.h \
  #replacements/cstring \
  #replacements/cstdlib \
  #replacements/stdlib.h \
  #replacements/malloc.h \
  #replacements/mm_malloc.h \
  #replacements/string.h \
  #replacements/queue \
  #replacements/sstream \
  #replacements/stack \
  #replacements/set \
  #replacements/string \
  #replacements/vector \
  #replacements/stdio.h \
  #replacements/signal.h \
  #replacements/ssthg_pthread_clear.h \
  #replacements/wait.h \
  #replacements/sys/time.h \
  #replacements/sys/signal.h 
endif
# libTooling needs C++17; keep -std=c++17 after user CXXFLAGS.
ssthg_clang_CXXFLAGS = -fno-rtti $(AM_CXXFLAGS) -std=c++17
ssthg_clang_CPPFLAGS = $(CLANG_CPPFLAGS) $(AM_CPPFLAGS) $(SST_CPPFLAGS)

if HAVE_CXX17
ssthg_clang_SOURCES += \
	clang/annotatePragma.cc \
	clang/memoizePragma.cc \
	clang/memoizeVariableCaptureAnalyzer.cc \
	clang/ompPuppetizePragma.cc \
	clang/memoizeVariable.cc

old_standards=-std=c++11 -std=c++0x -std=c++1y -std=c++14 -std=c++1z
libclang17_a_CXXFLAGS = $(filter-out $(old_standards), $(sstmac_clang_CXXFLAGS))

library_includedir=$(includedir)/memoization
library_include_HEADERS = clang/memoization/capture.h
endif

# Install hgcc + replacement headers (with clang).
install-data-local:
	$(MKDIR_P) $(DESTDIR)$(includedir)/hgcc/libraries/pthread
	cp $(srcdir)/hgcc_include/libraries/pthread/hgcc_pthread_spinlock.h $(DESTDIR)$(includedir)/hgcc/libraries/pthread/
	$(MKDIR_P) $(DESTDIR)$(includedir)/replacements
	$(MKDIR_P) $(DESTDIR)$(includedir)/replacements/sys
	$(MKDIR_P) $(DESTDIR)$(includedir)/replacements/linux
	$(MKDIR_P) $(DESTDIR)$(includedir)/replacements/mpi
	$(MKDIR_P) $(DESTDIR)$(includedir)/replacements/libraries
	$(MKDIR_P) $(DESTDIR)$(includedir)/replacements/libraries/pthread
	cp -r $(srcdir)/replacements/* $(DESTDIR)$(includedir)/replacements/
	@if test -f $(builddir)/replacements/ssthg_pthread_clear.h; then \
		cp $(builddir)/replacements/ssthg_pthread_clear.h $(DESTDIR)$(includedir)/replacements/; \
	fi
	@if test -f $(builddir)/replacements/libraries/pthread/sstmac_sys_types.h; then \
		cp $(builddir)/replacements/libraries/pthread/sstmac_sys_types.h $(DESTDIR)$(includedir)/replacements/libraries/pthread/; \
	fi
	rm -f $(DESTDIR)$(includedir)/replacements/ssthg_pthread_clear.h.in
	rm -f $(DESTDIR)$(includedir)/replacements/libraries/pthread/sstmac_sys_types.h.in

ssthg_clang_LDFLAGS = $(CLANG_LDFLAGS) \
 -Wl,-rpath,@CLANG_INSTALL_DIR@/lib \
 @LINK_GROUP_START@ \
 -lclang-cpp \
 -lclang \
 -lclangFrontend \
 -lclangFrontendTool \
 -lclangTooling \
 -lclangBasic \
 -lclangASTMatchers \
 -lclangFormat \
 -lclangFrontend \
 -lclangDriver \
 -lclangParse \
 -lclangSerialization \
 -lclangSema \
 -lclangEdit \
 -lclangAnalysis \
 -lclangToolingCore \
 -lclangAST \
 -lclangRewrite \
 -lclangLex \
 -lclangBasic \
 -lclangSupport \
 -lclangAPINotes \
 -lclangOptions \
 -lclangAnalysisLifetimeSafety \
 @CLANG_LIBTOOLING_LIBS@ \
 @CLANG_LIBTOOLING_SYSTEM_LIBS@ @LINK_GROUP_END@

# TODO Figure out why this was included
# if CLANG_NEED_LIBCPP
# ssthg_clang_LDFLAGS += -lclang-cpp
# endif

else
# Headers-only install when clang is disabled.
install-data-local:
	$(MKDIR_P) $(DESTDIR)$(includedir)/hgcc/libraries/pthread
	cp $(srcdir)/hgcc_include/libraries/pthread/hgcc_pthread_spinlock.h $(DESTDIR)$(includedir)/hgcc/libraries/pthread/
	$(MKDIR_P) $(DESTDIR)$(includedir)/replacements
	$(MKDIR_P) $(DESTDIR)$(includedir)/replacements/sys
	$(MKDIR_P) $(DESTDIR)$(includedir)/replacements/linux
	$(MKDIR_P) $(DESTDIR)$(includedir)/replacements/mpi
	$(MKDIR_P) $(DESTDIR)$(includedir)/replacements/libraries
	$(MKDIR_P) $(DESTDIR)$(includedir)/replacements/libraries/pthread
	cp -r $(srcdir)/replacements/* $(DESTDIR)$(includedir)/replacements/
endif
