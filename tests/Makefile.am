# Build tests/test_tls.cc into a loadable SST element library using hg++,
# installed where SST's element loader can dlopen it.
#
# Included from the top-level Makefile.am inside `if HAVE_CLANG`, so all
# paths are relative to the top of the build/source tree.

TESTS_EXT_LIB   = tests/libtest_tls.so
LIT_RUNNER      = $(srcdir)/tests/lit-tests/run-lit.sh
LIT_JUNIT       = $(abs_top_builddir)/tests/lit-junit.xml
TEST_TLS_RUNNER = $(srcdir)/tests/run_test_tls.sh

# The CUDA integration tests are gated on the configure CUDA probe
# (HAVE_CUDA_SUPPORT). When CUDA is unavailable these stay empty, so the shared
# check/install/installcheck recipes below transparently skip them.
if HAVE_CUDA_SUPPORT
CUDA_TEST_EXT_LIBS = tests/libtest_cuda_vecadd.so \
                     tests/libtest_cuda_streams.so \
                     tests/libtest_cuda_mpi_halo.so
CUDA_TEST_OBJS     = tests/test_cuda_vecadd.o \
                     tests/test_cuda_streams.o \
                     tests/test_cuda_mpi_halo.o
CUDA_TEST_LOGS     = tests/out_cuda.log tests/out_bw_hi.log tests/out_bw_lo.log \
                     tests/out_cal.log tests/out_streams.log \
                     tests/out_halo_off.log tests/out_halo_on.log
CUDA_TEST_RUNNERS  = $(srcdir)/tests/run_test_cuda_vecadd.sh \
                     $(srcdir)/tests/run_test_cuda_streams.sh \
                     $(srcdir)/tests/run_test_cuda_mpi_halo.sh \
                     $(srcdir)/tests/run_test_cuda_arch.sh
endif

check-local: $(TESTS_EXT_LIB) $(CUDA_TEST_EXT_LIBS) lit-tests

.PHONY: lit-tests
lit-tests: $(top_builddir)/hgcc $(top_builddir)/hg++ $(top_builddir)/ssthg_clang
	@$(MKDIR_P) tests
	@lit_dir=`test -n "@LIT@" && dirname "@LIT@" || true`; \
	if test -n "@LIT@" || command -v lit >/dev/null 2>&1 \
	   || @pyexe@ -m lit --help >/dev/null 2>&1; then \
	    PATH=$${lit_dir:+$$lit_dir:}$$PATH \
	    HGCC=$(abs_top_builddir)/hgcc \
	    HGXX=$(abs_top_builddir)/hg++ \
	    SSTHG_CLANG=$(abs_top_builddir)/ssthg_clang \
	    FILECHECK=$${FILECHECK:-@FILECHECK@} \
	    PYTHONPATH=$(abs_top_builddir):$(abs_top_srcdir)$${PYTHONPATH:+:$${PYTHONPATH}} \
	    LLVM_ROOT=$${LLVM_ROOT:-@CLANG_INSTALL_DIR@} \
	    bash $(LIT_RUNNER) --xunit-xml-output=$(LIT_JUNIT); \
	else \
	    echo "lit-tests: lit not installed; skipping (configure with --enable-strict-tests to enforce)"; \
	fi

installcheck-local:
	@if test -n "@SST_PROG@" || command -v sst >/dev/null 2>&1; then \
	    SST_BIN=$${SST_BIN:-@SST_PROG@} \
	    bash $(TEST_TLS_RUNNER); \
	    for r in $(CUDA_TEST_RUNNERS); do \
	        SST_BIN=$${SST_BIN:-@SST_PROG@} bash $$r; \
	    done; \
	else \
	    echo "test_tls: sst not installed; skipping (configure with --enable-strict-tests to enforce)"; \
	fi

tests/test_tls.o: $(srcdir)/tests/test_tls.cc \
                  $(top_builddir)/ssthg_clang \
                  $(top_builddir)/hg++
	@$(MKDIR_P) tests
	PYTHONPATH=$(top_builddir):$(top_srcdir)$${PYTHONPATH:+:$${PYTHONPATH}} \
	SSTHG_CLANG=$(top_builddir)/ssthg_clang \
	    $(top_builddir)/hg++ -c $(srcdir)/tests/test_tls.cc -o $@

$(TESTS_EXT_LIB): tests/test_tls.o $(top_builddir)/hg++
	PYTHONPATH=$(top_builddir):$(top_srcdir)$${PYTHONPATH:+:$${PYTHONPATH}} \
	SSTHG_CLANG=$(top_builddir)/ssthg_clang \
	    $(top_builddir)/hg++ tests/test_tls.o -o $@

# CUDA tests build like plain C++ (default ENCAPSULATE mode, no
# SST_HG_SKELETONIZE): the launch rewrite and device-body strip fire in every
# mode. Each .cu becomes its own loadable element; the sst_hg_cuda ABI resolves
# from libhg at load.
tests/test_cuda_vecadd.o: $(srcdir)/tests/test_cuda_vecadd.cu \
                          $(top_builddir)/ssthg_clang $(top_builddir)/hg++
	@$(MKDIR_P) tests
	PYTHONPATH=$(top_builddir):$(top_srcdir)$${PYTHONPATH:+:$${PYTHONPATH}} \
	SSTHG_CLANG=$(top_builddir)/ssthg_clang \
	    $(top_builddir)/hg++ -c $(srcdir)/tests/test_cuda_vecadd.cu -o $@

tests/libtest_cuda_vecadd.so: tests/test_cuda_vecadd.o $(top_builddir)/hg++
	PYTHONPATH=$(top_builddir):$(top_srcdir)$${PYTHONPATH:+:$${PYTHONPATH}} \
	SSTHG_CLANG=$(top_builddir)/ssthg_clang \
	    $(top_builddir)/hg++ tests/test_cuda_vecadd.o -o $@

tests/test_cuda_streams.o: $(srcdir)/tests/test_cuda_streams.cu \
                           $(top_builddir)/ssthg_clang $(top_builddir)/hg++
	@$(MKDIR_P) tests
	PYTHONPATH=$(top_builddir):$(top_srcdir)$${PYTHONPATH:+:$${PYTHONPATH}} \
	SSTHG_CLANG=$(top_builddir)/ssthg_clang \
	    $(top_builddir)/hg++ -c $(srcdir)/tests/test_cuda_streams.cu -o $@

tests/libtest_cuda_streams.so: tests/test_cuda_streams.o $(top_builddir)/hg++
	PYTHONPATH=$(top_builddir):$(top_srcdir)$${PYTHONPATH:+:$${PYTHONPATH}} \
	SSTHG_CLANG=$(top_builddir)/ssthg_clang \
	    $(top_builddir)/hg++ tests/test_cuda_streams.o -o $@

tests/test_cuda_mpi_halo.o: $(srcdir)/tests/test_cuda_mpi_halo.cu \
                            $(top_builddir)/ssthg_clang $(top_builddir)/hg++
	@$(MKDIR_P) tests
	PYTHONPATH=$(top_builddir):$(top_srcdir)$${PYTHONPATH:+:$${PYTHONPATH}} \
	SSTHG_CLANG=$(top_builddir)/ssthg_clang \
	    $(top_builddir)/hg++ -c $(srcdir)/tests/test_cuda_mpi_halo.cu -o $@

tests/libtest_cuda_mpi_halo.so: tests/test_cuda_mpi_halo.o $(top_builddir)/hg++
	PYTHONPATH=$(top_builddir):$(top_srcdir)$${PYTHONPATH:+:$${PYTHONPATH}} \
	SSTHG_CLANG=$(top_builddir)/ssthg_clang \
	    $(top_builddir)/hg++ tests/test_cuda_mpi_halo.o -o $@

install-exec-local: $(TESTS_EXT_LIB) $(CUDA_TEST_EXT_LIBS)
	$(MKDIR_P) $(DESTDIR)$(SST_ELEMENT_LIBRARY_EXT_LIBDIR)
	$(INSTALL_DATA) -m 755 $(TESTS_EXT_LIB) \
	    $(DESTDIR)$(SST_ELEMENT_LIBRARY_EXT_LIBDIR)/
	@for lib in $(CUDA_TEST_EXT_LIBS); do \
	    echo " $(INSTALL_DATA) -m 755 $$lib $(DESTDIR)$(SST_ELEMENT_LIBRARY_EXT_LIBDIR)/"; \
	    $(INSTALL_DATA) -m 755 $$lib $(DESTDIR)$(SST_ELEMENT_LIBRARY_EXT_LIBDIR)/; \
	done

uninstall-local:
	rm -f $(DESTDIR)$(SST_ELEMENT_LIBRARY_EXT_LIBDIR)/libtest_tls.so
	rm -f $(DESTDIR)$(SST_ELEMENT_LIBRARY_EXT_LIBDIR)/libtest_cuda_vecadd.so
	rm -f $(DESTDIR)$(SST_ELEMENT_LIBRARY_EXT_LIBDIR)/libtest_cuda_streams.so
	rm -f $(DESTDIR)$(SST_ELEMENT_LIBRARY_EXT_LIBDIR)/libtest_cuda_mpi_halo.so

CLEANFILES   = tests/test_tls.o $(TESTS_EXT_LIB) $(LIT_JUNIT) tests/out.log \
               $(CUDA_TEST_OBJS) $(CUDA_TEST_EXT_LIBS) $(CUDA_TEST_LOGS)
EXTRA_DIST  += tests/test_tls.cc tests/test_tls.py tests/run_test_tls.sh tests/lit-tests \
               tests/test_cuda_vecadd.cu tests/test_cuda_vecadd.py tests/run_test_cuda_vecadd.sh \
               tests/test_cuda_streams.cu tests/test_cuda_streams.py tests/run_test_cuda_streams.sh \
               tests/test_cuda_mpi_halo.cu tests/test_cuda_mpi_halo.py tests/run_test_cuda_mpi_halo.sh \
               tests/run_test_cuda_arch.sh \
               tests/cuda_calibration.json
