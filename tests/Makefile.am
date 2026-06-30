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
                     tests/libtest_cuda_mpi_halo.so \
                     tests/libmercury_llm_train.so \
                     tests/libmercury_megatron.so \
                     tests/libmercury_3d.so \
                     tests/libmercury_fsdp.so \
                     tests/libmercury_moe.so \
                     tests/libmercury_inference.so \
                     tests/libmercury_context.so
CUDA_TEST_OBJS     = tests/test_cuda_vecadd.o \
                     tests/test_cuda_streams.o \
                     tests/test_cuda_mpi_halo.o \
                     tests/llm_train.o \
                     tests/megatron.o \
                     tests/parallel3d.o \
                     tests/fsdp.o \
                     tests/moe.o \
                     tests/infer.o \
                     tests/ring_attn.o
CUDA_TEST_LOGS     = tests/out_cuda.log tests/out_bw_hi.log tests/out_bw_lo.log \
                     tests/out_cal.log tests/out_streams.log \
                     tests/out_halo_off.log tests/out_halo_on.log \
                     tests/out_llm_slow_ov.log tests/out_llm_slow_noov.log \
                     tests/out_llm_fast_gd.log tests/out_llm_fast_nogd.log \
                     tests/out_mega_pp_m1.log tests/out_mega_pp_m4.log \
                     tests/out_mega_tp_slow.log tests/out_mega_tp_fast.log \
                     tests/out_3d_full.log tests/out_3d_pp.log tests/out_3d_tp.log \
                     tests/out_3d_dp_slow.log tests/out_3d_dp_fast.log \
                     tests/out_fsdp_prefetch.log tests/out_fsdp_noprefetch.log \
                     tests/out_moe_uniform.log tests/out_moe_skewed.log \
                     tests/out_infer_s16.log tests/out_infer_s32.log \
                     tests/out_ctx_overlap.log tests/out_ctx_nooverlap.log
CUDA_TEST_RUNNERS  = $(srcdir)/tests/run_test_cuda_vecadd.sh \
                     $(srcdir)/tests/run_test_cuda_streams.sh \
                     $(srcdir)/tests/run_test_cuda_mpi_halo.sh \
                     $(srcdir)/tests/run_test_cuda_llm_train.sh \
                     $(srcdir)/tests/run_test_cuda_megatron.sh \
                     $(srcdir)/tests/run_test_cuda_3d.sh \
                     $(srcdir)/tests/run_test_cuda_fsdp.sh \
                     $(srcdir)/tests/run_test_cuda_moe.sh \
                     $(srcdir)/tests/run_test_cuda_inference.sh \
                     $(srcdir)/tests/run_test_cuda_context.sh \
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

# The data-parallel LLM training smoke test builds the actual demo skeleton from
# examples/ (the run script drives examples/mercury_llm_train/train.py). The lib
# name must match the app name (ssthg_app_name mercury_llm_train).
tests/llm_train.o: $(srcdir)/examples/mercury_llm_train/train.cu \
                   $(top_builddir)/ssthg_clang $(top_builddir)/hg++
	@$(MKDIR_P) tests
	PYTHONPATH=$(top_builddir):$(top_srcdir)$${PYTHONPATH:+:$${PYTHONPATH}} \
	SSTHG_CLANG=$(top_builddir)/ssthg_clang \
	    $(top_builddir)/hg++ -c $(srcdir)/examples/mercury_llm_train/train.cu -o $@

tests/libmercury_llm_train.so: tests/llm_train.o $(top_builddir)/hg++
	PYTHONPATH=$(top_builddir):$(top_srcdir)$${PYTHONPATH:+:$${PYTHONPATH}} \
	SSTHG_CLANG=$(top_builddir)/ssthg_clang \
	    $(top_builddir)/hg++ tests/llm_train.o -o $@

# The tensor+pipeline parallel smoke test, likewise built from the example source
# (the run script drives examples/mercury_megatron/megatron.py).
tests/megatron.o: $(srcdir)/examples/mercury_megatron/megatron.cu \
                  $(top_builddir)/ssthg_clang $(top_builddir)/hg++
	@$(MKDIR_P) tests
	PYTHONPATH=$(top_builddir):$(top_srcdir)$${PYTHONPATH:+:$${PYTHONPATH}} \
	SSTHG_CLANG=$(top_builddir)/ssthg_clang \
	    $(top_builddir)/hg++ -c $(srcdir)/examples/mercury_megatron/megatron.cu -o $@

tests/libmercury_megatron.so: tests/megatron.o $(top_builddir)/hg++
	PYTHONPATH=$(top_builddir):$(top_srcdir)$${PYTHONPATH:+:$${PYTHONPATH}} \
	SSTHG_CLANG=$(top_builddir)/ssthg_clang \
	    $(top_builddir)/hg++ tests/megatron.o -o $@

# The 3D-parallel (DP x PP x TP) smoke test, likewise built from the example
# source (the run script drives examples/mercury_3d/parallel3d.py).
tests/parallel3d.o: $(srcdir)/examples/mercury_3d/parallel3d.cu \
                    $(top_builddir)/ssthg_clang $(top_builddir)/hg++
	@$(MKDIR_P) tests
	PYTHONPATH=$(top_builddir):$(top_srcdir)$${PYTHONPATH:+:$${PYTHONPATH}} \
	SSTHG_CLANG=$(top_builddir)/ssthg_clang \
	    $(top_builddir)/hg++ -c $(srcdir)/examples/mercury_3d/parallel3d.cu -o $@

tests/libmercury_3d.so: tests/parallel3d.o $(top_builddir)/hg++
	PYTHONPATH=$(top_builddir):$(top_srcdir)$${PYTHONPATH:+:$${PYTHONPATH}} \
	SSTHG_CLANG=$(top_builddir)/ssthg_clang \
	    $(top_builddir)/hg++ tests/parallel3d.o -o $@

# FSDP / ZeRO-3 smoke test.
tests/fsdp.o: $(srcdir)/examples/mercury_fsdp/fsdp.cu \
              $(top_builddir)/ssthg_clang $(top_builddir)/hg++
	@$(MKDIR_P) tests
	PYTHONPATH=$(top_builddir):$(top_srcdir)$${PYTHONPATH:+:$${PYTHONPATH}} \
	SSTHG_CLANG=$(top_builddir)/ssthg_clang \
	    $(top_builddir)/hg++ -c $(srcdir)/examples/mercury_fsdp/fsdp.cu -o $@

tests/libmercury_fsdp.so: tests/fsdp.o $(top_builddir)/hg++
	PYTHONPATH=$(top_builddir):$(top_srcdir)$${PYTHONPATH:+:$${PYTHONPATH}} \
	SSTHG_CLANG=$(top_builddir)/ssthg_clang \
	    $(top_builddir)/hg++ tests/fsdp.o -o $@

# Expert-parallel MoE smoke test.
tests/moe.o: $(srcdir)/examples/mercury_moe/moe.cu \
             $(top_builddir)/ssthg_clang $(top_builddir)/hg++
	@$(MKDIR_P) tests
	PYTHONPATH=$(top_builddir):$(top_srcdir)$${PYTHONPATH:+:$${PYTHONPATH}} \
	SSTHG_CLANG=$(top_builddir)/ssthg_clang \
	    $(top_builddir)/hg++ -c $(srcdir)/examples/mercury_moe/moe.cu -o $@

tests/libmercury_moe.so: tests/moe.o $(top_builddir)/hg++
	PYTHONPATH=$(top_builddir):$(top_srcdir)$${PYTHONPATH:+:$${PYTHONPATH}} \
	SSTHG_CLANG=$(top_builddir)/ssthg_clang \
	    $(top_builddir)/hg++ tests/moe.o -o $@

# Disaggregated inference smoke test.
tests/infer.o: $(srcdir)/examples/mercury_inference/infer.cu \
               $(top_builddir)/ssthg_clang $(top_builddir)/hg++
	@$(MKDIR_P) tests
	PYTHONPATH=$(top_builddir):$(top_srcdir)$${PYTHONPATH:+:$${PYTHONPATH}} \
	SSTHG_CLANG=$(top_builddir)/ssthg_clang \
	    $(top_builddir)/hg++ -c $(srcdir)/examples/mercury_inference/infer.cu -o $@

tests/libmercury_inference.so: tests/infer.o $(top_builddir)/hg++
	PYTHONPATH=$(top_builddir):$(top_srcdir)$${PYTHONPATH:+:$${PYTHONPATH}} \
	SSTHG_CLANG=$(top_builddir)/ssthg_clang \
	    $(top_builddir)/hg++ tests/infer.o -o $@

# Context parallelism / ring attention smoke test.
tests/ring_attn.o: $(srcdir)/examples/mercury_context/ring_attn.cu \
                   $(top_builddir)/ssthg_clang $(top_builddir)/hg++
	@$(MKDIR_P) tests
	PYTHONPATH=$(top_builddir):$(top_srcdir)$${PYTHONPATH:+:$${PYTHONPATH}} \
	SSTHG_CLANG=$(top_builddir)/ssthg_clang \
	    $(top_builddir)/hg++ -c $(srcdir)/examples/mercury_context/ring_attn.cu -o $@

tests/libmercury_context.so: tests/ring_attn.o $(top_builddir)/hg++
	PYTHONPATH=$(top_builddir):$(top_srcdir)$${PYTHONPATH:+:$${PYTHONPATH}} \
	SSTHG_CLANG=$(top_builddir)/ssthg_clang \
	    $(top_builddir)/hg++ tests/ring_attn.o -o $@

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
	rm -f $(DESTDIR)$(SST_ELEMENT_LIBRARY_EXT_LIBDIR)/libmercury_llm_train.so
	rm -f $(DESTDIR)$(SST_ELEMENT_LIBRARY_EXT_LIBDIR)/libmercury_megatron.so
	rm -f $(DESTDIR)$(SST_ELEMENT_LIBRARY_EXT_LIBDIR)/libmercury_3d.so
	rm -f $(DESTDIR)$(SST_ELEMENT_LIBRARY_EXT_LIBDIR)/libmercury_fsdp.so
	rm -f $(DESTDIR)$(SST_ELEMENT_LIBRARY_EXT_LIBDIR)/libmercury_moe.so
	rm -f $(DESTDIR)$(SST_ELEMENT_LIBRARY_EXT_LIBDIR)/libmercury_inference.so
	rm -f $(DESTDIR)$(SST_ELEMENT_LIBRARY_EXT_LIBDIR)/libmercury_context.so

CLEANFILES   = tests/test_tls.o $(TESTS_EXT_LIB) $(LIT_JUNIT) tests/out.log \
               $(CUDA_TEST_OBJS) $(CUDA_TEST_EXT_LIBS) $(CUDA_TEST_LOGS)
EXTRA_DIST  += tests/test_tls.cc tests/test_tls.py tests/run_test_tls.sh tests/lit-tests \
               tests/test_cuda_vecadd.cu tests/test_cuda_vecadd.py tests/run_test_cuda_vecadd.sh \
               tests/test_cuda_streams.cu tests/test_cuda_streams.py tests/run_test_cuda_streams.sh \
               tests/test_cuda_mpi_halo.cu tests/test_cuda_mpi_halo.py tests/run_test_cuda_mpi_halo.sh \
               tests/run_test_cuda_llm_train.sh tests/run_test_cuda_megatron.sh \
               tests/run_test_cuda_3d.sh \
               tests/run_test_cuda_fsdp.sh \
               tests/run_test_cuda_moe.sh \
               tests/run_test_cuda_inference.sh \
               tests/run_test_cuda_context.sh \
               tests/run_test_cuda_arch.sh \
               tests/cuda_calibration.json
