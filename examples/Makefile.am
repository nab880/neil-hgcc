# Runnable and compile-only examples for sst-hgcc.
# Included from the top-level Makefile.am inside `if HAVE_CLANG`.

HGPP_ENV = PYTHONPATH=$(top_builddir):$(top_srcdir)$${PYTHONPATH:+:$${PYTHONPATH}} \
           SSTHG_CLANG=$(top_builddir)/ssthg_clang

EXAMPLES_LIBS = \
	examples/libmercury_hello.so \
	examples/libmercury_hello_pragma.so \
	examples/libcompute_demo.so \
	examples/libblocking_demo.so

EXAMPLES_OBJS = \
	examples/mercury_hello/hello.o \
	examples/mercury_hello_pragma/hello_pragma.o \
	examples/pragmas/compute/demo_full/compute_app.o \
	examples/pragmas/blocking/demo_full/blocking_app.o

MEMOIZE_DEMO_OBJ = examples/pragmas/memoize/demo.o

.PHONY: examples install-examples uninstall-examples run-examples
examples: $(EXAMPLES_LIBS) $(MEMOIZE_DEMO_OBJ)

install-examples: $(EXAMPLES_LIBS)
	$(MKDIR_P) $(DESTDIR)$(SST_ELEMENT_LIBRARY_EXT_LIBDIR)
	$(INSTALL_DATA) -m 755 $(EXAMPLES_LIBS) \
	    $(DESTDIR)$(SST_ELEMENT_LIBRARY_EXT_LIBDIR)/

uninstall-examples:
	rm -f $(DESTDIR)$(SST_ELEMENT_LIBRARY_EXT_LIBDIR)/libmercury_hello.so
	rm -f $(DESTDIR)$(SST_ELEMENT_LIBRARY_EXT_LIBDIR)/libmercury_hello_pragma.so
	rm -f $(DESTDIR)$(SST_ELEMENT_LIBRARY_EXT_LIBDIR)/libcompute_demo.so
	rm -f $(DESTDIR)$(SST_ELEMENT_LIBRARY_EXT_LIBDIR)/libblocking_demo.so

run-examples: examples
	@if test -n "@SST_PROG@" || command -v sst >/dev/null 2>&1; then \
	    SST_BIN=$${SST_BIN:-@SST_PROG@} \
	    bash $(srcdir)/examples/mercury_hello/run.sh && \
	    bash $(srcdir)/examples/mercury_hello_pragma/run.sh && \
	    bash $(srcdir)/examples/pragmas/compute/demo_full/run.sh && \
	    bash $(srcdir)/examples/pragmas/blocking/demo_full/run.sh; \
	else \
	    echo "run-examples: sst not installed; skipping"; \
	fi

examples/mercury_hello/hello.o: $(srcdir)/examples/mercury_hello/hello.cc \
                                $(top_builddir)/ssthg_clang $(top_builddir)/hg++
	@$(MKDIR_P) examples/mercury_hello
	$(HGPP_ENV) $(top_builddir)/hg++ -c $(srcdir)/examples/mercury_hello/hello.cc -o $@

examples/mercury_hello_pragma/hello_pragma.o: $(srcdir)/examples/mercury_hello_pragma/hello_pragma.cc \
                                              $(top_builddir)/ssthg_clang $(top_builddir)/hg++
	@$(MKDIR_P) examples/mercury_hello_pragma
	$(HGPP_ENV) $(top_builddir)/hg++ -c $(srcdir)/examples/mercury_hello_pragma/hello_pragma.cc -o $@

examples/pragmas/compute/demo_full/compute_app.o: $(srcdir)/examples/pragmas/compute/demo_full/compute_app.cc \
                                                  $(top_builddir)/ssthg_clang $(top_builddir)/hg++
	@$(MKDIR_P) examples/pragmas/compute/demo_full
	$(HGPP_ENV) $(top_builddir)/hg++ -c $(srcdir)/examples/pragmas/compute/demo_full/compute_app.cc -o $@

examples/pragmas/blocking/demo_full/blocking_app.o: $(srcdir)/examples/pragmas/blocking/demo_full/blocking_app.cc \
                                                    $(top_builddir)/ssthg_clang $(top_builddir)/hg++
	@$(MKDIR_P) examples/pragmas/blocking/demo_full
	$(HGPP_ENV) $(top_builddir)/hg++ -c $(srcdir)/examples/pragmas/blocking/demo_full/blocking_app.cc -o $@

examples/libmercury_hello.so: examples/mercury_hello/hello.o $(top_builddir)/hg++
	$(HGPP_ENV) $(top_builddir)/hg++ examples/mercury_hello/hello.o -o $@

examples/libmercury_hello_pragma.so: examples/mercury_hello_pragma/hello_pragma.o $(top_builddir)/hg++
	$(HGPP_ENV) $(top_builddir)/hg++ examples/mercury_hello_pragma/hello_pragma.o -o $@

examples/libcompute_demo.so: examples/pragmas/compute/demo_full/compute_app.o $(top_builddir)/hg++
	$(HGPP_ENV) $(top_builddir)/hg++ examples/pragmas/compute/demo_full/compute_app.o -o $@

examples/libblocking_demo.so: examples/pragmas/blocking/demo_full/blocking_app.o $(top_builddir)/hg++
	$(HGPP_ENV) $(top_builddir)/hg++ examples/pragmas/blocking/demo_full/blocking_app.o -o $@

examples/pragmas/memoize/demo.o: $(srcdir)/examples/pragmas/memoize/demo.cc \
                                 $(top_builddir)/ssthg_clang $(top_builddir)/hg++
	@$(MKDIR_P) examples/pragmas/memoize
	$(HGPP_ENV) $(top_builddir)/hg++ --memoize -c $(srcdir)/examples/pragmas/memoize/demo.cc -o $@

CLEANFILES += $(EXAMPLES_OBJS) $(EXAMPLES_LIBS) $(MEMOIZE_DEMO_OBJ) \
	examples/mercury_hello/out.log \
	examples/mercury_hello_pragma/out.log \
	examples/pragmas/compute/demo_full/out.log \
	examples/pragmas/blocking/demo_full/out.log

EXTRA_DIST += examples/platform_file_hg_test.py \
	examples/scale_topo.py \
	examples/make_scaling_chart.py \
	examples/fig_A_commswall.py \
	examples/fig_B_3dsplit.py \
	examples/fig_C_allreduce_fit.py \
	examples/fig_D_overlap.py \
	examples/fig_pipeline.py \
	examples/mercury_hello/hello.cc examples/mercury_hello/hello.py examples/mercury_hello/run.sh \
	examples/mercury_hello_pragma/hello_pragma.cc examples/mercury_hello_pragma/hello_pragma.py \
	examples/mercury_hello_pragma/run.sh \
	examples/pragmas/compute/demo_full/compute_app.cc \
	examples/pragmas/compute/demo_full/compute_app.py \
	examples/pragmas/compute/demo_full/run.sh \
	examples/pragmas/blocking/demo_full/blocking_app.cc \
	examples/pragmas/blocking/demo_full/blocking_app.py \
	examples/pragmas/blocking/demo_full/run.sh \
	examples/pragmas/memoize/demo.cc \
	examples/mercury_llm_train/train.cu examples/mercury_llm_train/train.py \
	examples/mercury_llm_train/run.sh examples/mercury_llm_train/README.md \
	examples/mercury_megatron/megatron.cu examples/mercury_megatron/megatron.py \
	examples/mercury_megatron/run.sh examples/mercury_megatron/README.md \
	examples/mercury_3d/parallel3d.cu examples/mercury_3d/parallel3d.py \
	examples/mercury_3d/run.sh examples/mercury_3d/factorization_sweep.sh \
	examples/mercury_3d/README.md \
	examples/mercury_fsdp/fsdp.cu examples/mercury_fsdp/fsdp.py \
	examples/mercury_fsdp/run.sh examples/mercury_fsdp/README.md \
	examples/mercury_inference/infer.cu examples/mercury_inference/infer.py \
	examples/mercury_inference/run.sh examples/mercury_inference/README.md \
	examples/mercury_context/ring_attn.cu examples/mercury_context/ring_attn.py \
	examples/mercury_context/run.sh examples/mercury_context/README.md \
	examples/mercury_moe/moe.cu examples/mercury_moe/moe.py \
	examples/mercury_moe/skew_sweep.sh \
	examples/mercury_moe/README.md \
	examples/mercury_cuda_halo/halo.cu examples/mercury_cuda_halo/halo.py \
	examples/mercury_cuda_halo/run.sh examples/mercury_cuda_halo/gpu_calibration.json \
	examples/mercury_cuda_halo/README.md \
	examples/validation/analytic_baseline.py examples/validation/validate.sh \
	examples/validation/README.md
