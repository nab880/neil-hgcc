# Build tests/test_tls.cc into a loadable SST element library using hg++,
# installed where SST's element loader can dlopen it.
#
# Included from the top-level Makefile.am inside `if HAVE_CLANG`, so all
# paths are relative to the top of the build/source tree.

TESTS_EXT_LIB   = tests/libtest_tls.so
LIT_RUNNER      = $(srcdir)/tests/lit-tests/run-lit.sh
LIT_JUNIT       = $(abs_top_builddir)/tests/lit-junit.xml
TEST_TLS_RUNNER = $(srcdir)/tests/run_test_tls.sh

check-local: $(TESTS_EXT_LIB) lit-tests

.PHONY: lit-tests
lit-tests: $(top_builddir)/hgcc $(top_builddir)/hg++ $(top_builddir)/ssthg_clang
	@$(MKDIR_P) tests
	@lit_dir=`test -n "@LIT@" && dirname "@LIT@" || true`; \
	if test -n "@LIT@" || command -v lit >/dev/null 2>&1 \
	   || @pyexe@ -m lit --help >/dev/null 2>&1; then \
	    PATH=$${lit_dir:+$$lit_dir:}$$PATH \
	    HGCC=$(abs_top_builddir)/hgcc \
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

install-exec-local: $(TESTS_EXT_LIB)
	$(MKDIR_P) $(DESTDIR)$(SST_ELEMENT_LIBRARY_EXT_LIBDIR)
	$(INSTALL_DATA) -m 755 $(TESTS_EXT_LIB) \
	    $(DESTDIR)$(SST_ELEMENT_LIBRARY_EXT_LIBDIR)/

uninstall-local:
	rm -f $(DESTDIR)$(SST_ELEMENT_LIBRARY_EXT_LIBDIR)/libtest_tls.so

CLEANFILES   = tests/test_tls.o $(TESTS_EXT_LIB) $(LIT_JUNIT) tests/out.log
EXTRA_DIST  += tests/test_tls.cc tests/test_tls.py tests/run_test_tls.sh tests/lit-tests
