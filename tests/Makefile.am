# ext_LTLIBRARIES = libtest_tls.la
# extdir = $(SST_ELEMENT_LIBRARY_EXT_LIBDIR)

# AM_CPPFLAGS += -I$(top_srcdir)/src/sst/elements

# libtest_tls_la_SOURCES = $(srcdir)/tests/test_tls.cc
# libtest_tls_la_LDFLAGS = -module -avoid-version

# libtest_tls.la: $(srcdir)/tests/test_tls.cc
# 	hg++ -o $@ $^

# # install in proper location elem/ext...
# # install-exec-hook:
# # 	$(SST_REGISTER_TOOL) SST_ELEMENT_SOURCE     mask-mpi=$(abs_srcdir)
# # 	$(SST_REGISTER_TOOL) SST_ELEMENT_TESTS      mask-mpi=$(abs_srcdir)/tests
libtest_tls.so: test_tls.o
	hg++ $(srcdir)/tests/test_tls.o -o $(srcdir)/tests/libtest_tls.so

test_tls.o: $(srcdir)/tests/test_tls.cc
	hg++ -c $(srcdir)/tests/test_tls.cc -o $(srcdir)/tests/test_tls.o

clean:
	rm -rf $(srcdir)/tests/*.o $(srcdir)/tests/*.so
