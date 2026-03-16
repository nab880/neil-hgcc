check_PROGRAMS = test_tls
test_tls_SOURCES = $(srcdir)/tests/test_tls.cc

TESTS = $(check_PROGRAMS)

test_tls: $(test_tls_SOURCES) hg++
	hg++ -c $(test_tls_SOURCES)
	hg++ test_tls.o -o test_tls.so
