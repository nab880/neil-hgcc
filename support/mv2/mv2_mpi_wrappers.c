/* Global MPI_* wrappers for macOS bundle (weak PMPI_* stays local); hgcc_* stubs for tests. */
#include <string.h>
#include <stdio.h>

#define HGCC_NO_REPLACEMENTS 1
#include <unistd.h>

int hgcc_gethostname(const char* name, size_t sz)
{
    snprintf((char*)name, sz, "ssthost");
    return 0;
}

long hgcc_gethostid(void)
{
    return 0;
}

int hgcc_usleep(unsigned usecs)
{
    (void)usecs;
    return 0;
}

#undef HGCC_NO_REPLACEMENTS
#include <mpi.h>

/* The host stubs above are always compiled. The MPI_* forwarding wrappers
 * below can instead be auto-generated from libmpi by gen_symbol_wrappers.py;
 * when hglink does that (HGCC_MV2_GEN_WRAPPERS=1) it compiles this file with
 * -DHGCC_MV2_GENERATED_MPI_WRAPPERS so the hand-written MPI_* set is skipped
 * (the generated tail-call trampolines provide them) while the host stubs,
 * which are real stubs and cannot be generated from the archive, are kept. */
#ifndef HGCC_MV2_GENERATED_MPI_WRAPPERS

/* Timers. These are weak PMPI_ aliases too; without strong wrappers they are
 * left as dynamic_lookup references in the macOS bundle (the wrapper-check
 * flags them). Every nsx benchmark calls MPI_Wtime for its timing. */
double MPI_Wtime(void)                                         { return PMPI_Wtime(); }
double MPI_Wtick(void)                                         { return PMPI_Wtick(); }

int MPI_Abort(MPI_Comm c, int e)                               { return PMPI_Abort(c, e); }
int MPI_Alloc_mem(MPI_Aint s, MPI_Info i, void *p)             { return PMPI_Alloc_mem(s, i, p); }
int MPI_Allreduce(const void *s, void *r, int c,
                  MPI_Datatype d, MPI_Op o, MPI_Comm cm)        { return PMPI_Allreduce(s, r, c, d, o, cm); }
int MPI_Barrier(MPI_Comm c)                                     { return PMPI_Barrier(c); }
int MPI_Comm_dup(MPI_Comm c, MPI_Comm *n)                      { return PMPI_Comm_dup(c, n); }
int MPI_Comm_free(MPI_Comm *c)                                  { return PMPI_Comm_free(c); }
int MPI_Comm_rank(MPI_Comm c, int *r)                           { return PMPI_Comm_rank(c, r); }
int MPI_Comm_size(MPI_Comm c, int *s)                           { return PMPI_Comm_size(c, s); }
int MPI_Finalize(void)                                          { return PMPI_Finalize(); }
int MPI_Free_mem(void *b)                                       { return PMPI_Free_mem(b); }
int MPI_Get(void *o, int oc, MPI_Datatype od, int tr,
            MPI_Aint td, int tc, MPI_Datatype tt, MPI_Win w)   { return PMPI_Get(o, oc, od, tr, td, tc, tt, w); }
int MPI_Init(int *argc, char ***argv)                           { return PMPI_Init(argc, argv); }
int MPI_Put(const void *o, int oc, MPI_Datatype od, int tr,
            MPI_Aint td, int tc, MPI_Datatype tt, MPI_Win w)   { return PMPI_Put(o, oc, od, tr, td, tc, tt, w); }
int MPI_Recv(void *b, int c, MPI_Datatype d, int s,
             int t, MPI_Comm cm, MPI_Status *st)                { return PMPI_Recv(b, c, d, s, t, cm, st); }
int MPI_Send(const void *b, int c, MPI_Datatype d,
             int dest, int t, MPI_Comm cm)                      { return PMPI_Send(b, c, d, dest, t, cm); }
int MPI_Testsome(int ic, MPI_Request *r, int *oc,
                 int *idx, MPI_Status *st)                      { return PMPI_Testsome(ic, r, oc, idx, st); }
int MPI_Win_create(void *b, MPI_Aint s, int d,
                   MPI_Info i, MPI_Comm c, MPI_Win *w)          { return PMPI_Win_create(b, s, d, i, c, w); }
int MPI_Win_free(MPI_Win *w)                                    { return PMPI_Win_free(w); }
int MPI_Win_lock(int lt, int r, int a, MPI_Win w)              { return PMPI_Win_lock(lt, r, a, w); }
int MPI_Win_unlock(int r, MPI_Win w)                            { return PMPI_Win_unlock(r, w); }

int MPI_Bcast(void *b, int c, MPI_Datatype d, int r, MPI_Comm cm)
                                                                { return PMPI_Bcast(b, c, d, r, cm); }
int MPI_Reduce(const void *s, void *r, int c, MPI_Datatype d,
               MPI_Op o, int root, MPI_Comm cm)                 { return PMPI_Reduce(s, r, c, d, o, root, cm); }
int MPI_Gather(const void *sb, int sc, MPI_Datatype st,
               void *rb, int rc, MPI_Datatype rt,
               int root, MPI_Comm cm)                           { return PMPI_Gather(sb, sc, st, rb, rc, rt, root, cm); }
int MPI_Allgather(const void *sb, int sc, MPI_Datatype st,
                  void *rb, int rc, MPI_Datatype rt,
                  MPI_Comm cm)                                  { return PMPI_Allgather(sb, sc, st, rb, rc, rt, cm); }
int MPI_Scatter(const void *sb, int sc, MPI_Datatype st,
                void *rb, int rc, MPI_Datatype rt,
                int root, MPI_Comm cm)                          { return PMPI_Scatter(sb, sc, st, rb, rc, rt, root, cm); }
int MPI_Alltoall(const void *sb, int sc, MPI_Datatype st,
                 void *rb, int rc, MPI_Datatype rt,
                 MPI_Comm cm)                                   { return PMPI_Alltoall(sb, sc, st, rb, rc, rt, cm); }
int MPI_Reduce_scatter(const void *sb, void *rb, const int *rc,
                       MPI_Datatype d, MPI_Op o, MPI_Comm cm)   { return PMPI_Reduce_scatter(sb, rb, rc, d, o, cm); }

/* Nonblocking point-to-point + completion family. These MPI_* entry points are
 * weak aliases of PMPI_* in libmpi; under the macOS bundle (-undefined
 * dynamic_lookup) an unwrapped weak MPI_* is left as a flat-namespace dynamic
 * reference that resolves to garbage at load time (observed: a call to
 * MPI_Irecv jumped to an MPI builtin-handle-shaped address and SIGSEGV'd).
 * Exporting strong MPI_* wrappers here -- exactly as for MPI_Send/MPI_Recv --
 * gives dynamic_lookup a real symbol to bind to. (halo3d uses Isend/Irecv/
 * Waitall; the rest of the family is included so future skeletons don't trip
 * the same latent bug.) */
int MPI_Isend(const void *b, int c, MPI_Datatype d, int dest,
              int t, MPI_Comm cm, MPI_Request *r)               { return PMPI_Isend(b, c, d, dest, t, cm, r); }
int MPI_Issend(const void *b, int c, MPI_Datatype d, int dest,
               int t, MPI_Comm cm, MPI_Request *r)              { return PMPI_Issend(b, c, d, dest, t, cm, r); }
int MPI_Irecv(void *b, int c, MPI_Datatype d, int s,
              int t, MPI_Comm cm, MPI_Request *r)               { return PMPI_Irecv(b, c, d, s, t, cm, r); }
int MPI_Wait(MPI_Request *r, MPI_Status *st)                    { return PMPI_Wait(r, st); }
int MPI_Test(MPI_Request *r, int *f, MPI_Status *st)            { return PMPI_Test(r, f, st); }
int MPI_Request_free(MPI_Request *r)                            { return PMPI_Request_free(r); }
int MPI_Waitany(int c, MPI_Request *r, int *i, MPI_Status *st)  { return PMPI_Waitany(c, r, i, st); }
int MPI_Testany(int c, MPI_Request *r, int *i,
                int *f, MPI_Status *st)                         { return PMPI_Testany(c, r, i, f, st); }
int MPI_Waitall(int c, MPI_Request *r, MPI_Status *st)          { return PMPI_Waitall(c, r, st); }
int MPI_Testall(int c, MPI_Request *r, int *f, MPI_Status *st)  { return PMPI_Testall(c, r, f, st); }
int MPI_Probe(int s, int t, MPI_Comm cm, MPI_Status *st)        { return PMPI_Probe(s, t, cm, st); }
int MPI_Iprobe(int s, int t, MPI_Comm cm,
               int *f, MPI_Status *st)                          { return PMPI_Iprobe(s, t, cm, f, st); }
int MPI_Get_count(const MPI_Status *st, MPI_Datatype d, int *c) { return PMPI_Get_count(st, d, c); }
int MPI_Sendrecv(const void *sb, int sc, MPI_Datatype sd, int dest, int stag,
                 void *rb, int rc, MPI_Datatype rd, int src, int rtag,
                 MPI_Comm cm, MPI_Status *st)                   { return PMPI_Sendrecv(sb, sc, sd, dest, stag, rb, rc, rd, src, rtag, cm, st); }

#endif /* HGCC_MV2_GENERATED_MPI_WRAPPERS */
