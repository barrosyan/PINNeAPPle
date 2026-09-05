/* Write a small, genuinely spec-compliant CGNS/HDF5 file using the real,
 * official CGNS Mid-Level Library (MLL) API -- the same reference
 * implementation used by SU2, Pointwise-adjacent tools, and the CGNS
 * project's own cgns_utils. This is NOT code from PINNeAPPle's own reader --
 * it links against libcgns (built by Homebrew's `cgns` formula) and calls
 * its public C API (cg_open/cg_base_write/cg_zone_write/cg_coord_write/
 * cg_section_write/cg_sol_write/cg_field_write/cg_close) to do the actual
 * SIDS-to-HDF5 encoding, exactly as any real CGNS-writing CFD tool would.
 *
 * Mesh: the same 8-node / 6-tet unit cube used for the other three format
 * fixtures, with a linear "Temperature" vertex field and a "Pressure"
 * (distance-from-origin) vertex field, so results can be checked exactly.
 */
#include "cgnslib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <out.cgns>\n", argv[0]);
        return 1;
    }
    const char *out_path = argv[1];
    int index_file, index_base, index_zone, index_coord, index_section, index_flow, index_field;
    int icelldim = 3, iphysdim = 3;
    cgsize_t isize[3];
    char basename[33] = "Base";
    char zonename[33] = "Zone1";

    const int NNODES = 8;
    const int NELEM = 6;

    double x[8] = {0,1,1,0,0,1,1,0};
    double y[8] = {0,0,1,1,0,0,1,1};
    double z[8] = {0,0,0,0,1,1,1,1};

    /* 1-based node ids, same 6-tet decomposition of the unit cube used
     * elsewhere in this validation pass. */
    cgsize_t elements[6*4] = {
        1,2,4,5,
        2,3,4,7,
        2,4,5,7,
        4,5,7,8,
        2,5,6,7,
        1,4,5,2,
    };

    double temperature[8], pressure[8];
    for (int i = 0; i < NNODES; i++) {
        temperature[i] = x[i] + 2.0*y[i] + 3.0*z[i];
        pressure[i] = sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]);
    }

    if (cg_open(out_path, CG_MODE_WRITE, &index_file)) cg_error_exit();

    if (cg_base_write(index_file, basename, icelldim, iphysdim, &index_base)) cg_error_exit();

    isize[0] = NNODES; /* vertex size */
    isize[1] = NELEM;  /* cell size */
    isize[2] = 0;      /* boundary vertex size, 0 = unsorted */

    if (cg_zone_write(index_file, index_base, zonename, isize, CGNS_ENUMV(Unstructured), &index_zone)) cg_error_exit();

    if (cg_coord_write(index_file, index_base, index_zone, CGNS_ENUMV(RealDouble), "CoordinateX", x, &index_coord)) cg_error_exit();
    if (cg_coord_write(index_file, index_base, index_zone, CGNS_ENUMV(RealDouble), "CoordinateY", y, &index_coord)) cg_error_exit();
    if (cg_coord_write(index_file, index_base, index_zone, CGNS_ENUMV(RealDouble), "CoordinateZ", z, &index_coord)) cg_error_exit();

    if (cg_section_write(index_file, index_base, index_zone, "Elements", CGNS_ENUMV(TETRA_4), 1, NELEM, 0, elements, &index_section)) cg_error_exit();

    if (cg_sol_write(index_file, index_base, index_zone, "FlowSolution", CGNS_ENUMV(Vertex), &index_flow)) cg_error_exit();
    if (cg_field_write(index_file, index_base, index_zone, index_flow, CGNS_ENUMV(RealDouble), "Temperature", temperature, &index_field)) cg_error_exit();
    if (cg_field_write(index_file, index_base, index_zone, index_flow, CGNS_ENUMV(RealDouble), "Pressure", pressure, &index_field)) cg_error_exit();

    if (cg_close(index_file)) cg_error_exit();

    printf("wrote %s via real CGNS MLL API\n", out_path);
    return 0;
}
