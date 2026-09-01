/* Reimplementation of the demonstrator kernel, checked against golden_routes.csv.
 *
 * The environment and the scripted demonstrator are the only things in this
 * repo that turn geometry into numbers, and every result downstream inherits
 * whatever they do. They exist once, in torch. This is a second implementation
 * in C with nothing shared: its own constants, its own arithmetic, doubles
 * instead of float32.
 *
 * It reads verify/golden_routes.csv, which verify/export_golden.py wrote by
 * calling the repo's own code, and for every row recomputes the action the
 * demonstrator takes from that state and the state the environment moves it
 * to. A disagreement anywhere is a disagreement between the two kernels.
 *
 * Columns are resolved by name from the header, so reordering or inserting a
 * column cannot silently shift what is being compared.
 *
 *   cc -O2 -o route verify/route.c -lm && ./route verify/golden_routes.csv
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* stated independently of vla/envs.py rather than generated from it */
#define OBSTACLE_Y 0.0
#define OBSTACLE_HALF_W 0.30
#define OBSTACLE_HALF_H 0.10
#define MAX_SPEED 0.16
#define DETOUR_GAP 0.22
#define DETOUR_LIFT 0.18
#define BELOW_MARGIN 0.05

#define TOL 5e-6
#define MAXCOL 32

static int col_index(char *header, const char *want) {
    char buf[4096];
    snprintf(buf, sizeof buf, "%s", header);
    int i = 0;
    for (char *t = strtok(buf, ",\r\n"); t; t = strtok(NULL, ",\r\n"), i++)
        if (strcmp(t, want) == 0) return i;
    return -1;
}

static double clamp(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: route <golden_routes.csv>\n"); return 2; }
    FILE *fh = fopen(argv[1], "r");
    if (!fh) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }

    char line[4096];
    if (!fgets(line, sizeof line, fh)) { fprintf(stderr, "empty file\n"); return 2; }
    const char *names[] = {"target_x", "side", "agent_x", "agent_y",
                           "action_x", "action_y", "blocked", "next_x", "next_y"};
    int idx[9];
    for (int i = 0; i < 9; i++) {
        idx[i] = col_index(line, names[i]);
        if (idx[i] < 0) { fprintf(stderr, "missing column %s\n", names[i]); return 2; }
    }

    long rows = 0, bad = 0;
    double worst = 0.0;
    const char *worst_col = "none";
    while (fgets(line, sizeof line, fh)) {
        if (line[0] == '\n' || line[0] == '\r') continue;
        double f[MAXCOL];
        int n = 0;
        for (char *t = strtok(line, ",\r\n"); t && n < MAXCOL; t = strtok(NULL, ",\r\n"))
            f[n++] = atof(t);
        int need = 0;
        for (int i = 0; i < 9; i++) if (idx[i] > need) need = idx[i];
        if (n <= need) {
            fprintf(stderr, "row %ld has %d fields, needs %d\n", rows + 1, n, need + 1);
            return 1;
        }
        double target_x = f[idx[0]], side = f[idx[1]];
        double ax = f[idx[2]], ay = f[idx[3]];

        /* waypoint: around the obstacle on `side` while below it, then the target */
        int below = ay < OBSTACLE_Y + OBSTACLE_HALF_H + BELOW_MARGIN;
        double detour = side > 0 ? OBSTACLE_HALF_W + DETOUR_GAP
                                 : -(OBSTACLE_HALF_W + DETOUR_GAP);
        double wx = below ? detour : target_x;
        double wy = below ? OBSTACLE_Y + OBSTACLE_HALF_H + DETOUR_LIFT : 0.72;
        double dx = wx - ax, dy = wy - ay;
        double norm = sqrt(dx * dx + dy * dy);
        if (norm < 1e-6) norm = 1e-6;
        double a_x = clamp(dx / norm, -1.0, 1.0), a_y = clamp(dy / norm, -1.0, 1.0);

        double nx = clamp(ax + a_x * MAX_SPEED, -1.0, 1.0);
        double ny = clamp(ay + a_y * MAX_SPEED, -1.0, 1.0);
        int blocked = fabs(nx) < OBSTACLE_HALF_W && fabs(ny - OBSTACLE_Y) < OBSTACLE_HALF_H;
        if (blocked) { nx = ax; ny = ay; }

        struct { const char *name; double got, want; } cmp[5] = {
            {"action_x", a_x, f[idx[4]]}, {"action_y", a_y, f[idx[5]]},
            {"blocked", (double)blocked, f[idx[6]]},
            {"next_x", nx, f[idx[7]]}, {"next_y", ny, f[idx[8]]},
        };
        for (int i = 0; i < 5; i++) {
            double d = fabs(cmp[i].got - cmp[i].want);
            if (d > worst) { worst = d; worst_col = cmp[i].name; }
            if (d > TOL) {
                if (bad < 5)
                    fprintf(stderr, "row %ld %s: C %.9f vs golden %.9f (diff %.3e)\n",
                            rows + 1, cmp[i].name, cmp[i].got, cmp[i].want, d);
                bad++;
            }
        }
        rows++;
    }
    fclose(fh);
    if (rows == 0) { fprintf(stderr, "no data rows\n"); return 1; }
    printf("route.c: %ld rows, %ld disagreements, worst %.3e on %s (tol %.0e)\n",
           rows, bad, worst, worst_col, TOL);
    return bad ? 1 : 0;
}
