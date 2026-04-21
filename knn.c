/*
 * knn.c - K-Nearest Neighbors classifier implemented in C.
 *
 * This is a mirror of knn.py in the same repository. Both programs implement
 * the identical algorithm, the identical CLI, and the identical deterministic
 * shuffle (xorshift32), so running them with the same seed on the same CSV
 * produces bit-for-bit identical output.
 *
 * Build:   make                                   (or:  gcc -O2 -Wall -Wextra -std=c11 -o knn knn.c -lm)
 * Run:     ./knn trees.csv 3 0.2 42
 *            arg1 = CSV file      (required)
 *            arg2 = K             (default 3)
 *            arg3 = test ratio    (default 0.2)
 *            arg4 = random seed   (default: derived from time(NULL))
 *
 * CSV format: comma-separated, optional header row, numeric features followed
 * by a string label in the last column. Example (trees.csv):
 *
 *     height,width,label
 *     24.049352,3.134282,Cedar
 *     26.617310,0.423567,Pine
 *     ...
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

#define MAX_LINE      1024
#define MAX_FEATURES    64
#define MAX_SAMPLES   5000
#define MAX_CLASSES    128
#define LABEL_LEN       64

typedef struct {
    double features[MAX_FEATURES];
    char   label[LABEL_LEN];
} Sample;

typedef struct {
    double distance;
    int    index;     /* index into the training array */
} Neighbor;

/* ---------- xorshift32 PRNG --------------------------------------------- */
/*  Portable replacement for rand()/srand(). The algorithm is fixed so that  */
/*  knn.py can implement the exact same sequence and both programs produce  */
/*  identical shuffles for any given seed on any platform.                   */

static uint32_t prng_state = 1;

static void prng_seed(uint32_t seed) {
    prng_state = seed == 0 ? 1u : seed;  /* xorshift requires non-zero state */
}

static uint32_t prng_next(void) {
    uint32_t x = prng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    prng_state = x;
    return x;
}

/* ---------- helpers ------------------------------------------------------ */

static void trim(char *s) {
    size_t n = strlen(s);
    while (n > 0 && (s[n-1] == '\n' || s[n-1] == '\r' ||
                     s[n-1] == ' '  || s[n-1] == '\t')) {
        s[--n] = '\0';
    }
}

static int looks_like_header(const char *line) {
    /* If the first field is not a number, treat the row as a header. */
    while (*line == ' ' || *line == '\t') line++;
    if (*line == '-' || *line == '+') line++;
    return !(isdigit((unsigned char)*line) || *line == '.');
}

/* ---------- CSV loader --------------------------------------------------- */

static int load_csv(const char *path, Sample *out, int *n_features_out) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "error: cannot open '%s'\n", path);
        return -1;
    }

    char line[MAX_LINE];
    int n = 0;
    int n_features = -1;
    int first_line = 1;

    while (fgets(line, sizeof(line), f)) {
        trim(line);
        if (line[0] == '\0') continue;

        if (first_line && looks_like_header(line)) {
            first_line = 0;
            continue;
        }
        first_line = 0;

        if (n >= MAX_SAMPLES) {
            fprintf(stderr, "warning: more than %d samples; truncating\n", MAX_SAMPLES);
            break;
        }

        /* Tokenize by ',' */
        char *tokens[MAX_FEATURES + 2];
        int  n_tokens = 0;
        char *tok = strtok(line, ",");
        while (tok && n_tokens < (int)(sizeof(tokens)/sizeof(tokens[0]))) {
            tokens[n_tokens++] = tok;
            tok = strtok(NULL, ",");
        }
        if (n_tokens < 2) continue;

        int nf = n_tokens - 1;
        if (n_features < 0) {
            n_features = nf;
        } else if (nf != n_features) {
            fprintf(stderr, "warning: row %d has %d features (expected %d); skipping\n",
                    n + 1, nf, n_features);
            continue;
        }

        for (int i = 0; i < nf; i++) {
            out[n].features[i] = strtod(tokens[i], NULL);
        }
        strncpy(out[n].label, tokens[nf], LABEL_LEN - 1);
        out[n].label[LABEL_LEN - 1] = '\0';
        trim(out[n].label);
        n++;
    }
    fclose(f);

    *n_features_out = n_features;
    return n;
}

/* ---------- shuffle (Fisher-Yates, driven by xorshift32) ---------------- */

static void shuffle(Sample *a, int n) {
    for (int i = n - 1; i > 0; i--) {
        uint32_t r = prng_next();
        int j = (int)(r % (uint32_t)(i + 1));
        Sample tmp = a[i]; a[i] = a[j]; a[j] = tmp;
    }
}

/* ---------- distance ----------------------------------------------------- */

static double euclidean(const double *a, const double *b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return sqrt(s);
}

/* ---------- K smallest via partial insertion sort ------------------------ */
/*  neighbors[] is kept sorted ascending by distance, length <= k.           */
/*  Stable on ties: a new sample with the same distance is inserted AFTER   */
/*  existing ones, so the matching Python code must preserve the same rule. */

static void insert_neighbor(Neighbor *neighbors, int *count, int k,
                            double dist, int idx) {
    if (*count < k) {
        int i = *count;
        while (i > 0 && neighbors[i-1].distance > dist) {
            neighbors[i] = neighbors[i-1];
            i--;
        }
        neighbors[i].distance = dist;
        neighbors[i].index    = idx;
        (*count)++;
    } else if (dist < neighbors[k-1].distance) {
        int i = k - 1;
        while (i > 0 && neighbors[i-1].distance > dist) {
            neighbors[i] = neighbors[i-1];
            i--;
        }
        neighbors[i].distance = dist;
        neighbors[i].index    = idx;
    }
}

/* ---------- majority vote ------------------------------------------------ */
/*  On ties the first label encountered (nearest in distance) wins.         */

static const char *majority_vote(const Neighbor *neighbors, int k,
                                 const Sample *train) {
    char  labels[MAX_CLASSES][LABEL_LEN];
    int   counts[MAX_CLASSES] = {0};
    int   n_labels = 0;

    for (int i = 0; i < k; i++) {
        const char *lbl = train[neighbors[i].index].label;
        int found = -1;
        for (int j = 0; j < n_labels; j++) {
            if (strcmp(labels[j], lbl) == 0) { found = j; break; }
        }
        if (found < 0) {
            size_t ln = strlen(lbl);
            if (ln >= LABEL_LEN) ln = LABEL_LEN - 1;
            memcpy(labels[n_labels], lbl, ln);
            labels[n_labels][ln] = '\0';
            counts[n_labels] = 1;
            n_labels++;
        } else {
            counts[found]++;
        }
    }

    int best = 0;
    for (int j = 1; j < n_labels; j++) {
        if (counts[j] > counts[best]) best = j;
    }
    /* Return a pointer into the training set so it outlives this function. */
    for (int i = 0; i < k; i++) {
        if (strcmp(train[neighbors[i].index].label, labels[best]) == 0) {
            return train[neighbors[i].index].label;
        }
    }
    return train[neighbors[0].index].label;
}

/* ---------- classify one sample ----------------------------------------- */

static const char *knn_predict(const Sample *train, int n_train,
                               const Sample *query, int n_features, int k) {
    Neighbor neighbors[MAX_FEATURES];  /* k <= MAX_FEATURES is enforced in main */
    int count = 0;

    for (int i = 0; i < n_train; i++) {
        double d = euclidean(train[i].features, query->features, n_features);
        insert_neighbor(neighbors, &count, k, d, i);
    }
    return majority_vote(neighbors, k, train);
}

/* ---------- main --------------------------------------------------------- */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: %s <csv> [k=3] [test_ratio=0.2] [seed]\n", argv[0]);
        return 1;
    }

    const char *csv_path   = argv[1];
    int         k          = (argc > 2) ? atoi(argv[2])   : 3;
    double      test_ratio = (argc > 3) ? atof(argv[3])   : 0.2;
    uint32_t    seed       = (argc > 4) ? (uint32_t)strtoul(argv[4], NULL, 10)
                                        : (uint32_t)time(NULL);

    if (k <= 0 || k > MAX_FEATURES) {
        fprintf(stderr, "error: k must be in 1..%d\n", MAX_FEATURES);
        return 1;
    }
    if (test_ratio <= 0.0 || test_ratio >= 1.0) {
        fprintf(stderr, "error: test_ratio must be in (0, 1)\n");
        return 1;
    }

    static Sample data[MAX_SAMPLES];
    int n_features = 0;
    int n = load_csv(csv_path, data, &n_features);
    if (n <= 0) {
        fprintf(stderr, "error: no samples loaded\n");
        return 1;
    }
    if (n_features > MAX_FEATURES) {
        fprintf(stderr, "error: too many features (%d > %d)\n",
                n_features, MAX_FEATURES);
        return 1;
    }

    prng_seed(seed);
    shuffle(data, n);

    int n_test  = (int)(n * test_ratio + 0.5);
    if (n_test < 1) n_test = 1;
    int n_train = n - n_test;
    if (n_train < k) {
        fprintf(stderr, "error: not enough training samples (%d) for k=%d\n",
                n_train, k);
        return 1;
    }

    Sample *train = data;
    Sample *test  = data + n_train;

    printf("dataset : %s\n", csv_path);
    printf("samples : %d  (train=%d, test=%d)\n", n, n_train, n_test);
    printf("features: %d\n", n_features);
    printf("k       : %d\n", k);
    printf("seed    : %u\n\n", seed);

    int correct = 0;
    printf("idx  true              predicted         match\n");
    printf("---  ----------------  ----------------  -----\n");
    for (int i = 0; i < n_test; i++) {
        const char *pred = knn_predict(train, n_train, &test[i], n_features, k);
        int ok = (strcmp(pred, test[i].label) == 0);
        if (ok) correct++;
        printf("%3d  %-16s  %-16s  %s\n",
               i, test[i].label, pred, ok ? "yes" : "NO");
    }

    double acc = (double)correct / (double)n_test;
    printf("\naccuracy: %d/%d = %.4f\n", correct, n_test, acc);
    return 0;
}
