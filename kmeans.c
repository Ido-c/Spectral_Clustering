/*
 Created by ido & jacob on 13/12/2020.
*/
#define PY_SSIZE_T_CLEAN
#include <Python.h>

int findClosestCluster(double *x, double **centroids, int k, int d);

int updateCentroid(int *clusterVec, double *centroidVec, double **allVecs, int d, double *vecOfSums);

double getDistance(double *vec1, double *vec2, int d);

/*
 * arguments:
 * k - number of clusters
 * n - number of vectors
 * d - number of values in each vector
 * MAX_ITER - max number of iterations
 */
static PyObject *kmeans(PyObject *self, PyObject *args) {
    PyObject *temp_centroids, *temp_vectors, *item, *python_list;
    int k, n, d, MAX_ITER, i, j, tries, change, *p3, **clusters, temp_len;
    double *p1, *p2, **vectors, **centroids, *vecOfSums;

    if(!PyArg_ParseTuple(args, "iiiiOO", &k, &n, &d, &MAX_ITER, &temp_vectors, &temp_centroids)) {
        return NULL;
    }

    /*assert args*/
    if (k <= 0 || n <= 0 || d <= 0 || MAX_ITER <= 0 || k > n) {
        PyErr_SetString(PyExc_ValueError, "arguments should satisfy: k > 0 && n > 0 && d > 0 && MAX_ITER > 0 && k <= n");
        return NULL;
    }


    /*unpack temp_vectors into p1*/
    temp_len = PyObject_Length(temp_vectors);
    if (temp_len != (n * d)) {
        PyErr_SetString(PyExc_ValueError, "wrong number of total elements");
        return NULL;
    }
    p1 = calloc(n * d, sizeof(double)); /*allocate memory for array for all vectors*/
    if(!p1) {
        return PyErr_NoMemory();
    }
    for(i=0; i < temp_len; i++) {
        item = PyList_GetItem(temp_vectors, i);
        if (!PyFloat_Check(item)){
            free(p1);
            PyErr_SetString(PyExc_TypeError, "not all elements in list are floats");
            return NULL;
        }
        p1[i] = PyFloat_AsDouble(item);
    }
    /*initialize pointer array. vectors[i] is a pointer to i * d location in p1*/
    vectors = calloc(n, sizeof(double *));
    assert(vectors != NULL);
    for (i = 0; i < n; ++i) {
        vectors[i] = p1 + i * d;
    }

    /*unpack temp_centroids into p2*/
    temp_len = PyObject_Length(temp_centroids);
    if (temp_len != (k * d)) {
        PyErr_SetString(PyExc_ValueError, "wrong number of centroid elements");
        return NULL;
    }
    p2 = calloc(k * d, sizeof(double)); /*allocate memory for array for all vectors*/
    if(!p2) {
        return PyErr_NoMemory();
    }
    for(i=0; i < temp_len; i++) {
        item = PyList_GetItem(temp_centroids, i);
        if (!PyFloat_Check(item)){
            free(p1);
            free(vectors);
            free(p2);
            PyErr_SetString(PyExc_ValueError, "not all elements in centroids are floats");
            return NULL;
        }
        p2[i] = PyFloat_AsDouble(item);
    }
    /*initialize pointer array. centroids[i] is a pointer to i * d location in p2*/
    centroids = calloc(k, sizeof(double *));
    assert(centroids != NULL);
    for (i = 0; i < k; ++i) {
        centroids[i] = p2 + i * d;
    }

    /*same as before, for clusters vector
    for each cluster, index 0 will be used to keep size of cluster*/
    p3 = calloc(k * (n + 1), sizeof(int));
    clusters = calloc(k, sizeof(int *));
    for (i = 0; i < k; ++i) {
        clusters[i] = p3 + i * (n + 1);
    }
    assert(p3 != NULL);
    assert(clusters != NULL);

    /*allocate memory for vector to be used in updateCentroid*/
    vecOfSums = calloc(d, sizeof(double));
    assert(vecOfSums != NULL);

    /*start algorithm iterations*/
    for (tries = 0; tries < MAX_ITER; ++tries) {
        change = 0; /*a flag to keep track if clusters change*/

        /*reset the clusters*/
        for (i = 0; i < k; ++i) {
            clusters[i][0] = 0;
        }

        /* put vectors in clusters*/
        for (i = 0; i < n; ++i) {
            int index = findClosestCluster(vectors[i], (double **) centroids, k, d);
            clusters[index][++clusters[index][0]] = i; /*put vector in cluster and increment cluster size*/
        }

        /*calculate new centroids and check if they changed*/
        for (i = 0; i < k; ++i) {
            if (updateCentroid(clusters[i], centroids[i], vectors, d, vecOfSums)) {
                change = 1;
            }
        }
        if (!change) { break; }
    }

    /*build python list from p3*/
    python_list = PyList_New((k * (n + 1)));
    for (i = 0; i < (k * (n + 1)); ++i){
        PyObject* python_float = Py_BuildValue("f", p3[i]);
        PyList_SetItem(python_list, i, python_float);
    }

    free(centroids);
    free(vectors);
    free(clusters);
    free(p1);
    free(p2);
    free(p3);
    free(vecOfSums);
    return python_list;
}

/*calculates distance between to vectors
returns the standard euclidean distance*/
double getDistance(double *vec1, double *vec2, int d) {
    double sum = 0;
    int i;
    for (i = 0; i < d; ++i) {
        sum += (double) (vec1[i] - vec2[i])*(vec1[i] - vec2[i]);
    }
    return sum;
}

/*update new centroid after change in cluster
returns 0 if no change was made to centroid, else returns */
int updateCentroid(int *clusterVec, double *centroidVec, double **allVecs, const int d, double *vecOfSums) {
    int numOfElms, i, j, flag;
    double *curr;
    numOfElms = *clusterVec; /*first elm indicates number of vectors in cluster*/
    for (i = 0; i < d; ++i) { /*reset vecOfSums to 0 */
        vecOfSums[i] = 0;
    }
    flag = 0;/*to indicate whether a change has been made*/
    for (i = 1; i <= numOfElms; ++i) {
        curr = allVecs[clusterVec[i]];
        for (j = 0; j < d; ++j) {
            vecOfSums[j] += curr[j];
        }
    }
    for (i = 0; i < d; ++i) {
        vecOfSums[i] = (vecOfSums[i] / (double) numOfElms);
        if (centroidVec[i] != vecOfSums[i]) {
            centroidVec[i] = vecOfSums[i];
            flag = 1;
        }
    }
    return flag;
}

/*for given vector find the closest cluster according to distance from centroid*/
/*returns index of closest cluster*/
int findClosestCluster(double *x, double **centroids, int k, int d) {
    double distance,min;
    int i,minIndex;
    min = getDistance(x, centroids[0], d);
    minIndex = 0;
    for (i = 1; i < k; ++i) {
        distance = getDistance(x, centroids[i], d);/*calculate all the distances to x*/
        if (distance<min) {
            min = distance;
            minIndex = i;
        }
    }
    return minIndex;
}


/*module definition starts here*/

static PyMethodDef kmeansMethods[] = {
    {"kmeans",
      (PyCFunction) kmeans,
      METH_VARARGS,
      PyDoc_STR("K means clustering algorithm")},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    NULL,
    -1,
    kmeansMethods
};

PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}