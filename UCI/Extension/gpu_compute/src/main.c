#include "Python.h"

static PyObject* gpu_compute(PyObject* self)
{
return Py_BuildValue("s", "Hello, Python extensions!!");
}

static PyMethodDef gpu_compute_funcs[] = {
{"gpu_compute", (PyCFunction)gpu_compute,
METH_NOARGS},
{NULL, NULL}
};

static struct PyModuleDef Gpu_compute =
{
    PyModuleDef_HEAD_INIT,
    "gpu_compute", /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    gpu_compute_funcs
};

PyMODINIT_FUNC PyInit_Gpu_compute(void)
{
    return PyModule_Create(&Gpu_compute);
}