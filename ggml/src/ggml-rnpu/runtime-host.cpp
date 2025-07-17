#include <cstddef>
#include <string.h>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdint>

#include "runtime-host.hpp"

const char * npuGetErrorString(npuErr_t error) {
    return "npu runtime error\n";
}
const char * npuGetLastError() {
    return "npu last error\n";
}

npuErr_t npuGetDeviceProperties(npuDeviceProp * prop, int device){
    (void)(device);
    prop->name = "RNPU-0";
}

void npuSetDevice(int device) {
    (void)(device);
}

void npuGetDevice(int * device) {
    *device = 0;
}

npuErr_t npurtGetDeviceCount(uint32_t * count) {
    *count = 1;
    return npuSuccess;
}

npuErr_t npuDeviceCanAccessPeer(int * can_access, int device, int peer_device) {
    (void)(device);
    (void)(peer_device);

    *can_access = 0;

    return npuSuccess;
}

npuErr_t npuDeviceSynchronize() {
    return npuSuccess;
}

npuErr_t npuMalloc(void ** ptr, size_t size) {
    void * res = malloc(size);
    *ptr = res;

    if (res == nullptr)
        return npuError;

    return npuSuccess;
}

npuErr_t npuFree(void * ptr) {
    free(ptr);

    return npuSuccess;
}

npuErr_t npuMallocHost(void ** ptr, size_t size) {
    void * res = malloc(size);
    *ptr = res;

    if (res == nullptr)
        return npuError;

    return npuSuccess;
}

npuErr_t npuFreeHost(void * ptr) {
    free(ptr);

    return npuSuccess;
}

npuErr_t npuMemset(void * ptr, int value, size_t size) {
    void * res = memset(ptr, value, size);

    if (res != ptr)
        return npuError;

    return npuSuccess;
}

npuErr_t npuMemGetInfo(size_t * free, size_t * total) {
    *free = 1024L * 1024 * 1024; // FIXME: always have 1GB free memory
    *total = 16 * 1024L * 1024L * 1024L; // 16GB total memory
    return npuSuccess;
}




