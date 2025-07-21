#include <cstddef>
#include <string.h>
#include <string>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cstdint>

#include "ggml-impl.h"
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
    return npuSuccess;
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


#define DEVICE_MALIGNMENT 128
#define HOST_MALIGNMENT 128
npuErr_t npuMalloc(void ** ptr, size_t size) {
    if (size == 0) {
        *ptr = nullptr;
        return npuSuccess;
    }

    const int alignment = 128;
    void * alloc_ptr = nullptr;
    int result = posix_memalign(&alloc_ptr, alignment, size);

    if (result != 0) {
        *ptr = nullptr;
        return npuError;
    }

    *ptr = alloc_ptr;

    return npuSuccess;
}

npuErr_t npuFree(void * ptr) {
    free(ptr);

    return npuSuccess;
}

npuErr_t npuMallocHost(void ** ptr, size_t size) {
    if (size == 0) {
        *ptr = nullptr;
        return npuSuccess;
    }

    const int alignment = 128;
    void * alloc_ptr = nullptr;
    int result = posix_memalign(&alloc_ptr, alignment, size);

    if (result != 0) {
        *ptr = nullptr;
        return npuError;
    }

    *ptr = alloc_ptr;

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

npuErr_t npuMemcpy(void * dst, const void * src, size_t size, npuMemcpy_t kind) {
    void * res = nullptr;

    if (kind == npuMemcpyHostToDevice ||
        kind == npuMemcpyDeviceToDevice ||
        kind == npuMemcpyDeviceToHost) {
        res = memcpy(dst, src, size);
    }

    if (res != dst) {
        return npuError;
    }

    return npuSuccess;
}

npuErr_t npuMemGetInfo(size_t * free, size_t * total) {
    *free = 16 * 1024L * 1024L * 1024L; // FIXME: always have 1GB free memory
    *total = 16 * 1024L * 1024L * 1024L; // 16GB total memory
    return npuSuccess;
}




