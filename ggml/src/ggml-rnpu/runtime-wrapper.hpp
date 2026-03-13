#pragma once
#include <cstdint>
#include <cstdlib>

enum npuErr_t {
    npuSuccess = 0,
    npuError
};

struct npuDeviceProp {
    const char * name;
};

const char * npuGetErrorString(npuErr_t error);
const char * npuGetLastError();

npuErr_t npuGetDeviceProperties(npuDeviceProp * prop, int device);
void npuSetDevice(int device);
void npuGetDevice(int * device);
npuErr_t npurtGetDeviceCount(uint32_t * count);
npuErr_t npuDeviceCanAccessPeer(int * can_access, int device, int peer_device);
npuErr_t npuDeviceSynchronize();

enum npuMemcpy_t {
    npuMemcpyHostToDevice,
    npuMemcpyDeviceToHost,
    npuMemcpyDeviceToDevice,
};

npuErr_t npuMalloc(void ** ptr, size_t size);
npuErr_t npuFree(void * ptr);
npuErr_t npuMallocHost(void ** ptr, size_t size);
npuErr_t npuFreeHost(void * ptr);

npuErr_t npuMemset(void * ptr, int value, size_t size);
npuErr_t npuMemcpy(void * dst, const void * src, size_t size, npuMemcpy_t kind);
npuErr_t npuMemGetInfo(size_t * free, size_t * total);