/*
 * query_async_engines.cpp
 *
 * Standalone tool: query and print async engine (Copy Engine) information
 * for every visible CUDA device.
 *
 * Build: included in CMakeLists.txt as the 'query_async_engines' target.
 * Usage: ./query_async_engines [--json]
 *
 * Output fields per device:
 *   asyncEngineCount   – number of hardware copy engines (CE)
 *   gpuOverlap         – whether kernel exec and memcpy can overlap
 *   memcpyEngines      – alias: asyncEngineCount (CE count)
 *   unifiedAddressing  – whether UVA is enabled
 *   canMapHostMemory   – whether host pinned memory is mappable
 *   pciBusId / pciDeviceId / pciDomainId
 */

#include <cuda.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ── tiny CU error check ───────────────────────────────────────────────────────
#define CU_CHECK(call)                                                  \
    do {                                                                \
        CUresult _r = (call);                                           \
        if (_r != CUDA_SUCCESS) {                                       \
            const char *_s = nullptr;                                   \
            cuGetErrorString(_r, &_s);                                  \
            fprintf(stderr, "CUDA error at %s:%d – %s\n",              \
                    __FILE__, __LINE__, _s ? _s : "unknown");           \
            return 1;                                                    \
        }                                                               \
    } while (0)

// ── per-device info ───────────────────────────────────────────────────────────
struct DeviceEngineInfo {
    int    index;
    char   name[256];
    int    asyncEngineCount;    // number of copy engines
    int    gpuOverlap;          // kernel + memcpy overlap support
    int    unifiedAddressing;   // UVA enabled
    int    canMapHostMemory;    // host pinned memory mappable
    int    pciBus;
    int    pciDevice;
    int    pciDomain;
};

static int getAttr(int *out, CUdevice dev, CUdevice_attribute attr) {
    CUresult r = cuDeviceGetAttribute(out, attr, dev);
    if (r != CUDA_SUCCESS) { *out = -1; }
    return (r == CUDA_SUCCESS) ? 0 : -1;
}

// ── plain-text printer ────────────────────────────────────────────────────────
static void printText(const std::vector<DeviceEngineInfo> &devs) {
    printf("%-4s  %-36s  %17s  %11s  %17s  %15s  %s\n",
           "Idx", "Device Name",
           "asyncEngineCount", "gpuOverlap",
           "unifiedAddressing", "canMapHostMem",
           "PCI (domain:bus:dev)");
    printf("%s\n", std::string(110, '-').c_str());

    for (const auto &d : devs) {
        printf("%-4d  %-36s  %17d  %11d  %17d  %15d  %04x:%02x:%02x\n",
               d.index, d.name,
               d.asyncEngineCount, d.gpuOverlap,
               d.unifiedAddressing, d.canMapHostMemory,
               d.pciDomain, d.pciBus, d.pciDevice);
    }
    printf("\n");

    // summary
    printf("Legend:\n");
    printf("  asyncEngineCount  – number of hardware Copy Engines (CE) per GPU\n");
    printf("  gpuOverlap        – 1 if kernel execution and memcpy can overlap\n");
    printf("  unifiedAddressing – 1 if UVA is active (required for peer access)\n");
    printf("  canMapHostMem     – 1 if pinned host memory is directly mappable\n");
}

// ── JSON printer ──────────────────────────────────────────────────────────────
static void printJson(const std::vector<DeviceEngineInfo> &devs) {
    printf("{\n  \"async_engines\": [\n");
    for (size_t i = 0; i < devs.size(); i++) {
        const auto &d = devs[i];
        printf("    {\n");
        printf("      \"index\": %d,\n", d.index);
        printf("      \"name\": \"%s\",\n", d.name);
        printf("      \"asyncEngineCount\": %d,\n", d.asyncEngineCount);
        printf("      \"gpuOverlap\": %d,\n", d.gpuOverlap);
        printf("      \"unifiedAddressing\": %d,\n", d.unifiedAddressing);
        printf("      \"canMapHostMemory\": %d,\n", d.canMapHostMemory);
        printf("      \"pciBus\": %d,\n", d.pciBus);
        printf("      \"pciDevice\": %d,\n", d.pciDevice);
        printf("      \"pciDomain\": %d\n", d.pciDomain);
        printf("    }%s\n", i + 1 < devs.size() ? "," : "");
    }
    printf("  ]\n}\n");
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    bool jsonMode = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0 || strcmp(argv[i], "-j") == 0)
            jsonMode = true;
    }

    CU_CHECK(cuInit(0));

    int deviceCount = 0;
    CU_CHECK(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return 1;
    }

    std::vector<DeviceEngineInfo> devs(deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        DeviceEngineInfo &d = devs[i];
        d.index = i;

        CUdevice dev;
        CU_CHECK(cuDeviceGet(&dev, i));
        CU_CHECK(cuDeviceGetName(d.name, sizeof(d.name), dev));

        getAttr(&d.asyncEngineCount,  dev, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT);
        getAttr(&d.gpuOverlap,        dev, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP);
        getAttr(&d.unifiedAddressing, dev, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING);
        getAttr(&d.canMapHostMemory,  dev, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY);
        getAttr(&d.pciBus,            dev, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID);
        getAttr(&d.pciDevice,         dev, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID);
        getAttr(&d.pciDomain,         dev, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID);
    }

    if (jsonMode)
        printJson(devs);
    else
        printText(devs);

    return 0;
}
