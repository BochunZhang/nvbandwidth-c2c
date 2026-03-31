/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Custom concurrent-transport testcases.
// Contains:
//   - HostToAnyCE / AnyToHostCE  (moved from testcases_ce.cpp)
//   - host_to_device_ce_ce / host_to_device_ce_sm
//   - device_to_host_ce_ce / device_to_host_ce_sm
//   - device_to_device_ce_ce / device_to_device_ce_sm

#include <cuda.h>

#include "common.h"
#include "output.h"
#include "testcase.h"
#include "memcpy.h"

// ---------------------------------------------------------------------------
// Moved from testcases_ce.cpp
// ---------------------------------------------------------------------------

// void HostToAnyCE::run(unsigned long long size, unsigned long long loopCount) {
//     VERBOSE << "\n=== HostToAnyCE Test Configuration ===" << std::endl;
//     VERBOSE << "GPU IDs: ";
//     for (int gpuId : gpuIds) { VERBOSE << gpuId << " "; }
//     VERBOSE << std::endl;
//     VERBOSE << "Streams per GPU: " << ::streamCount << std::endl;
//     VERBOSE << "Buffer size: " << (size / _MiB) << " MiB" << std::endl;
//     VERBOSE << "Loop count: " << loopCount << std::endl;
//     VERBOSE << "====================================\n" << std::endl;

//     PeerValueMatrix<double> bandwidthValues(1, gpuIds.size(), key);
//     CustomMemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE(), PREFER_SRC_CONTEXT, MemcpyOperation::CONCURRENT_BW);

//     anyHostHelper(size, memcpyInstance, gpuIds, ::streamCount, bandwidthValues, true);

//     output->addTestcaseResults(bandwidthValues,
//         "memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s) with multi-stream");
// }

// void AnyToHostCE::run(unsigned long long size, unsigned long long loopCount) {
//     VERBOSE << "\n=== AnyToHostCE Test Configuration ===" << std::endl;
//     VERBOSE << "GPU IDs: ";
//     for (int gpuId : gpuIds) { VERBOSE << gpuId << " "; }
//     VERBOSE << std::endl;
//     VERBOSE << "Streams per GPU: " << ::streamCount << std::endl;
//     VERBOSE << "Buffer size: " << (size / _MiB) << " MiB" << std::endl;
//     VERBOSE << "Loop count: " << loopCount << std::endl;
//     VERBOSE << "====================================\n" << std::endl;

//     PeerValueMatrix<double> bandwidthValues(1, gpuIds.size(), key);
//     CustomMemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE(), PREFER_SRC_CONTEXT, MemcpyOperation::CONCURRENT_BW);

//     // sourceIsHost=false: GPU -> Host
//     anyHostHelper(size, memcpyInstance, gpuIds, ::streamCount, bandwidthValues, false);

//     output->addTestcaseResults(bandwidthValues,
//         "memcpy CE CPU(row) <- GPU(column) bandwidth (GB/s) with multi-stream");
// }



// ---------------------------------------------------------------------------
// host_to_device_ce_ce: two CE streams, host->device (NVLink-C2C)
// ---------------------------------------------------------------------------
void HostToDeviceCECE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (1, deviceCount, key + "_ce1");
    PeerValueMatrix<double> bw1  (1, deviceCount, key + "_ce2");
    PeerValueMatrix<double> bwTotal(1, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostBuffer hostBuffer1(size, deviceId), hostBuffer2(size, deviceId);
        DeviceBuffer deviceBuffer1(size, deviceId), deviceBuffer2(size, deviceId);

        std::vector<const MemcpyBuffer*> srcBufs = {&hostBuffer1, &hostBuffer2};
        std::vector<const MemcpyBuffer*> dstBufs = {&deviceBuffer1, &deviceBuffer2};
        std::vector<InitiatorType> types = {InitiatorType::CE, InitiatorType::CE};

        auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
        bw0   .value(0, deviceId) = results[0];
        bw1   .value(0, deviceId) = results[1];
        bwTotal.value(0, deviceId) = results[0] + results[1];
    }

    output->addTestcaseResults(bw0,    "CE stream0: CPU -> GPU bandwidth (GB/s)");
    output->addTestcaseResults(bw1,    "CE stream1: CPU -> GPU bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal,"total:   CPU -> GPU bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// host_to_device_ce_sm: CE + SM streams, host->device (NVLink-C2C)
// ---------------------------------------------------------------------------
void HostToDeviceCESM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (1, deviceCount, key + "_ce1");
    PeerValueMatrix<double> bw1  (1, deviceCount, key + "_ce2");
    PeerValueMatrix<double> bwTotal(1, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostBuffer hostBuffer1(size, deviceId), hostBuffer2(size, deviceId);
        DeviceBuffer deviceBuffer1(size, deviceId), deviceBuffer2(size, deviceId);

        std::vector<const MemcpyBuffer*> srcBufs = {&hostBuffer1, &hostBuffer2};
        std::vector<const MemcpyBuffer*> dstBufs = {&deviceBuffer1, &deviceBuffer2};
        std::vector<InitiatorType> types = {InitiatorType::CE, InitiatorType::SM};

        auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
        bw0   .value(0, deviceId) = results[0];
        bw1   .value(0, deviceId) = results[1];
        bwTotal.value(0, deviceId) = results[0] + results[1];
    }

    output->addTestcaseResults(bw0,    "CE stream: CPU -> GPU bandwidth (GB/s)");
    output->addTestcaseResults(bw1,    "SM stream: CPU -> GPU bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal,"total:   CPU -> GPU bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// device_to_host_ce_ce: two CE streams, device->host (NVLink-C2C)
// ---------------------------------------------------------------------------
void DeviceToHostCECE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (1, deviceCount, key + "_ce1");
    PeerValueMatrix<double> bw1  (1, deviceCount, key + "_ce2");
    PeerValueMatrix<double> bwTotal(1, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostBuffer hostBuffer1(size, deviceId), hostBuffer2(size, deviceId);
        DeviceBuffer deviceBuffer1(size, deviceId), deviceBuffer2(size, deviceId);

        std::vector<const MemcpyBuffer*> srcBufs = {&deviceBuffer1, &deviceBuffer2};
        std::vector<const MemcpyBuffer*> dstBufs = {&hostBuffer1, &hostBuffer2};
        std::vector<InitiatorType> types = {InitiatorType::CE, InitiatorType::CE};

        auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
        bw0   .value(0, deviceId) = results[0];
        bw1   .value(0, deviceId) = results[1];
        bwTotal.value(0, deviceId) = results[0] + results[1];
    }

    output->addTestcaseResults(bw0,    "CE stream0: CPU <- GPU bandwidth (GB/s)");
    output->addTestcaseResults(bw1,    "CE stream1: CPU <- GPU bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal,"total:   CPU <- GPU bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// device_to_host_ce_sm: CE + SM streams, device->host (NVLink-C2C)
// ---------------------------------------------------------------------------
void DeviceToHostCESM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (1, deviceCount, key + "_ce");
    PeerValueMatrix<double> bw1  (1, deviceCount, key + "_sm");
    PeerValueMatrix<double> bwTotal(1, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostBuffer hostBuffer1(size, deviceId), hostBuffer2(size, deviceId);
        DeviceBuffer deviceBuffer1(size, deviceId), deviceBuffer2(size, deviceId);

        std::vector<const MemcpyBuffer*> srcBufs = {&deviceBuffer1, &deviceBuffer2};
        std::vector<const MemcpyBuffer*> dstBufs = {&hostBuffer1, &hostBuffer2};
        std::vector<InitiatorType> types = {InitiatorType::CE, InitiatorType::SM};

        auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
        bw0   .value(0, deviceId) = results[0];
        bw1   .value(0, deviceId) = results[1];
        bwTotal.value(0, deviceId) = results[0] + results[1];
    }

    output->addTestcaseResults(bw0,    "CE stream: CPU <- GPU bandwidth (GB/s)");
    output->addTestcaseResults(bw1,    "SM stream: CPU <- GPU bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal,"total:   CPU <- GPU bandwidth (GB/s)");
}


// ---------------------------------------------------------------------------
// device_to_device_read_ce_ce: two CE streams per peer pair (NVLink P2P)
// ---------------------------------------------------------------------------
void DeviceToDeviceReadCECE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (1, deviceCount, key + "_ce1");
    PeerValueMatrix<double> bw1  (1, deviceCount, key + "_ce2");
    PeerValueMatrix<double> bwTotal(1, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_DST_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int srcId = 0; srcId < deviceCount; srcId++) {
        for (int dstId = 0; dstId < deviceCount; dstId++) {
            if (srcId == dstId) {
                continue;
            }

            DeviceBuffer srcBuffer1(size, srcId), srcBuffer2(size, srcId);
            DeviceBuffer dstBuffer1(size, dstId), dstBuffer2(size, dstId);
            
            if (!srcBuffer1.enablePeerAcess(dstBuffer1) || !srcBuffer2.enablePeerAcess(dstBuffer2)) {
                continue;
            }
            
            std::vector<const MemcpyBuffer*> srcBufs = {&srcBuffer1, &srcBuffer2};
            std::vector<const MemcpyBuffer*> dstBufs = {&dstBuffer1, &dstBuffer2};
            std::vector<InitiatorType> types = {InitiatorType::CE, InitiatorType::CE};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bw0   .value(srcId, dstId) = results[0];
            bw1   .value(srcId, dstId) = results[1];
            bwTotal.value(srcId, dstId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bw0,    "CE stream0: GPU -> GPU bandwidth (GB/s)");
    output->addTestcaseResults(bw1,    "CE stream1: GPU -> GPU bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal,"total:   GPU -> GPU bandwidth (GB/s)");
}


// ---------------------------------------------------------------------------
// device_to_device_write_ce_ce: two CE streams per peer pair (NVLink P2P)
// ---------------------------------------------------------------------------
void DeviceToDeviceWriteCECE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (1, deviceCount, key + "_ce1");
    PeerValueMatrix<double> bw1  (1, deviceCount, key + "_ce2");
    PeerValueMatrix<double> bwTotal(1, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int srcId = 0; srcId < deviceCount; srcId++) {
        for (int dstId = 0; dstId < deviceCount; dstId++) {
            if (srcId == dstId) {
                continue;
            }
        
            DeviceBuffer srcBuffer1(size, srcId), srcBuffer2(size, srcId);
            DeviceBuffer dstBuffer1(size, dstId), dstBuffer2(size, dstId);

            if (!srcBuffer1.enablePeerAcess(dstBuffer1) || !srcBuffer2.enablePeerAcess(dstBuffer2)) {
                continue;
            }

            std::vector<const MemcpyBuffer*> srcBufs = {&srcBuffer1, &srcBuffer2};
            std::vector<const MemcpyBuffer*> dstBufs = {&dstBuffer1, &dstBuffer2};
            std::vector<InitiatorType> types = {InitiatorType::CE, InitiatorType::CE};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bw0   .value(srcId, dstId) = results[0];
            bw1   .value(srcId, dstId) = results[1];
            bwTotal.value(srcId, dstId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bw0,    "CE stream0: GPU -> GPU bandwidth (GB/s)");
    output->addTestcaseResults(bw1,    "CE stream1: GPU -> GPU bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal,"total:   GPU -> GPU bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// device_to_device_read_ce_sm: CE + SM streams per peer pair (NVLink P2P)
// ---------------------------------------------------------------------------
void DeviceToDeviceReadCESM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (1, deviceCount, key + "_ce");
    PeerValueMatrix<double> bw1  (1, deviceCount, key + "_sm");
    PeerValueMatrix<double> bwTotal(1, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_DST_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int srcId = 0; srcId < deviceCount; srcId++) {
        for (int dstId = 0; dstId < deviceCount; dstId++) {
            if (srcId == dstId) {
                continue;
            }

            DeviceBuffer srcBuffer1(size, srcId), srcBuffer2(size, srcId);
            DeviceBuffer dstBuffer1(size, dstId), dstBuffer2(size, dstId);
            
            if (!srcBuffer1.enablePeerAcess(dstBuffer1) || !srcBuffer2.enablePeerAcess(dstBuffer2)) {
                continue;
            }
            
            std::vector<const MemcpyBuffer*> srcBufs = {&srcBuffer1, &srcBuffer2};
            std::vector<const MemcpyBuffer*> dstBufs = {&dstBuffer1, &dstBuffer2};
            std::vector<InitiatorType> types = {InitiatorType::CE, InitiatorType::SM};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bw0   .value(srcId, dstId) = results[0];
            bw1   .value(srcId, dstId) = results[1];
            bwTotal.value(srcId, dstId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bw0,    "CE stream: GPU -> GPU bandwidth (GB/s)");
    output->addTestcaseResults(bw1,    "SM stream: GPU -> GPU bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal,"total:   GPU -> GPU bandwidth (GB/s)");
}


// ---------------------------------------------------------------------------
// device_to_device_write_ce_sm: CE + SM streams per peer pair (NVLink P2P)
// ---------------------------------------------------------------------------
void DeviceToDeviceWriteCESM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (1, deviceCount, key + "_ce");
    PeerValueMatrix<double> bw1  (1, deviceCount, key + "_sm");
    PeerValueMatrix<double> bwTotal(1, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int srcId = 0; srcId < deviceCount; srcId++) {
        for (int dstId = 0; dstId < deviceCount; dstId++) {
            if (srcId == dstId) {
                continue;
            }

            DeviceBuffer srcBuffer1(size, srcId), srcBuffer2(size, srcId);
            DeviceBuffer dstBuffer1(size, dstId), dstBuffer2(size, dstId);

            if (!srcBuffer1.enablePeerAcess(dstBuffer1) || !srcBuffer2.enablePeerAcess(dstBuffer2)) {
                continue;
            }

            std::vector<const MemcpyBuffer*> srcBufs = {&srcBuffer1, &srcBuffer2};
            std::vector<const MemcpyBuffer*> dstBufs = {&dstBuffer1, &dstBuffer2};
            std::vector<InitiatorType> types = {InitiatorType::CE, InitiatorType::SM};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bw0   .value(srcId, dstId) = results[0];
            bw1   .value(srcId, dstId) = results[1];
            bwTotal.value(srcId, dstId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bw0,    "CE stream: GPU -> GPU bandwidth (GB/s)");
    output->addTestcaseResults(bw1,    "SM stream: GPU -> GPU bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal,"total:   GPU -> GPU bandwidth (GB/s)");
}
