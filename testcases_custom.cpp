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
// host_to_device_sm_ce: SM + CE streams, host->device (NVLink-C2C)
// ---------------------------------------------------------------------------
void HostToDeviceSMCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (1, deviceCount, key + "_sm1");
    PeerValueMatrix<double> bw1  (1, deviceCount, key + "_ce1");
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
        std::vector<InitiatorType> types = {InitiatorType::SM, InitiatorType::CE};

        auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
        bw0   .value(0, deviceId) = results[0];
        bw1   .value(0, deviceId) = results[1];
        bwTotal.value(0, deviceId) = results[0] + results[1];
    }

    output->addTestcaseResults(bw0,    "SM stream: CPU -> GPU bandwidth (GB/s)");
    output->addTestcaseResults(bw1,    "CE stream: CPU -> GPU bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal,"total:   CPU -> GPU bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// host_to_device_sm_sm: two SM streams, host->device (NVLink-C2C)
// ---------------------------------------------------------------------------
void HostToDeviceSMSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (1, deviceCount, key + "_sm1");
    PeerValueMatrix<double> bw1  (1, deviceCount, key + "_sm2");
    PeerValueMatrix<double> bwTotal(1, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostBuffer hostBuffer1(size, deviceId), hostBuffer2(size, deviceId);
        DeviceBuffer deviceBuffer1(size, deviceId), deviceBuffer2(size, deviceId);

        std::vector<const MemcpyBuffer*> srcBufs = {&hostBuffer1, &hostBuffer2};
        std::vector<const MemcpyBuffer*> dstBufs = {&deviceBuffer1, &deviceBuffer2};
        std::vector<InitiatorType> types = {InitiatorType::SM, InitiatorType::SM};

        auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
        bw0   .value(0, deviceId) = results[0];
        bw1   .value(0, deviceId) = results[1];
        bwTotal.value(0, deviceId) = results[0] + results[1];
    }

    output->addTestcaseResults(bw0,    "SM stream0: CPU -> GPU bandwidth (GB/s)");
    output->addTestcaseResults(bw1,    "SM stream1: CPU -> GPU bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal,"total:      CPU -> GPU bandwidth (GB/s)");
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
// device_to_host_sm_ce: SM + CE streams, device->host (NVLink-C2C)
// ---------------------------------------------------------------------------
void DeviceToHostSMCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (1, deviceCount, key + "_sm");
    PeerValueMatrix<double> bw1  (1, deviceCount, key + "_ce");
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
        std::vector<InitiatorType> types = {InitiatorType::SM, InitiatorType::CE};

        auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
        bw0   .value(0, deviceId) = results[0];
        bw1   .value(0, deviceId) = results[1];
        bwTotal.value(0, deviceId) = results[0] + results[1];
    }

    output->addTestcaseResults(bw0,    "SM stream: CPU <- GPU bandwidth (GB/s)");
    output->addTestcaseResults(bw1,    "CE stream: CPU <- GPU bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal,"total:   CPU <- GPU bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// device_to_host_sm_sm: two SM streams, device->host (NVLink-C2C)
// ---------------------------------------------------------------------------
void DeviceToHostSMSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (1, deviceCount, key + "_sm1");
    PeerValueMatrix<double> bw1  (1, deviceCount, key + "_sm2");
    PeerValueMatrix<double> bwTotal(1, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostBuffer hostBuffer1(size, deviceId), hostBuffer2(size, deviceId);
        DeviceBuffer deviceBuffer1(size, deviceId), deviceBuffer2(size, deviceId);

        std::vector<const MemcpyBuffer*> srcBufs = {&deviceBuffer1, &deviceBuffer2};
        std::vector<const MemcpyBuffer*> dstBufs = {&hostBuffer1, &hostBuffer2};
        std::vector<InitiatorType> types = {InitiatorType::SM, InitiatorType::SM};

        auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
        bw0   .value(0, deviceId) = results[0];
        bw1   .value(0, deviceId) = results[1];
        bwTotal.value(0, deviceId) = results[0] + results[1];
    }

    output->addTestcaseResults(bw0,    "SM stream0: CPU <- GPU bandwidth (GB/s)");
    output->addTestcaseResults(bw1,    "SM stream1: CPU <- GPU bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal,"total:      CPU <- GPU bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// device_to_device_read_ce_ce: two CE streams per peer pair (NVLink P2P)
// ---------------------------------------------------------------------------
void DeviceToDeviceReadCECE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (deviceCount, deviceCount, key + "_ce1");
    PeerValueMatrix<double> bw1  (deviceCount, deviceCount, key + "_ce2");
    PeerValueMatrix<double> bwTotal(deviceCount, deviceCount, key + "_total");

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

            auto results = memcpyInstance.doMemcpyVector(srcBufs, srcBufs, types);
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
    PeerValueMatrix<double> bw0  (deviceCount, deviceCount, key + "_ce");
    PeerValueMatrix<double> bw1  (deviceCount, deviceCount, key + "_sm");
    PeerValueMatrix<double> bwTotal(deviceCount, deviceCount, key + "_total");

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

            auto results = memcpyInstance.doMemcpyVector(dstBufs, srcBufs, types);
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
// device_to_device_read_sm_ce: SM + CE streams per peer pair (NVLink P2P)
// ---------------------------------------------------------------------------
void DeviceToDeviceReadSMCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (deviceCount, deviceCount, key + "_sm");
    PeerValueMatrix<double> bw1  (deviceCount, deviceCount, key + "_ce");
    PeerValueMatrix<double> bwTotal(deviceCount, deviceCount, key + "_total");

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
            std::vector<InitiatorType> types = {InitiatorType::SM, InitiatorType::CE};

            auto results = memcpyInstance.doMemcpyVector(dstBufs, srcBufs, types);
            bw0   .value(srcId, dstId) = results[0];
            bw1   .value(srcId, dstId) = results[1];
            bwTotal.value(srcId, dstId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bw0,    "SM stream: GPU -> GPU bandwidth (GB/s)");
    output->addTestcaseResults(bw1,    "CE stream: GPU -> GPU bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal,"total:   GPU -> GPU bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// device_to_device_read_sm_sm: two SM streams per peer pair (NVLink P2P, read)
// ---------------------------------------------------------------------------
void DeviceToDeviceReadSMSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (deviceCount, deviceCount, key + "_sm1");
    PeerValueMatrix<double> bw1  (deviceCount, deviceCount, key + "_sm2");
    PeerValueMatrix<double> bwTotal(deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_DST_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int srcId = 0; srcId < deviceCount; srcId++) {
        for (int dstId = 0; dstId < deviceCount; dstId++) {
            if (srcId == dstId) continue;

            DeviceBuffer srcBuffer1(size, srcId), srcBuffer2(size, srcId);
            DeviceBuffer dstBuffer1(size, dstId), dstBuffer2(size, dstId);

            if (!srcBuffer1.enablePeerAcess(dstBuffer1) || !srcBuffer2.enablePeerAcess(dstBuffer2)) continue;

            std::vector<const MemcpyBuffer*> srcBufs = {&srcBuffer1, &srcBuffer2};
            std::vector<const MemcpyBuffer*> dstBufs = {&dstBuffer1, &dstBuffer2};
            std::vector<InitiatorType> types = {InitiatorType::SM, InitiatorType::SM};

            auto results = memcpyInstance.doMemcpyVector(dstBufs, srcBufs, types);
            bw0   .value(srcId, dstId) = results[0];
            bw1   .value(srcId, dstId) = results[1];
            bwTotal.value(srcId, dstId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bw0,    "SM stream0: GPU -> GPU bandwidth (GB/s)");
    output->addTestcaseResults(bw1,    "SM stream1: GPU -> GPU bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal,"total:      GPU -> GPU bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// device_to_device_write_ce_ce: two CE streams per peer pair (NVLink P2P)
// ---------------------------------------------------------------------------
void DeviceToDeviceWriteCECE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (deviceCount, deviceCount, key + "_ce1");
    PeerValueMatrix<double> bw1  (deviceCount, deviceCount, key + "_ce2");
    PeerValueMatrix<double> bwTotal(deviceCount, deviceCount, key + "_total");

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
// device_to_device_write_ce_sm: CE + SM streams per peer pair (NVLink P2P)
// ---------------------------------------------------------------------------
void DeviceToDeviceWriteCESM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (deviceCount, deviceCount, key + "_ce");
    PeerValueMatrix<double> bw1  (deviceCount, deviceCount, key + "_sm");
    PeerValueMatrix<double> bwTotal(deviceCount, deviceCount, key + "_total");

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

// ---------------------------------------------------------------------------
// device_to_device_write_sm_ce: SM + CE streams per peer pair (NVLink P2P)
// ---------------------------------------------------------------------------
void DeviceToDeviceWriteSMCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (deviceCount, deviceCount, key + "_sm");
    PeerValueMatrix<double> bw1  (deviceCount, deviceCount, key + "_ce");
    PeerValueMatrix<double> bwTotal(deviceCount, deviceCount, key + "_total");

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
            std::vector<InitiatorType> types = {InitiatorType::SM, InitiatorType::CE};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bw0   .value(srcId, dstId) = results[0];
            bw1   .value(srcId, dstId) = results[1];
            bwTotal.value(srcId, dstId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bw0,    "SM stream: GPU -> GPU bandwidth (GB/s)");
    output->addTestcaseResults(bw1,    "CE stream: GPU -> GPU bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal,"total:   GPU -> GPU bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// device_to_device_write_sm_sm: two SM streams per peer pair (NVLink P2P, write)
// ---------------------------------------------------------------------------
void DeviceToDeviceWriteSMSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (deviceCount, deviceCount, key + "_sm1");
    PeerValueMatrix<double> bw1  (deviceCount, deviceCount, key + "_sm2");
    PeerValueMatrix<double> bwTotal(deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int srcId = 0; srcId < deviceCount; srcId++) {
        for (int dstId = 0; dstId < deviceCount; dstId++) {
            if (srcId == dstId) continue;

            DeviceBuffer srcBuffer1(size, srcId), srcBuffer2(size, srcId);
            DeviceBuffer dstBuffer1(size, dstId), dstBuffer2(size, dstId);

            if (!srcBuffer1.enablePeerAcess(dstBuffer1) || !srcBuffer2.enablePeerAcess(dstBuffer2)) continue;

            std::vector<const MemcpyBuffer*> srcBufs = {&srcBuffer1, &srcBuffer2};
            std::vector<const MemcpyBuffer*> dstBufs = {&dstBuffer1, &dstBuffer2};
            std::vector<InitiatorType> types = {InitiatorType::SM, InitiatorType::SM};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bw0   .value(srcId, dstId) = results[0];
            bw1   .value(srcId, dstId) = results[1];
            bwTotal.value(srcId, dstId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bw0,    "SM stream0: GPU -> GPU bandwidth (GB/s)");
    output->addTestcaseResults(bw1,    "SM stream1: GPU -> GPU bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal,"total:      GPU -> GPU bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// device_to_host_ce_device_to_device_read_ce: concurrent D->H CE + DtoD read CE
// Row=primary(D->H src and DtoD read dst), Col=peer(DtoD read src)
// ---------------------------------------------------------------------------
void DeviceToHostCEDeviceToDeviceReadCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bwDtoH  (deviceCount, deviceCount, key + "_d2h_ce");
    PeerValueMatrix<double> bwDtoD  (deviceCount, deviceCount, key + "_d2d_read_ce");
    PeerValueMatrix<double> bwTotal (deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_DST_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int primId = 0; primId < deviceCount; primId++) {
        for (int peerId = 0; peerId < deviceCount; peerId++) {
            if (primId == peerId) continue;

            DeviceBuffer primBuf1(size, primId);  // D->H source
            DeviceBuffer primBuf2(size, primId);  // DtoD read destination
            DeviceBuffer peerBuf (size, peerId);  // DtoD read source
            HostBuffer   hostBuf (size, primId);  // D->H destination

            if (!primBuf2.enablePeerAcess(peerBuf)) continue;

            // Stream 0 (CE): primBuf1 -> hostBuf   [D->H, PREFER_DST_CONTEXT: host no ctx -> primId]
            // Stream 1 (CE): peerBuf  -> primBuf2  [DtoD read, PREFER_DST_CONTEXT: primId]
            std::vector<const MemcpyBuffer*> srcBufs = {&primBuf1, &peerBuf};
            std::vector<const MemcpyBuffer*> dstBufs = {&hostBuf,  &primBuf2};
            std::vector<InitiatorType> types = {InitiatorType::CE, InitiatorType::CE};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bwDtoH .value(primId, peerId) = results[0];
            bwDtoD .value(primId, peerId) = results[1];
            bwTotal.value(primId, peerId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bwDtoH,  "CE stream (D->H): CPU <- GPU(row) bandwidth (GB/s)");
    output->addTestcaseResults(bwDtoD,  "CE stream (DtoD read): GPU(row) <- GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal, "total bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// device_to_host_ce_device_to_device_read_sm: concurrent D->H CE + DtoD read SM
// ---------------------------------------------------------------------------
void DeviceToHostCEDeviceToDeviceReadSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bwDtoH  (deviceCount, deviceCount, key + "_d2h_ce");
    PeerValueMatrix<double> bwDtoD  (deviceCount, deviceCount, key + "_d2d_read_sm");
    PeerValueMatrix<double> bwTotal (deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_DST_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int primId = 0; primId < deviceCount; primId++) {
        for (int peerId = 0; peerId < deviceCount; peerId++) {
            if (primId == peerId) continue;

            DeviceBuffer primBuf1(size, primId);
            DeviceBuffer primBuf2(size, primId);
            DeviceBuffer peerBuf (size, peerId);
            HostBuffer   hostBuf (size, primId);

            if (!primBuf2.enablePeerAcess(peerBuf)) continue;

            std::vector<const MemcpyBuffer*> srcBufs = {&primBuf1, &peerBuf};
            std::vector<const MemcpyBuffer*> dstBufs = {&hostBuf,  &primBuf2};
            std::vector<InitiatorType> types = {InitiatorType::CE, InitiatorType::SM};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bwDtoH .value(primId, peerId) = results[0];
            bwDtoD .value(primId, peerId) = results[1];
            bwTotal.value(primId, peerId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bwDtoH,  "CE stream (D->H): CPU <- GPU(row) bandwidth (GB/s)");
    output->addTestcaseResults(bwDtoD,  "SM stream (DtoD read): GPU(row) <- GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal, "total bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// device_to_host_sm_device_to_device_read_ce: concurrent D->H SM + DtoD read CE
// ---------------------------------------------------------------------------
void DeviceToHostSMDeviceToDeviceReadCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bwDtoH  (deviceCount, deviceCount, key + "_d2h_sm");
    PeerValueMatrix<double> bwDtoD  (deviceCount, deviceCount, key + "_d2d_read_ce");
    PeerValueMatrix<double> bwTotal (deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_DST_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int primId = 0; primId < deviceCount; primId++) {
        for (int peerId = 0; peerId < deviceCount; peerId++) {
            if (primId == peerId) continue;

            DeviceBuffer primBuf1(size, primId);
            DeviceBuffer primBuf2(size, primId);
            DeviceBuffer peerBuf (size, peerId);
            HostBuffer   hostBuf (size, primId);

            if (!primBuf2.enablePeerAcess(peerBuf)) continue;

            std::vector<const MemcpyBuffer*> srcBufs = {&primBuf1, &peerBuf};
            std::vector<const MemcpyBuffer*> dstBufs = {&hostBuf,  &primBuf2};
            std::vector<InitiatorType> types = {InitiatorType::SM, InitiatorType::CE};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bwDtoH .value(primId, peerId) = results[0];
            bwDtoD .value(primId, peerId) = results[1];
            bwTotal.value(primId, peerId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bwDtoH,  "SM stream (D->H): CPU <- GPU(row) bandwidth (GB/s)");
    output->addTestcaseResults(bwDtoD,  "CE stream (DtoD read): GPU(row) <- GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal, "total bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// device_to_host_sm_device_to_device_read_sm: concurrent D->H SM + DtoD read SM
// ---------------------------------------------------------------------------
void DeviceToHostSMDeviceToDeviceReadSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bwDtoH  (deviceCount, deviceCount, key + "_d2h_sm");
    PeerValueMatrix<double> bwDtoD  (deviceCount, deviceCount, key + "_d2d_read_sm");
    PeerValueMatrix<double> bwTotal (deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_DST_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int primId = 0; primId < deviceCount; primId++) {
        for (int peerId = 0; peerId < deviceCount; peerId++) {
            if (primId == peerId) continue;

            DeviceBuffer primBuf1(size, primId);
            DeviceBuffer primBuf2(size, primId);
            DeviceBuffer peerBuf (size, peerId);
            HostBuffer   hostBuf (size, primId);

            if (!primBuf2.enablePeerAcess(peerBuf)) continue;

            std::vector<const MemcpyBuffer*> srcBufs = {&primBuf1, &peerBuf};
            std::vector<const MemcpyBuffer*> dstBufs = {&hostBuf,  &primBuf2};
            std::vector<InitiatorType> types = {InitiatorType::SM, InitiatorType::SM};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bwDtoH .value(primId, peerId) = results[0];
            bwDtoD .value(primId, peerId) = results[1];
            bwTotal.value(primId, peerId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bwDtoH,  "SM stream (D->H): CPU <- GPU(row) bandwidth (GB/s)");
    output->addTestcaseResults(bwDtoD,  "SM stream (DtoD read): GPU(row) <- GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal, "total bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// device_to_host_ce_device_to_device_write_ce: concurrent D->H CE + DtoD write CE
// Row=primary(D->H src and DtoD write src), Col=peer(DtoD write dst)
// ---------------------------------------------------------------------------
void DeviceToHostCEDeviceToDeviceWriteCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bwDtoH  (deviceCount, deviceCount, key + "_d2h_ce");
    PeerValueMatrix<double> bwDtoD  (deviceCount, deviceCount, key + "_d2d_write_ce");
    PeerValueMatrix<double> bwTotal (deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int primId = 0; primId < deviceCount; primId++) {
        for (int peerId = 0; peerId < deviceCount; peerId++) {
            if (primId == peerId) continue;

            DeviceBuffer primBuf1(size, primId);  // D->H source
            DeviceBuffer primBuf2(size, primId);  // DtoD write source
            DeviceBuffer peerBuf (size, peerId);  // DtoD write destination
            HostBuffer   hostBuf (size, primId);  // D->H destination

            if (!primBuf1.enablePeerAcess(peerBuf)) continue;

            // Stream 0 (CE): primBuf1 -> hostBuf  [D->H, PREFER_SRC_CONTEXT: primId]
            // Stream 1 (CE): primBuf2 -> peerBuf  [DtoD write, PREFER_SRC_CONTEXT: primId]
            std::vector<const MemcpyBuffer*> srcBufs = {&primBuf1, &primBuf2};
            std::vector<const MemcpyBuffer*> dstBufs = {&hostBuf,  &peerBuf};
            std::vector<InitiatorType> types = {InitiatorType::CE, InitiatorType::CE};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bwDtoH .value(primId, peerId) = results[0];
            bwDtoD .value(primId, peerId) = results[1];
            bwTotal.value(primId, peerId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bwDtoH,  "CE stream (D->H): CPU <- GPU(row) bandwidth (GB/s)");
    output->addTestcaseResults(bwDtoD,  "CE stream (DtoD write): GPU(row) -> GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal, "total bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// device_to_host_ce_device_to_device_write_sm: concurrent D->H CE + DtoD write SM
// ---------------------------------------------------------------------------
void DeviceToHostCEDeviceToDeviceWriteSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bwDtoH  (deviceCount, deviceCount, key + "_d2h_ce");
    PeerValueMatrix<double> bwDtoD  (deviceCount, deviceCount, key + "_d2d_write_sm");
    PeerValueMatrix<double> bwTotal (deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int primId = 0; primId < deviceCount; primId++) {
        for (int peerId = 0; peerId < deviceCount; peerId++) {
            if (primId == peerId) continue;

            DeviceBuffer primBuf1(size, primId);
            DeviceBuffer primBuf2(size, primId);
            DeviceBuffer peerBuf (size, peerId);
            HostBuffer   hostBuf (size, primId);

            if (!primBuf1.enablePeerAcess(peerBuf)) continue;

            std::vector<const MemcpyBuffer*> srcBufs = {&primBuf1, &primBuf2};
            std::vector<const MemcpyBuffer*> dstBufs = {&hostBuf,  &peerBuf};
            std::vector<InitiatorType> types = {InitiatorType::CE, InitiatorType::SM};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bwDtoH .value(primId, peerId) = results[0];
            bwDtoD .value(primId, peerId) = results[1];
            bwTotal.value(primId, peerId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bwDtoH,  "CE stream (D->H): CPU <- GPU(row) bandwidth (GB/s)");
    output->addTestcaseResults(bwDtoD,  "SM stream (DtoD write): GPU(row) -> GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal, "total bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// device_to_host_sm_device_to_device_write_ce: concurrent D->H SM + DtoD write CE
// ---------------------------------------------------------------------------
void DeviceToHostSMDeviceToDeviceWriteCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bwDtoH  (deviceCount, deviceCount, key + "_d2h_sm");
    PeerValueMatrix<double> bwDtoD  (deviceCount, deviceCount, key + "_d2d_write_ce");
    PeerValueMatrix<double> bwTotal (deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int primId = 0; primId < deviceCount; primId++) {
        for (int peerId = 0; peerId < deviceCount; peerId++) {
            if (primId == peerId) continue;

            DeviceBuffer primBuf1(size, primId);
            DeviceBuffer primBuf2(size, primId);
            DeviceBuffer peerBuf (size, peerId);
            HostBuffer   hostBuf (size, primId);

            if (!primBuf1.enablePeerAcess(peerBuf)) continue;

            std::vector<const MemcpyBuffer*> srcBufs = {&primBuf1, &primBuf2};
            std::vector<const MemcpyBuffer*> dstBufs = {&hostBuf,  &peerBuf};
            std::vector<InitiatorType> types = {InitiatorType::SM, InitiatorType::CE};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bwDtoH .value(primId, peerId) = results[0];
            bwDtoD .value(primId, peerId) = results[1];
            bwTotal.value(primId, peerId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bwDtoH,  "SM stream (D->H): CPU <- GPU(row) bandwidth (GB/s)");
    output->addTestcaseResults(bwDtoD,  "CE stream (DtoD write): GPU(row) -> GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal, "total bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// device_to_host_sm_device_to_device_write_sm: concurrent D->H SM + DtoD write SM
// ---------------------------------------------------------------------------
void DeviceToHostSMDeviceToDeviceWriteSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bwDtoH  (deviceCount, deviceCount, key + "_d2h_sm");
    PeerValueMatrix<double> bwDtoD  (deviceCount, deviceCount, key + "_d2d_write_sm");
    PeerValueMatrix<double> bwTotal (deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int primId = 0; primId < deviceCount; primId++) {
        for (int peerId = 0; peerId < deviceCount; peerId++) {
            if (primId == peerId) continue;

            DeviceBuffer primBuf1(size, primId);
            DeviceBuffer primBuf2(size, primId);
            DeviceBuffer peerBuf (size, peerId);
            HostBuffer   hostBuf (size, primId);

            if (!primBuf1.enablePeerAcess(peerBuf)) continue;

            std::vector<const MemcpyBuffer*> srcBufs = {&primBuf1, &primBuf2};
            std::vector<const MemcpyBuffer*> dstBufs = {&hostBuf,  &peerBuf};
            std::vector<InitiatorType> types = {InitiatorType::SM, InitiatorType::SM};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bwDtoH .value(primId, peerId) = results[0];
            bwDtoD .value(primId, peerId) = results[1];
            bwTotal.value(primId, peerId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bwDtoH,  "SM stream (D->H): CPU <- GPU(row) bandwidth (GB/s)");
    output->addTestcaseResults(bwDtoD,  "SM stream (DtoD write): GPU(row) -> GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal, "total bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// host_to_device_ce_device_to_device_read_ce: concurrent H->D CE + DtoD read CE
// Row=primary(H->D dst and DtoD read dst), Col=peer(DtoD read src)
// ---------------------------------------------------------------------------
void HostToDeviceCEDeviceToDeviceReadCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bwHtoD  (deviceCount, deviceCount, key + "_h2d_ce");
    PeerValueMatrix<double> bwDtoD  (deviceCount, deviceCount, key + "_d2d_read_ce");
    PeerValueMatrix<double> bwTotal (deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_DST_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int primId = 0; primId < deviceCount; primId++) {
        for (int peerId = 0; peerId < deviceCount; peerId++) {
            if (primId == peerId) continue;

            HostBuffer   hostBuf (size, primId);  // H->D source
            DeviceBuffer primBuf1(size, primId);  // H->D destination
            DeviceBuffer peerBuf (size, peerId);  // DtoD read source
            DeviceBuffer primBuf2(size, primId);  // DtoD read destination

            if (!primBuf2.enablePeerAcess(peerBuf)) continue;

            // Stream 0 (CE): hostBuf -> primBuf1  [H->D, PREFER_DST_CONTEXT: host no ctx -> primId]
            // Stream 1 (CE): peerBuf  -> primBuf2 [DtoD read, PREFER_DST_CONTEXT: primId]
            std::vector<const MemcpyBuffer*> srcBufs = {&hostBuf,  &peerBuf};
            std::vector<const MemcpyBuffer*> dstBufs = {&primBuf1, &primBuf2};
            std::vector<InitiatorType> types = {InitiatorType::CE, InitiatorType::CE};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bwHtoD .value(primId, peerId) = results[0];
            bwDtoD .value(primId, peerId) = results[1];
            bwTotal.value(primId, peerId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bwHtoD,  "CE stream (H->D): CPU -> GPU(row) bandwidth (GB/s)");
    output->addTestcaseResults(bwDtoD,  "CE stream (DtoD read): GPU(row) <- GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal, "total bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// host_to_device_ce_device_to_device_read_sm: concurrent H->D CE + DtoD read SM
// ---------------------------------------------------------------------------
void HostToDeviceCEDeviceToDeviceReadSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bwHtoD  (deviceCount, deviceCount, key + "_h2d_ce");
    PeerValueMatrix<double> bwDtoD  (deviceCount, deviceCount, key + "_d2d_read_sm");
    PeerValueMatrix<double> bwTotal (deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_DST_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int primId = 0; primId < deviceCount; primId++) {
        for (int peerId = 0; peerId < deviceCount; peerId++) {
            if (primId == peerId) continue;

            HostBuffer   hostBuf (size, primId);
            DeviceBuffer primBuf1(size, primId);
            DeviceBuffer peerBuf (size, peerId);
            DeviceBuffer primBuf2(size, primId);

            if (!primBuf2.enablePeerAcess(peerBuf)) continue;

            std::vector<const MemcpyBuffer*> srcBufs = {&hostBuf,  &peerBuf};
            std::vector<const MemcpyBuffer*> dstBufs = {&primBuf1, &primBuf2};
            std::vector<InitiatorType> types = {InitiatorType::CE, InitiatorType::SM};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bwHtoD .value(primId, peerId) = results[0];
            bwDtoD .value(primId, peerId) = results[1];
            bwTotal.value(primId, peerId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bwHtoD,  "CE stream (H->D): CPU -> GPU(row) bandwidth (GB/s)");
    output->addTestcaseResults(bwDtoD,  "SM stream (DtoD read): GPU(row) <- GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal, "total bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// host_to_device_sm_device_to_device_read_ce: concurrent H->D SM + DtoD read CE
// ---------------------------------------------------------------------------
void HostToDeviceSMDeviceToDeviceReadCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bwHtoD  (deviceCount, deviceCount, key + "_h2d_sm");
    PeerValueMatrix<double> bwDtoD  (deviceCount, deviceCount, key + "_d2d_read_ce");
    PeerValueMatrix<double> bwTotal (deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_DST_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int primId = 0; primId < deviceCount; primId++) {
        for (int peerId = 0; peerId < deviceCount; peerId++) {
            if (primId == peerId) continue;

            HostBuffer   hostBuf (size, primId);
            DeviceBuffer primBuf1(size, primId);
            DeviceBuffer peerBuf (size, peerId);
            DeviceBuffer primBuf2(size, primId);

            if (!primBuf2.enablePeerAcess(peerBuf)) continue;

            std::vector<const MemcpyBuffer*> srcBufs = {&hostBuf,  &peerBuf};
            std::vector<const MemcpyBuffer*> dstBufs = {&primBuf1, &primBuf2};
            std::vector<InitiatorType> types = {InitiatorType::SM, InitiatorType::CE};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bwHtoD .value(primId, peerId) = results[0];
            bwDtoD .value(primId, peerId) = results[1];
            bwTotal.value(primId, peerId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bwHtoD,  "SM stream (H->D): CPU -> GPU(row) bandwidth (GB/s)");
    output->addTestcaseResults(bwDtoD,  "CE stream (DtoD read): GPU(row) <- GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal, "total bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// host_to_device_sm_device_to_device_read_sm: concurrent H->D SM + DtoD read SM
// ---------------------------------------------------------------------------
void HostToDeviceSMDeviceToDeviceReadSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bwHtoD  (deviceCount, deviceCount, key + "_h2d_sm");
    PeerValueMatrix<double> bwDtoD  (deviceCount, deviceCount, key + "_d2d_read_sm");
    PeerValueMatrix<double> bwTotal (deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_DST_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int primId = 0; primId < deviceCount; primId++) {
        for (int peerId = 0; peerId < deviceCount; peerId++) {
            if (primId == peerId) continue;

            HostBuffer   hostBuf (size, primId);
            DeviceBuffer primBuf1(size, primId);
            DeviceBuffer peerBuf (size, peerId);
            DeviceBuffer primBuf2(size, primId);

            if (!primBuf2.enablePeerAcess(peerBuf)) continue;

            std::vector<const MemcpyBuffer*> srcBufs = {&hostBuf,  &peerBuf};
            std::vector<const MemcpyBuffer*> dstBufs = {&primBuf1, &primBuf2};
            std::vector<InitiatorType> types = {InitiatorType::SM, InitiatorType::SM};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bwHtoD .value(primId, peerId) = results[0];
            bwDtoD .value(primId, peerId) = results[1];
            bwTotal.value(primId, peerId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bwHtoD,  "SM stream (H->D): CPU -> GPU(row) bandwidth (GB/s)");
    output->addTestcaseResults(bwDtoD,  "SM stream (DtoD read): GPU(row) <- GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal, "total bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// host_to_device_ce_device_to_device_write_ce: concurrent H->D CE + DtoD write CE
// Row=primary(H->D dst and DtoD write src), Col=peer(DtoD write dst)
// ---------------------------------------------------------------------------
void HostToDeviceCEDeviceToDeviceWriteCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bwHtoD  (deviceCount, deviceCount, key + "_h2d_ce");
    PeerValueMatrix<double> bwDtoD  (deviceCount, deviceCount, key + "_d2d_write_ce");
    PeerValueMatrix<double> bwTotal (deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int primId = 0; primId < deviceCount; primId++) {
        for (int peerId = 0; peerId < deviceCount; peerId++) {
            if (primId == peerId) continue;

            HostBuffer   hostBuf (size, primId);  // H->D source
            DeviceBuffer primBuf1(size, primId);  // H->D destination
            DeviceBuffer primBuf2(size, primId);  // DtoD write source
            DeviceBuffer peerBuf (size, peerId);  // DtoD write destination

            if (!primBuf1.enablePeerAcess(peerBuf)) continue;

            // Stream 0 (CE): hostBuf  -> primBuf1 [H->D, PREFER_SRC_CONTEXT: host no ctx -> primId]
            // Stream 1 (CE): primBuf2 -> peerBuf  [DtoD write, PREFER_SRC_CONTEXT: primId]
            std::vector<const MemcpyBuffer*> srcBufs = {&hostBuf,  &primBuf2};
            std::vector<const MemcpyBuffer*> dstBufs = {&primBuf1, &peerBuf};
            std::vector<InitiatorType> types = {InitiatorType::CE, InitiatorType::CE};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bwHtoD .value(primId, peerId) = results[0];
            bwDtoD .value(primId, peerId) = results[1];
            bwTotal.value(primId, peerId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bwHtoD,  "CE stream (H->D): CPU -> GPU(row) bandwidth (GB/s)");
    output->addTestcaseResults(bwDtoD,  "CE stream (DtoD write): GPU(row) -> GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal, "total bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// host_to_device_ce_device_to_device_write_sm: concurrent H->D CE + DtoD write SM
// ---------------------------------------------------------------------------
void HostToDeviceCEDeviceToDeviceWriteSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bwHtoD  (deviceCount, deviceCount, key + "_h2d_ce");
    PeerValueMatrix<double> bwDtoD  (deviceCount, deviceCount, key + "_d2d_write_sm");
    PeerValueMatrix<double> bwTotal (deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int primId = 0; primId < deviceCount; primId++) {
        for (int peerId = 0; peerId < deviceCount; peerId++) {
            if (primId == peerId) continue;

            HostBuffer   hostBuf (size, primId);
            DeviceBuffer primBuf1(size, primId);
            DeviceBuffer primBuf2(size, primId);
            DeviceBuffer peerBuf (size, peerId);

            if (!primBuf1.enablePeerAcess(peerBuf)) continue;

            std::vector<const MemcpyBuffer*> srcBufs = {&hostBuf,  &primBuf2};
            std::vector<const MemcpyBuffer*> dstBufs = {&primBuf1, &peerBuf};
            std::vector<InitiatorType> types = {InitiatorType::CE, InitiatorType::SM};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bwHtoD .value(primId, peerId) = results[0];
            bwDtoD .value(primId, peerId) = results[1];
            bwTotal.value(primId, peerId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bwHtoD,  "CE stream (H->D): CPU -> GPU(row) bandwidth (GB/s)");
    output->addTestcaseResults(bwDtoD,  "SM stream (DtoD write): GPU(row) -> GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal, "total bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// host_to_device_sm_device_to_device_write_ce: concurrent H->D SM + DtoD write CE
// ---------------------------------------------------------------------------
void HostToDeviceSMDeviceToDeviceWriteCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bwHtoD  (deviceCount, deviceCount, key + "_h2d_sm");
    PeerValueMatrix<double> bwDtoD  (deviceCount, deviceCount, key + "_d2d_write_ce");
    PeerValueMatrix<double> bwTotal (deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int primId = 0; primId < deviceCount; primId++) {
        for (int peerId = 0; peerId < deviceCount; peerId++) {
            if (primId == peerId) continue;

            HostBuffer   hostBuf (size, primId);
            DeviceBuffer primBuf1(size, primId);
            DeviceBuffer primBuf2(size, primId);
            DeviceBuffer peerBuf (size, peerId);

            if (!primBuf1.enablePeerAcess(peerBuf)) continue;

            std::vector<const MemcpyBuffer*> srcBufs = {&hostBuf,  &primBuf2};
            std::vector<const MemcpyBuffer*> dstBufs = {&primBuf1, &peerBuf};
            std::vector<InitiatorType> types = {InitiatorType::SM, InitiatorType::CE};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bwHtoD .value(primId, peerId) = results[0];
            bwDtoD .value(primId, peerId) = results[1];
            bwTotal.value(primId, peerId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bwHtoD,  "SM stream (H->D): CPU -> GPU(row) bandwidth (GB/s)");
    output->addTestcaseResults(bwDtoD,  "CE stream (DtoD write): GPU(row) -> GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal, "total bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// host_to_device_sm_device_to_device_write_sm: concurrent H->D SM + DtoD write SM
// ---------------------------------------------------------------------------
void HostToDeviceSMDeviceToDeviceWriteSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bwHtoD  (deviceCount, deviceCount, key + "_h2d_sm");
    PeerValueMatrix<double> bwDtoD  (deviceCount, deviceCount, key + "_d2d_write_sm");
    PeerValueMatrix<double> bwTotal (deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int primId = 0; primId < deviceCount; primId++) {
        for (int peerId = 0; peerId < deviceCount; peerId++) {
            if (primId == peerId) continue;

            HostBuffer   hostBuf (size, primId);
            DeviceBuffer primBuf1(size, primId);
            DeviceBuffer primBuf2(size, primId);
            DeviceBuffer peerBuf (size, peerId);

            if (!primBuf1.enablePeerAcess(peerBuf)) continue;

            std::vector<const MemcpyBuffer*> srcBufs = {&hostBuf,  &primBuf2};
            std::vector<const MemcpyBuffer*> dstBufs = {&primBuf1, &peerBuf};
            std::vector<InitiatorType> types = {InitiatorType::SM, InitiatorType::SM};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bwHtoD .value(primId, peerId) = results[0];
            bwDtoD .value(primId, peerId) = results[1];
            bwTotal.value(primId, peerId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bwHtoD,  "SM stream (H->D): CPU -> GPU(row) bandwidth (GB/s)");
    output->addTestcaseResults(bwDtoD,  "SM stream (DtoD write): GPU(row) -> GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal, "total bandwidth (GB/s)");
}

