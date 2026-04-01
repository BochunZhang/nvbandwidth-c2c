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
// device_to_device_read_ce_ce: two CE read streams per peer pair (NVLink P2P)
// Row=primId (reads data, PREFER_DST_CONTEXT), Col=peerId (source of data)
// ---------------------------------------------------------------------------
void DeviceToDeviceReadCECE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (deviceCount, deviceCount, key + "_ce1");
    PeerValueMatrix<double> bw1  (deviceCount, deviceCount, key + "_ce2");
    PeerValueMatrix<double> bwTotal(deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_DST_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int primId = 0; primId < deviceCount; primId++) {
        for (int peerId = 0; peerId < deviceCount; peerId++) {
            if (primId == peerId) continue;

            DeviceBuffer primBuf1(size, primId), primBuf2(size, primId);
            DeviceBuffer peerBuf1(size, peerId), peerBuf2(size, peerId);

            if (!primBuf1.enablePeerAcess(peerBuf1) || !primBuf2.enablePeerAcess(peerBuf2)) continue;

            // Read: peerBuf -> primBuf, using primId context (PREFER_DST_CONTEXT)
            std::vector<const MemcpyBuffer*> srcBufs = {&peerBuf1, &peerBuf2};
            std::vector<const MemcpyBuffer*> dstBufs = {&primBuf1, &primBuf2};
            std::vector<InitiatorType> types = {InitiatorType::CE, InitiatorType::CE};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bw0   .value(primId, peerId) = results[0];
            bw1   .value(primId, peerId) = results[1];
            bwTotal.value(primId, peerId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bw0,    "CE stream0: GPU(row) <- GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bw1,    "CE stream1: GPU(row) <- GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal,"total:      GPU(row) <- GPU(col) bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// device_to_device_read_ce_sm: CE + SM read streams per peer pair (NVLink P2P)
// ---------------------------------------------------------------------------
void DeviceToDeviceReadCESM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (deviceCount, deviceCount, key + "_ce");
    PeerValueMatrix<double> bw1  (deviceCount, deviceCount, key + "_sm");
    PeerValueMatrix<double> bwTotal(deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_DST_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int primId = 0; primId < deviceCount; primId++) {
        for (int peerId = 0; peerId < deviceCount; peerId++) {
            if (primId == peerId) continue;

            DeviceBuffer primBuf1(size, primId), primBuf2(size, primId);
            DeviceBuffer peerBuf1(size, peerId), peerBuf2(size, peerId);

            if (!primBuf1.enablePeerAcess(peerBuf1) || !primBuf2.enablePeerAcess(peerBuf2)) continue;

            // Read: peerBuf -> primBuf, using primId context (PREFER_DST_CONTEXT)
            std::vector<const MemcpyBuffer*> srcBufs = {&peerBuf1, &peerBuf2};
            std::vector<const MemcpyBuffer*> dstBufs = {&primBuf1, &primBuf2};
            std::vector<InitiatorType> types = {InitiatorType::CE, InitiatorType::SM};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bw0   .value(primId, peerId) = results[0];
            bw1   .value(primId, peerId) = results[1];
            bwTotal.value(primId, peerId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bw0,    "CE stream: GPU(row) <- GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bw1,    "SM stream: GPU(row) <- GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal,"total:     GPU(row) <- GPU(col) bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// device_to_device_read_sm_ce: SM + CE read streams per peer pair (NVLink P2P)
// ---------------------------------------------------------------------------
void DeviceToDeviceReadSMCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (deviceCount, deviceCount, key + "_sm");
    PeerValueMatrix<double> bw1  (deviceCount, deviceCount, key + "_ce");
    PeerValueMatrix<double> bwTotal(deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_DST_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int primId = 0; primId < deviceCount; primId++) {
        for (int peerId = 0; peerId < deviceCount; peerId++) {
            if (primId == peerId) continue;

            DeviceBuffer primBuf1(size, primId), primBuf2(size, primId);
            DeviceBuffer peerBuf1(size, peerId), peerBuf2(size, peerId);

            if (!primBuf1.enablePeerAcess(peerBuf1) || !primBuf2.enablePeerAcess(peerBuf2)) continue;

            // Read: peerBuf -> primBuf, using primId context (PREFER_DST_CONTEXT)
            std::vector<const MemcpyBuffer*> srcBufs = {&peerBuf1, &peerBuf2};
            std::vector<const MemcpyBuffer*> dstBufs = {&primBuf1, &primBuf2};
            std::vector<InitiatorType> types = {InitiatorType::SM, InitiatorType::CE};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bw0   .value(primId, peerId) = results[0];
            bw1   .value(primId, peerId) = results[1];
            bwTotal.value(primId, peerId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bw0,    "SM stream: GPU(row) <- GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bw1,    "CE stream: GPU(row) <- GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal,"total:     GPU(row) <- GPU(col) bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// device_to_device_read_sm_sm: two SM read streams per peer pair (NVLink P2P)
// ---------------------------------------------------------------------------
void DeviceToDeviceReadSMSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (deviceCount, deviceCount, key + "_sm1");
    PeerValueMatrix<double> bw1  (deviceCount, deviceCount, key + "_sm2");
    PeerValueMatrix<double> bwTotal(deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_DST_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int primId = 0; primId < deviceCount; primId++) {
        for (int peerId = 0; peerId < deviceCount; peerId++) {
            if (primId == peerId) continue;

            DeviceBuffer primBuf1(size, primId), primBuf2(size, primId);
            DeviceBuffer peerBuf1(size, peerId), peerBuf2(size, peerId);

            if (!primBuf1.enablePeerAcess(peerBuf1) || !primBuf2.enablePeerAcess(peerBuf2)) continue;

            // Read: peerBuf -> primBuf, using primId context (PREFER_DST_CONTEXT)
            std::vector<const MemcpyBuffer*> srcBufs = {&peerBuf1, &peerBuf2};
            std::vector<const MemcpyBuffer*> dstBufs = {&primBuf1, &primBuf2};
            std::vector<InitiatorType> types = {InitiatorType::SM, InitiatorType::SM};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bw0   .value(primId, peerId) = results[0];
            bw1   .value(primId, peerId) = results[1];
            bwTotal.value(primId, peerId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bw0,    "SM stream0: GPU(row) <- GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bw1,    "SM stream1: GPU(row) <- GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal,"total:      GPU(row) <- GPU(col) bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// device_to_device_write_ce_ce: two CE write streams per peer pair (NVLink P2P)
// Row=primId (writes data, PREFER_SRC_CONTEXT), Col=peerId (destination)
// ---------------------------------------------------------------------------
void DeviceToDeviceWriteCECE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (deviceCount, deviceCount, key + "_ce1");
    PeerValueMatrix<double> bw1  (deviceCount, deviceCount, key + "_ce2");
    PeerValueMatrix<double> bwTotal(deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int primId = 0; primId < deviceCount; primId++) {
        for (int peerId = 0; peerId < deviceCount; peerId++) {
            if (primId == peerId) continue;

            DeviceBuffer primBuf1(size, primId), primBuf2(size, primId);
            DeviceBuffer peerBuf1(size, peerId), peerBuf2(size, peerId);

            if (!primBuf1.enablePeerAcess(peerBuf1) || !primBuf2.enablePeerAcess(peerBuf2)) continue;

            // Write: primBuf -> peerBuf, using primId context (PREFER_SRC_CONTEXT)
            std::vector<const MemcpyBuffer*> srcBufs = {&primBuf1, &primBuf2};
            std::vector<const MemcpyBuffer*> dstBufs = {&peerBuf1, &peerBuf2};
            std::vector<InitiatorType> types = {InitiatorType::CE, InitiatorType::CE};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bw0   .value(primId, peerId) = results[0];
            bw1   .value(primId, peerId) = results[1];
            bwTotal.value(primId, peerId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bw0,    "CE stream0: GPU(row) -> GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bw1,    "CE stream1: GPU(row) -> GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal,"total:      GPU(row) -> GPU(col) bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// device_to_device_write_ce_sm: CE + SM write streams per peer pair (NVLink P2P)
// ---------------------------------------------------------------------------
void DeviceToDeviceWriteCESM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (deviceCount, deviceCount, key + "_ce");
    PeerValueMatrix<double> bw1  (deviceCount, deviceCount, key + "_sm");
    PeerValueMatrix<double> bwTotal(deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int primId = 0; primId < deviceCount; primId++) {
        for (int peerId = 0; peerId < deviceCount; peerId++) {
            if (primId == peerId) continue;

            DeviceBuffer primBuf1(size, primId), primBuf2(size, primId);
            DeviceBuffer peerBuf1(size, peerId), peerBuf2(size, peerId);

            if (!primBuf1.enablePeerAcess(peerBuf1) || !primBuf2.enablePeerAcess(peerBuf2)) continue;

            // Write: primBuf -> peerBuf, using primId context (PREFER_SRC_CONTEXT)
            std::vector<const MemcpyBuffer*> srcBufs = {&primBuf1, &primBuf2};
            std::vector<const MemcpyBuffer*> dstBufs = {&peerBuf1, &peerBuf2};
            std::vector<InitiatorType> types = {InitiatorType::CE, InitiatorType::SM};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bw0   .value(primId, peerId) = results[0];
            bw1   .value(primId, peerId) = results[1];
            bwTotal.value(primId, peerId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bw0,    "CE stream: GPU(row) -> GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bw1,    "SM stream: GPU(row) -> GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal,"total:     GPU(row) -> GPU(col) bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// device_to_device_write_sm_ce: SM + CE write streams per peer pair (NVLink P2P)
// ---------------------------------------------------------------------------
void DeviceToDeviceWriteSMCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (deviceCount, deviceCount, key + "_sm");
    PeerValueMatrix<double> bw1  (deviceCount, deviceCount, key + "_ce");
    PeerValueMatrix<double> bwTotal(deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int primId = 0; primId < deviceCount; primId++) {
        for (int peerId = 0; peerId < deviceCount; peerId++) {
            if (primId == peerId) continue;

            DeviceBuffer primBuf1(size, primId), primBuf2(size, primId);
            DeviceBuffer peerBuf1(size, peerId), peerBuf2(size, peerId);

            if (!primBuf1.enablePeerAcess(peerBuf1) || !primBuf2.enablePeerAcess(peerBuf2)) continue;

            // Write: primBuf -> peerBuf, using primId context (PREFER_SRC_CONTEXT)
            std::vector<const MemcpyBuffer*> srcBufs = {&primBuf1, &primBuf2};
            std::vector<const MemcpyBuffer*> dstBufs = {&peerBuf1, &peerBuf2};
            std::vector<InitiatorType> types = {InitiatorType::SM, InitiatorType::CE};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bw0   .value(primId, peerId) = results[0];
            bw1   .value(primId, peerId) = results[1];
            bwTotal.value(primId, peerId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bw0,    "SM stream: GPU(row) -> GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bw1,    "CE stream: GPU(row) -> GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal,"total:     GPU(row) -> GPU(col) bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// device_to_device_write_sm_sm: two SM write streams per peer pair (NVLink P2P)
// ---------------------------------------------------------------------------
void DeviceToDeviceWriteSMSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bw0  (deviceCount, deviceCount, key + "_sm1");
    PeerValueMatrix<double> bw1  (deviceCount, deviceCount, key + "_sm2");
    PeerValueMatrix<double> bwTotal(deviceCount, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int primId = 0; primId < deviceCount; primId++) {
        for (int peerId = 0; peerId < deviceCount; peerId++) {
            if (primId == peerId) continue;

            DeviceBuffer primBuf1(size, primId), primBuf2(size, primId);
            DeviceBuffer peerBuf1(size, peerId), peerBuf2(size, peerId);

            if (!primBuf1.enablePeerAcess(peerBuf1) || !primBuf2.enablePeerAcess(peerBuf2)) continue;

            // Write: primBuf -> peerBuf, using primId context (PREFER_SRC_CONTEXT)
            std::vector<const MemcpyBuffer*> srcBufs = {&primBuf1, &primBuf2};
            std::vector<const MemcpyBuffer*> dstBufs = {&peerBuf1, &peerBuf2};
            std::vector<InitiatorType> types = {InitiatorType::SM, InitiatorType::SM};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types);
            bw0   .value(primId, peerId) = results[0];
            bw1   .value(primId, peerId) = results[1];
            bwTotal.value(primId, peerId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bw0,    "SM stream0: GPU(row) -> GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bw1,    "SM stream1: GPU(row) -> GPU(col) bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal,"total:      GPU(row) -> GPU(col) bandwidth (GB/s)");
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

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    // D->H uses src context (device is source); DtoD read uses dst context (local dst pulls from peer).
    const std::vector<ContextPreference> ctxPrefs = {PREFER_SRC_CONTEXT, PREFER_DST_CONTEXT};

    for (int primId = 0; primId < deviceCount; primId++) {
        for (int peerId = 0; peerId < deviceCount; peerId++) {
            if (primId == peerId) continue;

            DeviceBuffer primBuf1(size, primId);  // D->H source
            DeviceBuffer primBuf2(size, primId);  // DtoD read destination
            DeviceBuffer peerBuf (size, peerId);  // DtoD read source
            HostBuffer   hostBuf (size, primId);  // D->H destination

            if (!primBuf2.enablePeerAcess(peerBuf)) continue;

            // Stream 0 (CE): primBuf1 -> hostBuf   [D->H, PREFER_SRC_CONTEXT: primId]
            // Stream 1 (CE): peerBuf  -> primBuf2  [DtoD read, PREFER_DST_CONTEXT: primId]
            std::vector<const MemcpyBuffer*> srcBufs = {&primBuf1, &peerBuf};
            std::vector<const MemcpyBuffer*> dstBufs = {&hostBuf,  &primBuf2};
            std::vector<InitiatorType> types = {InitiatorType::CE, InitiatorType::CE};

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types, ctxPrefs);
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

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    const std::vector<ContextPreference> ctxPrefs = {PREFER_SRC_CONTEXT, PREFER_DST_CONTEXT};

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

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types, ctxPrefs);
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

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    const std::vector<ContextPreference> ctxPrefs = {PREFER_SRC_CONTEXT, PREFER_DST_CONTEXT};

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

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types, ctxPrefs);
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

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    const std::vector<ContextPreference> ctxPrefs = {PREFER_SRC_CONTEXT, PREFER_DST_CONTEXT};

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

            auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types, ctxPrefs);
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

// ---------------------------------------------------------------------------
// four_gpu_concurrent_ce
// Stream 0 (CE): host  -> GPU0   [H->D,        PREFER_DST_CONTEXT: GPU0]
// Stream 1 (CE): GPU0  -> host   [D->H,        PREFER_SRC_CONTEXT: GPU0]
// Stream 2 (CE): GPU1  -> GPU0   [DtoD read,   PREFER_DST_CONTEXT: GPU0]
// Stream 3 (CE): GPU0  -> GPU1   [DtoD write,  PREFER_SRC_CONTEXT: GPU0]
// Stream 4 (CE): GPU1  -> GPU2   [DtoD write,  PREFER_SRC_CONTEXT: GPU1]
// Stream 5 (CE): GPU1  -> GPU3   [DtoD write,  PREFER_SRC_CONTEXT: GPU1]
// ---------------------------------------------------------------------------
void ConcurrentCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bwS0(1, 1, key + "_h2d_gpu0");
    PeerValueMatrix<double> bwS1(1, 1, key + "_d2h_gpu0");
    PeerValueMatrix<double> bwS2(1, 1, key + "_read_gpu1_to_gpu0");
    PeerValueMatrix<double> bwS3(1, 1, key + "_write_gpu0_to_gpu1");
    PeerValueMatrix<double> bwS4(1, 1, key + "_write_gpu1_to_gpu2");
    PeerValueMatrix<double> bwS5(1, 1, key + "_write_gpu1_to_gpu3");
    PeerValueMatrix<double> bwS6(1, 1, key + "_read_gpu2_to_gpu1");
    PeerValueMatrix<double> bwS7(1, 1, key + "_read_gpu3_to_gpu1");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    // GPU0 stream buffers
    HostBuffer   s0hostBuf(size, 0);   // s0 src: H->GPU0
    DeviceBuffer s0gpu0Buf(size, 0);   // s0 dst *
    DeviceBuffer s1gpu0Buf(size, 0);   // s1 src: GPU0->H *
    HostBuffer   s1hostBuf(size, 0);   // s1 dst
    DeviceBuffer s2gpu1Buf(size, 1);   // s2 src: GPU1->GPU0 read
    DeviceBuffer s2gpu0Buf(size, 0);   // s2 dst *
    DeviceBuffer s3gpu0Buf(size, 0);   // s3 src: GPU0->GPU1 write *
    DeviceBuffer s3gpu1Buf(size, 1);   // s3 dst
    // GPU1 stream buffers
    DeviceBuffer s4gpu1Buf(size, 1);   // s4 src: GPU1->GPU2 write *
    DeviceBuffer s4gpu2Buf(size, 2);   // s4 dst
    DeviceBuffer s5gpu1Buf(size, 1);   // s5 src: GPU1->GPU3 write *
    DeviceBuffer s5gpu3Buf(size, 3);   // s5 dst
    DeviceBuffer s6gpu2Buf(size, 2);   // s6 src: GPU2->GPU1 read
    DeviceBuffer s6gpu1Buf(size, 1);   // s6 dst * 
    DeviceBuffer s7gpu3Buf(size, 3);   // s7 src: GPU3->GPU1 read
    DeviceBuffer s7gpu1Buf(size, 1);   // s7 dst *

    // Enable peer access for each required device pair
    if (!s2gpu1Buf.enablePeerAcess(s2gpu0Buf)) {
        output->recordWarning("concurrent_ce: GPU0<->GPU1 peer access not available, skipping.");
        return;
    }
    if (!s3gpu0Buf.enablePeerAcess(s3gpu1Buf)) {
        output->recordWarning("concurrent_ce: GPU0<->GPU1 peer access not available, skipping.");
        return;
    }
    if (!s4gpu1Buf.enablePeerAcess(s4gpu2Buf)) {
        output->recordWarning("concurrent_ce: GPU1<->GPU2 peer access not available, skipping.");
        return;
    }

    if (!s5gpu1Buf.enablePeerAcess(s5gpu3Buf)) {
        output->recordWarning("concurrent_ce: GPU1<->GPU3 peer access not available, skipping.");
        return;
    }

    if (!s6gpu1Buf.enablePeerAcess(s6gpu2Buf)) {
        output->recordWarning("concurrent_ce: GPU1<->GPU2 peer access not available, skipping.");
        return;
    }

    if (!s7gpu1Buf.enablePeerAcess(s7gpu3Buf)) {
        output->recordWarning("concurrent_ce: GPU1<->GPU3 peer access not available, skipping.");
        return;
    }

    // Per-stream context preferences
    const std::vector<ContextPreference> ctxPrefs = {
        PREFER_DST_CONTEXT,   // s0: H->GPU0, host has no ctx -> GPU0
        PREFER_SRC_CONTEXT,   // s1: GPU0->H, src=GPU0 -> GPU0
        PREFER_DST_CONTEXT,   // s2: GPU1->GPU0 read, dst=GPU0 -> GPU0
        PREFER_SRC_CONTEXT,   // s3: GPU0->GPU1 write, src=GPU0 -> GPU0
        PREFER_SRC_CONTEXT,   // s4: GPU1->GPU2 write, src=GPU1 -> GPU1
        PREFER_SRC_CONTEXT,   // s5: GPU1->GPU3 write, src=GPU1 -> GPU1
        PREFER_DST_CONTEXT,   // s6: GPU2->GPU1 read, dst=GPU1 -> GPU1
        PREFER_DST_CONTEXT,   // s7: GPU3->GPU1 read, dst=GPU1 -> GPU1
    };

    std::vector<const MemcpyBuffer*> srcBufs = {&s0hostBuf, &s1gpu0Buf, &s2gpu1Buf, &s3gpu0Buf, &s4gpu1Buf, &s5gpu1Buf, &s6gpu2Buf, &s7gpu3Buf};
    std::vector<const MemcpyBuffer*> dstBufs = {&s0gpu0Buf, &s1hostBuf, &s2gpu0Buf, &s3gpu1Buf, &s4gpu2Buf, &s5gpu3Buf, &s6gpu1Buf, &s7gpu1Buf};
    std::vector<InitiatorType> types = {
        InitiatorType::CE, InitiatorType::CE, InitiatorType::CE, InitiatorType::CE,
        InitiatorType::CE, InitiatorType::CE, InitiatorType::CE, InitiatorType::CE,
    };

    auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types, ctxPrefs);

    bwS0   .value(0, 0) = results[0];
    bwS1   .value(0, 0) = results[1];
    bwS2   .value(0, 0) = results[2];
    bwS3   .value(0, 0) = results[3];
    bwS4   .value(0, 0) = results[4];
    bwS5   .value(0, 0) = results[5];
    bwS6   .value(0, 0) = results[6];
    bwS7   .value(0, 0) = results[7];

    output->addTestcaseResults(bwS0,    "CE stream0 (H->D):          CPU -> GPU0 bandwidth (GB/s)");
    output->addTestcaseResults(bwS1,    "CE stream1 (D->H):          GPU0 -> CPU bandwidth (GB/s)");
    output->addTestcaseResults(bwS2,    "CE stream2 (DtoD read):     GPU1 -> GPU0 bandwidth (GB/s)");
    output->addTestcaseResults(bwS3,    "CE stream3 (DtoD write):    GPU0 -> GPU1 bandwidth (GB/s)");
    output->addTestcaseResults(bwS4,    "CE stream4 (DtoD write):    GPU1 -> GPU2 bandwidth (GB/s)");
    output->addTestcaseResults(bwS5,    "CE stream5 (DtoD write):    GPU1 -> GPU3 bandwidth (GB/s)");
    output->addTestcaseResults(bwS6,    "CE stream6 (DtoD read):     GPU2 -> GPU1 bandwidth (GB/s)");
    output->addTestcaseResults(bwS7,    "CE stream7 (DtoD read):     GPU3 -> GPU1 bandwidth (GB/s)");
}

// ---------------------------------------------------------------------------
// host_to_device_bidirectional_memcpy helper macro (avoid code duplication)
// Stream 0: H->D using initiator type h2d_type (PREFER_DST_CONTEXT -> device)
// Stream 1: D->H using initiator type d2h_type (PREFER_SRC_CONTEXT -> device)
// Iterates over all devices; both streams share the same device context.
// ---------------------------------------------------------------------------
static void runHostDeviceBidir(unsigned long long size, unsigned long long loopCount,
                                InitiatorType h2dType, InitiatorType d2hType,
                                const std::string &key) {
    PeerValueMatrix<double> bwH2D(1, deviceCount, key + "_h2d");
    PeerValueMatrix<double> bwD2H(1, deviceCount, key + "_d2h");
    PeerValueMatrix<double> bwTotal(1, deviceCount, key + "_total");

    std::vector<MemcpyInitiator*> initiators(static_cast<size_t>(InitiatorType::INITIATOR_NUM), nullptr);
    initiators[static_cast<size_t>(InitiatorType::CE)] = new MemcpyInitiatorCE();
    initiators[static_cast<size_t>(InitiatorType::SM)] = new MemcpyInitiatorSM();

    CustomMemcpyOperation memcpyInstance(loopCount, initiators, PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    // Per-stream context preferences: H->D uses dst (device), D->H uses src (device)
    const std::vector<ContextPreference> ctxPrefs = {PREFER_DST_CONTEXT, PREFER_SRC_CONTEXT};

    for (int devId = 0; devId < deviceCount; devId++) {
        HostBuffer   hostH2D(size, devId);   // s0 src
        DeviceBuffer devH2D (size, devId);   // s0 dst
        DeviceBuffer devD2H (size, devId);   // s1 src
        HostBuffer   hostD2H(size, devId);   // s1 dst

        std::vector<const MemcpyBuffer*> srcBufs = {&hostH2D, &devD2H};
        std::vector<const MemcpyBuffer*> dstBufs = {&devH2D,  &hostD2H};
        std::vector<InitiatorType> types = {h2dType, d2hType};

        auto results = memcpyInstance.doMemcpyVector(srcBufs, dstBufs, types, ctxPrefs);

        bwH2D .value(0, devId) = results[0];
        bwD2H .value(0, devId) = results[1];
        bwTotal.value(0, devId) = results[0] + results[1];
    }

    output->addTestcaseResults(bwH2D,   "H->D stream: CPU -> GPU bandwidth (GB/s)");
    output->addTestcaseResults(bwD2H,   "D->H stream: GPU -> CPU bandwidth (GB/s)");
    output->addTestcaseResults(bwTotal, "total bandwidth (GB/s)");
}

void HostDeviceBidirCECE::run(unsigned long long size, unsigned long long loopCount) {
    runHostDeviceBidir(size, loopCount, InitiatorType::CE, InitiatorType::CE, key);
}

void HostDeviceBidirCESM::run(unsigned long long size, unsigned long long loopCount) {
    runHostDeviceBidir(size, loopCount, InitiatorType::CE, InitiatorType::SM, key);
}

void HostDeviceBidirSMCE::run(unsigned long long size, unsigned long long loopCount) {
    runHostDeviceBidir(size, loopCount, InitiatorType::SM, InitiatorType::CE, key);
}

void HostDeviceBidirSMSM::run(unsigned long long size, unsigned long long loopCount) {
    runHostDeviceBidir(size, loopCount, InitiatorType::SM, InitiatorType::SM, key);
}
