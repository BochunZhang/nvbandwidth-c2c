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

#include <cuda.h>

#include "common.h"
#include "output.h"
#include "testcase.h"
#include "memcpy.h"

void HostToDeviceCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE());

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostBuffer hostBuffer(size, deviceId);
        DeviceBuffer deviceBuffer(size, deviceId);

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(hostBuffer, deviceBuffer);
    }

    output->addTestcaseResults(bandwidthValues, "memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s)");
}

void DeviceToHostCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE());

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostBuffer hostBuffer(size, deviceId);
        DeviceBuffer deviceBuffer(size, deviceId);

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(deviceBuffer, hostBuffer);
    }

    output->addTestcaseResults(bandwidthValues, "memcpy CE CPU(row) <- GPU(column) bandwidth (GB/s)");
}

void HostToDeviceBidirCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE());

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        // Double the size of the interference copy to ensure it interferes correctly
        HostBuffer host1(size, deviceId), host2(size * 2, deviceId);
        DeviceBuffer dev1(size, deviceId), dev2(size * 2, deviceId);

        std::vector<const MemcpyBuffer*> srcBuffers = {&host1, &dev2};
        std::vector<const MemcpyBuffer*> dstBuffers = {&dev1, &host2};

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(srcBuffers, dstBuffers);
    }

    output->addTestcaseResults(bandwidthValues, "memcpy CE CPU(row) <-> GPU(column) bandwidth (GB/s)");
}

void DeviceToHostBidirCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE());

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        // Double the size of the interference copy to ensure it interferes correctly
        HostBuffer host1(size, deviceId), host2(size * 2, deviceId);
        DeviceBuffer dev1(size, deviceId), dev2(size * 2, deviceId);

        std::vector<const MemcpyBuffer*> srcBuffers = {&dev1, &host2};
        std::vector<const MemcpyBuffer*> dstBuffers = {&host1, &dev2};

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(srcBuffers, dstBuffers);
    }

    output->addTestcaseResults(bandwidthValues, "memcpy CE CPU(row) <-> GPU(column) bandwidth (GB/s)");
}

// DtoD Read test - copy from dst to src (backwards) using src contxt
void DeviceToDeviceReadCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE(), PREFER_DST_CONTEXT);

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            DeviceBuffer srcBuffer(size, srcDeviceId);
            DeviceBuffer peerBuffer(size, peerDeviceId);

            if (!srcBuffer.enablePeerAcess(peerBuffer)) {
                continue;
            }

            // swap src and peer nodes, but use srcBuffers (the copy's destination) context
            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(peerBuffer, srcBuffer);
        }
    }

    output->addTestcaseResults(bandwidthValues, "memcpy CE GPU(row) -> GPU(column) bandwidth (GB/s)");
}

// DtoD Write test - copy from src to dst using src context
void DeviceToDeviceWriteCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE());

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            DeviceBuffer srcBuffer(size, srcDeviceId);
            DeviceBuffer peerBuffer(size, peerDeviceId);

            if (!srcBuffer.enablePeerAcess(peerBuffer)) {
                continue;
            }

            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(srcBuffer, peerBuffer);
        }
    }

    output->addTestcaseResults(bandwidthValues, "memcpy CE GPU(row) <- GPU(column) bandwidth (GB/s)");
}

// DtoD Bidir Read test - copy from dst to src (backwards) using src contxt
void DeviceToDeviceBidirReadCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValuesRead1(deviceCount, deviceCount, key + "_read1");
    PeerValueMatrix<double> bandwidthValuesRead2(deviceCount, deviceCount, key + "_read2");
    PeerValueMatrix<double> bandwidthValuesTotal(deviceCount, deviceCount, key + "_total");
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE(), PREFER_DST_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            // Double the size of the interference copy to ensure it interferes correctly
            DeviceBuffer src1(size, srcDeviceId), src2(size, srcDeviceId);
            DeviceBuffer peer1(size, peerDeviceId), peer2(size, peerDeviceId);

            if (!src1.enablePeerAcess(peer1)) {
                continue;
            }

            // swap src and peer nodes, but use srcBuffers (the copy's destination) context
            std::vector<const MemcpyBuffer*> srcBuffers = {&peer1, &src2};
            std::vector<const MemcpyBuffer*> peerBuffers = {&src1, &peer2};

            auto results = memcpyInstance.doMemcpyVector(srcBuffers, peerBuffers);
            bandwidthValuesRead1.value(srcDeviceId, peerDeviceId) = results[0];
            bandwidthValuesRead2.value(srcDeviceId, peerDeviceId) = results[1];
            bandwidthValuesTotal.value(srcDeviceId, peerDeviceId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bandwidthValuesRead1, "memcpy CE GPU(row) <-> GPU(column) Read1 bandwidth (GB/s)");
    output->addTestcaseResults(bandwidthValuesRead2, "memcpy CE GPU(row) <-> GPU(column) Read2 bandwidth (GB/s)");
    output->addTestcaseResults(bandwidthValuesTotal, "memcpy CE GPU(row) <-> GPU(column) Total bandwidth (GB/s)");
}

// DtoD Bidir Write test - copy from src to dst using src context
void DeviceToDeviceBidirWriteCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValuesWrite1(deviceCount, deviceCount, key + "_write1");
    PeerValueMatrix<double> bandwidthValuesWrite2(deviceCount, deviceCount, key + "_write2");
    PeerValueMatrix<double> bandwidthValuesTotal(deviceCount, deviceCount, key + "_total");
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE(), PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            // Double the size of the interference copy to ensure it interferes correctly
            DeviceBuffer src1(size, srcDeviceId), src2(size, srcDeviceId);
            DeviceBuffer peer1(size, peerDeviceId), peer2(size, peerDeviceId);

            if (!src1.enablePeerAcess(peer1)) {
                continue;
            }

            // swap src and peer nodes, but use srcBuffers (the copy's destination) context
            std::vector<const MemcpyBuffer*> srcBuffers = {&peer1, &src2};
            std::vector<const MemcpyBuffer*> peerBuffers = {&src1, &peer2};

            auto results = memcpyInstance.doMemcpyVector(srcBuffers, peerBuffers);
            bandwidthValuesWrite1.value(srcDeviceId, peerDeviceId) = results[0];
            bandwidthValuesWrite2.value(srcDeviceId, peerDeviceId) = results[1];
            bandwidthValuesTotal.value(srcDeviceId, peerDeviceId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bandwidthValuesWrite1, "memcpy CE GPU(row) <-> GPU(column) Write1 bandwidth (GB/s)");
    output->addTestcaseResults(bandwidthValuesWrite2, "memcpy CE GPU(row) <-> GPU(column) Write2 bandwidth (GB/s)");
    output->addTestcaseResults(bandwidthValuesTotal, "memcpy CE GPU(row) <-> GPU(column) Total bandwidth (GB/s)");
}

void DeviceLocalCopy::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE());

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        DeviceBuffer deviceBuffer1(size, deviceId);
        DeviceBuffer deviceBuffer2(size, deviceId);

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(deviceBuffer2, deviceBuffer1);
    }

    output->addTestcaseResults(bandwidthValues, "memcpy local GPU(column) bandwidth (GB/s)");
}

void AllToHostCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE());

    allHostHelper(size, memcpyInstance, bandwidthValues, false);

    output->addTestcaseResults(bandwidthValues, "memcpy CE CPU(row) <- GPU(column) bandwidth (GB/s)");
}

void AllToHostBidirCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE());

    allHostBidirHelper(size, memcpyInstance, bandwidthValues, false);

    output->addTestcaseResults(bandwidthValues, "memcpy CE CPU(row) <-> GPU(column) bandwidth (GB/s)");
}

void HostToAllCE::run(unsigned long long size, unsigned long long loopCount) {
    VERBOSE << "\n=== HostToAllCE Test Configuration ===" << std::endl;
    VERBOSE << "Total GPUs participating: " << deviceCount << std::endl;
    VERBOSE << "Buffer size per transfer: " << (size / _MiB) << " MiB (" << size << " bytes)" << std::endl;
    VERBOSE << "Loop count (inner iterations): " << loopCount << std::endl;
    VERBOSE << "Average loop count (test repetitions): " << averageLoopCount << std::endl;
    VERBOSE << "Test pattern: CPU -> GPU[i] with interference from CPU -> GPU[j] (j != i)" << std::endl;
    VERBOSE << "BandwidthValue::USE_FIRST_BW" << std::endl;
    VERBOSE << "====================================\n" << std::endl;

    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE());

    allHostHelper(size, memcpyInstance, bandwidthValues, true);

    output->addTestcaseResults(bandwidthValues, "memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s)");
}

void HostToAllBidirCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE());

    allHostBidirHelper(size, memcpyInstance, bandwidthValues, true);

    output->addTestcaseResults(bandwidthValues, "memcpy CE CPU(row) <-> GPU(column) bandwidth (GB/s)");
}

void HostToAnyCE::run(unsigned long long size, unsigned long long loopCount) {
    // Use configured GPU IDs, or default to all GPUs
    std::vector<int> gpuIds = hostToAnyGpuIds;
    if (gpuIds.empty()) {
        for (int i = 0; i < deviceCount; i++) {
            gpuIds.push_back(i);
        }
    }

    int streamCount = hostToAnyStreamCount;

    VERBOSE << "\n=== HostToAnyCE Test Configuration ===" << std::endl;
    VERBOSE << "GPU IDs: ";
    for (int gpuId : gpuIds) {
        VERBOSE << gpuId << " ";
    }
    VERBOSE << std::endl;
    VERBOSE << "Streams per GPU: " << streamCount << std::endl;
    VERBOSE << "Buffer size: " << (size / _MiB) << " MiB" << std::endl;
    VERBOSE << "Loop count (inner iterations): " << loopCount << std::endl;
    VERBOSE << "Average loop count (test repetitions): " << averageLoopCount << std::endl;
    VERBOSE << "====================================\n" << std::endl;

    // Execute test
    PeerValueMatrix<double> bandwidthValues(1, gpuIds.size(), key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE(), PREFER_SRC_CONTEXT, MemcpyOperation::CONCURRENT_BW);

    anyHostHelper(size, memcpyInstance, gpuIds, streamCount, bandwidthValues, true);

    output->addTestcaseResults(bandwidthValues,
        "memcpy CE Host->GPU bandwidth (GB/s)\n  Columns: Per-stream BW + Aggregate");
}

void AnyToHostCE::run(unsigned long long size, unsigned long long loopCount) {
    VERBOSE << "\n=== AnyToHostCE Test Configuration ===" << std::endl;

    // Use configured GPU IDs, or default to all GPUs
    std::vector<int> gpuIds = hostToAnyGpuIds;
    if (gpuIds.empty()) {
        for (int i = 0; i < deviceCount; i++) {
            gpuIds.push_back(i);
        }
    }

    int streamCount = hostToAnyStreamCount;

    VERBOSE << "GPU IDs: ";
    for (int gpuId : gpuIds) {
        VERBOSE << gpuId << " ";
    }
    VERBOSE << std::endl;
    VERBOSE << "Streams per GPU: " << streamCount << std::endl;
    VERBOSE << "Buffer size: " << (size / _MiB) << " MiB" << std::endl;
    VERBOSE << "Loop count: " << loopCount << std::endl;
    VERBOSE << "====================================\n" << std::endl;

    // Execute test
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE(), PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);
    std::vector<double> results;
    anyHostHelper(size, memcpyInstance, gpuIds, streamCount, results, false);

    // Calculate statistics
    double totalBandwidth = 0.0;
    for (double bw : results) {
        totalBandwidth += bw;
    }

    // Print results
    OUTPUT << "\n=== AnyToHostCE Results ===" << std::endl;
    OUTPUT << "Direction: Device -> Host" << std::endl;
    OUTPUT << "Total streams: " << results.size() << " (" << gpuIds.size() << " GPUs × " << streamCount << " streams)" << std::endl;
    OUTPUT << std::endl;

    OUTPUT << std::setw(10) << "Stream" << std::setw(10) << "GPU"
           << std::setw(15) << "Bandwidth" << std::endl;

    int streamIdx = 0;
    for (size_t gpuIdx = 0; gpuIdx < gpuIds.size(); gpuIdx++) {
        for (int s = 0; s < streamCount; s++) {
            OUTPUT << std::setw(10) << streamIdx
                   << std::setw(10) << gpuIds[gpuIdx]
                   << std::setw(15) << std::fixed << std::setprecision(2)
                   << results[streamIdx] << " GB/s" << std::endl;
            streamIdx++;
        }
    }

    OUTPUT << std::endl;
    OUTPUT << "Total Aggregate Bandwidth: " << std::fixed << std::setprecision(2)
           << totalBandwidth << " GB/s" << std::endl;

    // Store results in a matrix for output system
    PeerValueMatrix<double> bandwidthValues(1, results.size() + 1, key);
    for (size_t i = 0; i < results.size(); i++) {
        bandwidthValues.value(0, i) = results[i];
    }
    bandwidthValues.value(0, results.size()) = totalBandwidth;

    output->addTestcaseResults(bandwidthValues,
        "memcpy CE GPU->Host bandwidth (GB/s)\n  Columns: Per-stream BW + Aggregate");
}

// Write test - copy from src to dst using src context
void AllToOneWriteCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE(), PREFER_SRC_CONTEXT, MemcpyOperation::TOTAL_BW);
    allToOneHelper(size, memcpyInstance, bandwidthValues, false);

    output->addTestcaseResults(bandwidthValues, "memcpy CE GPU(row) <- GPU(column) bandwidth (GB/s)");
}

// Read test - copy from dst to src (backwards) using src contxt
void AllToOneReadCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE(), PREFER_DST_CONTEXT, MemcpyOperation::TOTAL_BW);
    allToOneHelper(size, memcpyInstance, bandwidthValues, true);

    output->addTestcaseResults(bandwidthValues, "memcpy CE GPU(row) -> GPU(column) bandwidth (GB/s)");
}

// Write test - copy from src to dst using src context
void OneToAllWriteCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE(), PREFER_SRC_CONTEXT, MemcpyOperation::TOTAL_BW);
    oneToAllHelper(size, memcpyInstance, bandwidthValues, false);

    output->addTestcaseResults(bandwidthValues, "memcpy CE GPU(row) -> GPU(column) bandwidth (GB/s)");
}

// Read test - copy from dst to src (backwards) using src contxt
void OneToAllReadCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE(), PREFER_DST_CONTEXT, MemcpyOperation::TOTAL_BW);
    oneToAllHelper(size, memcpyInstance, bandwidthValues, true);

    output->addTestcaseResults(bandwidthValues, "memcpy CE GPU(row) <- GPU(column) bandwidth (GB/s)");
}
