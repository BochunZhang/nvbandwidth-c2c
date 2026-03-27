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

#include "testcase.h"
#include "kernels.cuh"
#include "memcpy.h"
#include "common.h"
#include "output.h"

void HostDeviceLatencySM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> latencyValues(1, deviceCount, key, perfFormatter, LATENCY);
    MemPtrChaseOperation ptrChaseOp(latencyMemAccessCnt);

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostBuffer dataBuffer(size, deviceId);
        latencyHelper(dataBuffer, false);
        latencyValues.value(0, deviceId) = ptrChaseOp.doPtrChase(deviceId, dataBuffer);
    }

    output->addTestcaseResults(latencyValues, "memory latency SM CPU(row) <-> GPU(column) (ns)");
}

void HostToDeviceSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM());

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostBuffer hostBuffer(size, deviceId);
        DeviceBuffer deviceBuffer(size, deviceId);

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(hostBuffer, deviceBuffer);
    }

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s)");
}

void DeviceToHostSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM());

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostBuffer hostBuffer(size, deviceId);
        DeviceBuffer deviceBuffer(size, deviceId);

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(deviceBuffer, hostBuffer);
    }

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) <- GPU(column) bandwidth (GB/s)");
}

void HostToDeviceBidirSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSMSplitWarp());

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostBuffer hostBuffer(size, deviceId);
        DeviceBuffer deviceBuffer(size, deviceId);

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(hostBuffer, deviceBuffer);
    }

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) <-> GPU(column) bandwidth (GB/s)");
}

void DeviceToHostBidirSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSMSplitWarp());

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostBuffer hostBuffer(size, deviceId);
        DeviceBuffer deviceBuffer(size, deviceId);

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(hostBuffer, deviceBuffer);
    }

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) <-> GPU(column) bandwidth (GB/s)");
}

// DtoD Read test - copy from dst to src (backwards) using src contxt
void DeviceToDeviceReadSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_DST_CONTEXT);

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

    output->addTestcaseResults(bandwidthValues, "memcpy SM GPU(row) -> GPU(column) bandwidth (GB/s)");
}

void DeviceToDeviceLatencySM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> latencyValues(deviceCount, deviceCount, key, perfFormatter, LATENCY);
    MemPtrChaseOperation ptrChaseOp(latencyMemAccessCnt);

    for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
        DeviceBuffer peerBuffer(size, peerDeviceId);
        latencyHelper(peerBuffer, true);

        for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            // Note: srcBuffer is not used in the pointer chase operation
            // It is simply used here to enable peer access
            DeviceBuffer srcBuffer(size, srcDeviceId);
            if (!srcBuffer.enablePeerAcess(peerBuffer)) {
                continue;
            }
            latencyValues.value(srcDeviceId, peerDeviceId) = ptrChaseOp.doPtrChase(srcDeviceId, peerBuffer);
        }
    }
    output->addTestcaseResults(latencyValues, "Device to Device Latency SM GPU(row) <-> GPU(column) (ns)");
}

// DtoD Write test - copy from src to dst using src context
void DeviceToDeviceWriteSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM());

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

    output->addTestcaseResults(bandwidthValues, "memcpy SM GPU(row) <- GPU(column) bandwidth (GB/s)");
}

// DtoD Bidir Read test - copy to dst from src (backwards) using dst contxt
void DeviceToDeviceBidirReadSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValuesRead1(deviceCount, deviceCount, key + "_read1");
    PeerValueMatrix<double> bandwidthValuesRead2(deviceCount, deviceCount, key + "_read2");
    PeerValueMatrix<double> bandwidthValuesTotal(deviceCount, deviceCount, key + "_total");
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_DST_CONTEXT, MemcpyOperation::VECTOR_BW);


    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

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

    output->addTestcaseResults(bandwidthValuesRead1, "memcpy SM GPU(row) <-> GPU(column) Read1 bandwidth (GB/s)");
    output->addTestcaseResults(bandwidthValuesRead2, "memcpy SM GPU(row) <-> GPU(column) Read2 bandwidth (GB/s)");
    output->addTestcaseResults(bandwidthValuesTotal, "memcpy SM GPU(row) <-> GPU(column) Total bandwidth (GB/s)");
}

// DtoD Bidir Write test - copy from  src to dst using src contxt
void DeviceToDeviceBidirWriteSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValuesWrite1(deviceCount, deviceCount, key + "_write1");
    PeerValueMatrix<double> bandwidthValuesWrite2(deviceCount, deviceCount, key + "_write2");
    PeerValueMatrix<double> bandwidthValuesTotal(deviceCount, deviceCount, key + "_total");
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);


    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

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

    output->addTestcaseResults(bandwidthValuesWrite1, "memcpy SM GPU(row) <-> GPU(column) Write1 bandwidth (GB/s)");
    output->addTestcaseResults(bandwidthValuesWrite2, "memcpy SM GPU(row) <-> GPU(column) Write2 bandwidth (GB/s)");
    output->addTestcaseResults(bandwidthValuesTotal, "memcpy SM GPU(row) <-> GPU(column) Total bandwidth (GB/s)");
}

void AllToHostSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::USE_FIRST_BW);

    allHostHelper(size, memcpyInstance, bandwidthValues, false);

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) <- GPU(column) bandwidth (GB/s)");
}

void AllToHostBidirSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSMSplitWarp(), PREFER_SRC_CONTEXT, MemcpyOperation::USE_FIRST_BW);

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        std::vector<const MemcpyBuffer*> srcBuffers;
        std::vector<const MemcpyBuffer*> dstBuffers;

        srcBuffers.push_back(new DeviceBuffer(size, deviceId));
        dstBuffers.push_back(new HostBuffer(size, deviceId));

        for (int interferenceDeviceId = 0; interferenceDeviceId < deviceCount; interferenceDeviceId++) {
            if (interferenceDeviceId == deviceId) {
                continue;
            }

            srcBuffers.push_back(new DeviceBuffer(size, interferenceDeviceId));
            dstBuffers.push_back(new HostBuffer(size, interferenceDeviceId));
        }

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(srcBuffers, dstBuffers);

        for (auto node : srcBuffers) {
            delete node;
        }

        for (auto node : dstBuffers) {
            delete node;
        }
    }

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) <-> GPU(column) bandwidth (GB/s)");
}

void HostToAllSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::USE_FIRST_BW);

    allHostHelper(size, memcpyInstance, bandwidthValues, true);

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s)");
}

void HostToAllBidirSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSMSplitWarp(), PREFER_SRC_CONTEXT, MemcpyOperation::USE_FIRST_BW);

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        std::vector<const MemcpyBuffer*> srcBuffers;
        std::vector<const MemcpyBuffer*> dstBuffers;

        srcBuffers.push_back(new HostBuffer(size, deviceId));
        dstBuffers.push_back(new DeviceBuffer(size, deviceId));

        for (int interferenceDeviceId = 0; interferenceDeviceId < deviceCount; interferenceDeviceId++) {
            if (interferenceDeviceId == deviceId) {
                continue;
            }

            srcBuffers.push_back(new DeviceBuffer(size, interferenceDeviceId));
            dstBuffers.push_back(new HostBuffer(size, interferenceDeviceId));
        }

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(srcBuffers, dstBuffers);

        for (auto node : srcBuffers) {
            delete node;
        }

        for (auto node : dstBuffers) {
            delete node;
        }
    }

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) <-> GPU(column) bandwidth (GB/s)");
}

/*
void HostToAnySM::run(unsigned long long size, unsigned long long loopCount) {
    // Use configured values or defaults
    int cpuId = hostToAnyCpuId;
    std::vector<int> gpuIds = ::gpuIds;
    unsigned long long streamBufferSize = (hostToAnyStreamBufferSize == 0) ? size : hostToAnyStreamBufferSize * _MiB;
    unsigned long long streamLoopCount = (hostToAnyStreamLoopCount == 0) ? loopCount : hostToAnyStreamLoopCount;
    unsigned long long interferenceBufferSize = 2 * streamBufferSize;  // 2x buffer for interference streams

    // If no GPUs specified, use all available GPUs
    if (gpuIds.empty()) {
        for (int i = 0; i < deviceCount; i++) {
            gpuIds.push_back(i);
        }
    }

    // Validate parameters
    if (cpuId < 0 || cpuId > 1) {
        OUTPUT << "ERROR: Invalid CPU ID " << cpuId << ". Must be 0 or 1." << std::endl;
        return;
    }
    for (int gpuId : gpuIds) {
        if (gpuId < 0 || gpuId >= deviceCount) {
            OUTPUT << "ERROR: Invalid GPU ID " << gpuId << ". Must be 0-" << (deviceCount - 1) << std::endl;
            return;
        }
    }
    if (streamCount < 1) {
        OUTPUT << "ERROR: Invalid stream count " << streamCount << ". Must be >= 1." << std::endl;
        return;
    }

    VERBOSE << "\n=== HostToAnySM Test Configuration ===" << std::endl;
    VERBOSE << "CPU ID: " << cpuId << std::endl;
    VERBOSE << "GPU IDs: ";
    for (int gpuId : gpuIds) {
        VERBOSE << gpuId << " ";
    }
    VERBOSE << std::endl;
    VERBOSE << "Streams per GPU (target): " << streamCount << std::endl;
    VERBOSE << "Buffer size per stream (target): " << (streamBufferSize / _MiB) << " MiB" << std::endl;
    VERBOSE << "Buffer size per stream (interference): " << (interferenceBufferSize / _MiB) << " MiB (2x target)" << std::endl;
    VERBOSE << "Loop count per stream: " << streamLoopCount << std::endl;
    VERBOSE << "Average loop count (test repetitions): " << averageLoopCount << std::endl;
    VERBOSE << "Test pattern: For each target GPU, test with interference from other GPUs" << std::endl;
    VERBOSE << "====================================\n" << std::endl;

    // Output matrix: rows = target GPUs, columns = streams + aggregate
    PeerValueMatrix<double> bandwidthValues(gpuIds.size(), streamCount + 1, key);

    // Test each GPU as target, with other GPUs as interference
    for (size_t targetIdx = 0; targetIdx < gpuIds.size(); targetIdx++) {
        int targetGpu = gpuIds[targetIdx];

        VERBOSE << "\n--- Testing Target GPU " << targetGpu << " ---" << std::endl;

        // Create buffers
        std::vector<const MemcpyBuffer*> hostBuffers;
        std::vector<const MemcpyBuffer*> deviceBuffers;

        // 1. Create target streams (this GPU being tested)
        VERBOSE << "Creating " << streamCount << " target stream(s) for GPU " << targetGpu
                << " with buffer size " << (streamBufferSize / _MiB) << " MiB" << std::endl;
        for (int s = 0; s < streamCount; s++) {
            hostBuffers.push_back(new HostBuffer(streamBufferSize, targetGpu));
            deviceBuffers.push_back(new DeviceBuffer(streamBufferSize, targetGpu));
        }

        // 2. Create interference streams (other GPUs with 2x buffer)
        for (size_t otherIdx = 0; otherIdx < gpuIds.size(); otherIdx++) {
            if (otherIdx == targetIdx) continue;
            int otherGpu = gpuIds[otherIdx];
            VERBOSE << "Creating " << streamCount << " interference stream(s) for GPU " << otherGpu
                    << " with buffer size " << (interferenceBufferSize / _MiB) << " MiB" << std::endl;
            for (int s = 0; s < streamCount; s++) {
                hostBuffers.push_back(new HostBuffer(interferenceBufferSize, otherGpu));
                deviceBuffers.push_back(new DeviceBuffer(interferenceBufferSize, otherGpu));
            }
        }

        // Execute all copies simultaneously
        MemcpyOperation memcpyInstance(streamLoopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);
        std::vector<double> allBandwidths = memcpyInstance.doMemcpyVector(hostBuffers, deviceBuffers);

        // Extract target GPU results (first streamCount entries)
        double targetAggregate = 0.0;
        VERBOSE << "Target GPU " << targetGpu << " stream bandwidths:" << std::endl;
        for (int s = 0; s < streamCount; s++) {
            double bw = allBandwidths[s];
            bandwidthValues.value(targetIdx, s) = bw;
            targetAggregate += bw;
            VERBOSE << "  Stream " << s << ": " << std::fixed << std::setprecision(2) << bw << " GB/s" << std::endl;
        }
        bandwidthValues.value(targetIdx, streamCount) = targetAggregate;
        VERBOSE << "  Aggregate: " << std::fixed << std::setprecision(2) << targetAggregate << " GB/s" << std::endl;

        // Cleanup
        for (auto buf : hostBuffers) delete buf;
        for (auto buf : deviceBuffers) delete buf;
    }

    // Print final results
    OUTPUT << "\n=== HostToAnySM Results ===" << std::endl;
    OUTPUT << "CPU " << cpuId << " -> GPUs, " << streamCount << " stream(s) per GPU" << std::endl;
    OUTPUT << "Target buffer: " << (streamBufferSize / _MiB) << " MiB, Interference buffer: " << (interferenceBufferSize / _MiB) << " MiB" << std::endl;
    OUTPUT << std::endl;

    OUTPUT << std::setw(10) << "Target GPU";
    for (int s = 0; s < streamCount; s++) {
        OUTPUT << std::setw(12) << ("Stream " + std::to_string(s));
    }
    OUTPUT << std::setw(12) << "Aggregate" << std::endl;

    for (size_t i = 0; i < gpuIds.size(); i++) {
        OUTPUT << std::setw(10) << ("GPU " + std::to_string(gpuIds[i]));
        for (int s = 0; s < streamCount; s++) {
            OUTPUT << std::setw(12) << std::fixed << std::setprecision(2) << bandwidthValues.value(i, s).value_or(0.0);
        }
        OUTPUT << std::setw(12) << std::fixed << std::setprecision(2) << bandwidthValues.value(i, streamCount).value_or(0.0);
        OUTPUT << " GB/s" << std::endl;
    }

    // Add results to standard output system (for JSON and perf formatter support)
    output->addTestcaseResults(bandwidthValues,
        "memcpy SM CPU -> GPU bandwidth (GB/s)\n"
        "  Rows: Target GPUs | Columns: Per-stream BW + Aggregate");
}

void AnyToHostSM::run(unsigned long long size, unsigned long long loopCount) {
    // Use configured values or defaults
    int cpuId = hostToAnyCpuId;
    std::vector<int> gpuIds = ::gpuIds;
    unsigned long long streamBufferSize = (hostToAnyStreamBufferSize == 0) ? size : hostToAnyStreamBufferSize * _MiB;
    unsigned long long streamLoopCount = (hostToAnyStreamLoopCount == 0) ? loopCount : hostToAnyStreamLoopCount;
    unsigned long long interferenceBufferSize = 2 * streamBufferSize;  // 2x buffer for interference streams

    // If no GPUs specified, use all available GPUs
    if (gpuIds.empty()) {
        for (int i = 0; i < deviceCount; i++) {
            gpuIds.push_back(i);
        }
    }

    // Validate parameters
    if (cpuId < 0 || cpuId > 1) {
        OUTPUT << "ERROR: Invalid CPU ID " << cpuId << ". Must be 0 or 1." << std::endl;
        return;
    }
    for (int gpuId : gpuIds) {
        if (gpuId < 0 || gpuId >= deviceCount) {
            OUTPUT << "ERROR: Invalid GPU ID " << gpuId << ". Must be 0-" << (deviceCount - 1) << std::endl;
            return;
        }
    }
    if (streamCount < 1) {
        OUTPUT << "ERROR: Invalid stream count " << streamCount << ". Must be >= 1." << std::endl;
        return;
    }

    VERBOSE << "\n=== AnyToHostSM Test Configuration ===" << std::endl;
    VERBOSE << "CPU ID: " << cpuId << std::endl;
    VERBOSE << "GPU IDs: ";
    for (int gpuId : gpuIds) {
        VERBOSE << gpuId << " ";
    }
    VERBOSE << std::endl;
    VERBOSE << "Streams per GPU (target): " << streamCount << std::endl;
    VERBOSE << "Buffer size per stream (target): " << (streamBufferSize / _MiB) << " MiB" << std::endl;
    VERBOSE << "Buffer size per stream (interference): " << (interferenceBufferSize / _MiB) << " MiB (2x target)" << std::endl;
    VERBOSE << "Loop count per stream: " << streamLoopCount << std::endl;
    VERBOSE << "Average loop count (test repetitions): " << averageLoopCount << std::endl;
    VERBOSE << "Test pattern: For each target GPU, test with interference from other GPUs" << std::endl;
    VERBOSE << "====================================\n" << std::endl;

    // Output matrix: rows = target GPUs, columns = streams + aggregate
    PeerValueMatrix<double> bandwidthValues(gpuIds.size(), streamCount + 1, key);

    // Test each GPU as target, with other GPUs as interference
    for (size_t targetIdx = 0; targetIdx < gpuIds.size(); targetIdx++) {
        int targetGpu = gpuIds[targetIdx];

        VERBOSE << "\n--- Testing Target GPU " << targetGpu << " ---" << std::endl;

        // Create buffers (reversed: device -> host)
        std::vector<const MemcpyBuffer*> deviceBuffers;  // source
        std::vector<const MemcpyBuffer*> hostBuffers;    // destination

        // 1. Create target streams (this GPU being tested)
        VERBOSE << "Creating " << streamCount << " target stream(s) for GPU " << targetGpu
                << " with buffer size " << (streamBufferSize / _MiB) << " MiB" << std::endl;
        for (int s = 0; s < streamCount; s++) {
            deviceBuffers.push_back(new DeviceBuffer(streamBufferSize, targetGpu));
            hostBuffers.push_back(new HostBuffer(streamBufferSize, targetGpu));
        }

        // 2. Create interference streams (other GPUs with 2x buffer)
        for (size_t otherIdx = 0; otherIdx < gpuIds.size(); otherIdx++) {
            if (otherIdx == targetIdx) continue;
            int otherGpu = gpuIds[otherIdx];
            VERBOSE << "Creating " << streamCount << " interference stream(s) for GPU " << otherGpu
                    << " with buffer size " << (interferenceBufferSize / _MiB) << " MiB" << std::endl;
            for (int s = 0; s < streamCount; s++) {
                deviceBuffers.push_back(new DeviceBuffer(interferenceBufferSize, otherGpu));
                hostBuffers.push_back(new HostBuffer(interferenceBufferSize, otherGpu));
            }
        }

        // Execute all copies simultaneously (device -> host)
        MemcpyOperation memcpyInstance(streamLoopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);
        std::vector<double> allBandwidths = memcpyInstance.doMemcpyVector(deviceBuffers, hostBuffers);

        // Extract target GPU results (first streamCount entries)
        double targetAggregate = 0.0;
        VERBOSE << "Target GPU " << targetGpu << " stream bandwidths:" << std::endl;
        for (int s = 0; s < streamCount; s++) {
            double bw = allBandwidths[s];
            bandwidthValues.value(targetIdx, s) = bw;
            targetAggregate += bw;
            VERBOSE << "  Stream " << s << ": " << std::fixed << std::setprecision(2) << bw << " GB/s" << std::endl;
        }
        bandwidthValues.value(targetIdx, streamCount) = targetAggregate;
        VERBOSE << "  Aggregate: " << std::fixed << std::setprecision(2) << targetAggregate << " GB/s" << std::endl;

        // Cleanup
        for (auto buf : hostBuffers) delete buf;
        for (auto buf : deviceBuffers) delete buf;
    }

    // Print final results
    OUTPUT << "\n=== AnyToHostSM Results ===" << std::endl;
    OUTPUT << "GPUs -> CPU " << cpuId << ", " << streamCount << " stream(s) per GPU" << std::endl;
    OUTPUT << "Target buffer: " << (streamBufferSize / _MiB) << " MiB, Interference buffer: " << (interferenceBufferSize / _MiB) << " MiB" << std::endl;
    OUTPUT << std::endl;

    OUTPUT << std::setw(10) << "Target GPU";
    for (int s = 0; s < streamCount; s++) {
        OUTPUT << std::setw(12) << ("Stream " + std::to_string(s));
    }
    OUTPUT << std::setw(12) << "Aggregate" << std::endl;

    for (size_t i = 0; i < gpuIds.size(); i++) {
        OUTPUT << std::setw(10) << ("GPU " + std::to_string(gpuIds[i]));
        for (int s = 0; s < streamCount; s++) {
            OUTPUT << std::setw(12) << std::fixed << std::setprecision(2) << bandwidthValues.value(i, s).value_or(0.0);
        }
        OUTPUT << std::setw(12) << std::fixed << std::setprecision(2) << bandwidthValues.value(i, streamCount).value_or(0.0);
        OUTPUT << " GB/s" << std::endl;
    }

    // Add results to standard output system (for JSON and perf formatter support)
    output->addTestcaseResults(bandwidthValues,
        "memcpy SM GPU -> CPU bandwidth (GB/s)\n"
        "  Rows: Target GPUs | Columns: Per-stream BW + Aggregate");
}
*/

// Write test - copy from src to dst using src context
void AllToOneWriteSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::TOTAL_BW);

    allToOneHelper(size, memcpyInstance, bandwidthValues, false);

    output->addTestcaseResults(bandwidthValues, "memcpy SM GPU(row) <- GPU(column) bandwidth (GB/s)");
}

// Read test - copy from dst to src (backwards) using src contxt
void AllToOneReadSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::TOTAL_BW);
    allToOneHelper(size, memcpyInstance, bandwidthValues, true);

    output->addTestcaseResults(bandwidthValues, "memcpy SM GPU(row) -> GPU(column) bandwidth (GB/s)");
}

// Write test - copy from src to dst using src context
void OneToAllWriteSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::TOTAL_BW);
    oneToAllHelper(size, memcpyInstance, bandwidthValues, false);

    output->addTestcaseResults(bandwidthValues, "memcpy SM GPU(row) -> GPU(column) bandwidth (GB/s)");
}

// Read test - copy from dst to src (backwards) using src contxt
void OneToAllReadSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::TOTAL_BW);
    oneToAllHelper(size, memcpyInstance, bandwidthValues, true);

    output->addTestcaseResults(bandwidthValues, "memcpy SM GPU(row) <- GPU(column) bandwidth (GB/s)");
}
