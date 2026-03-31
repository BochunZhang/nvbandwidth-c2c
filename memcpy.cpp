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

#include "common.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include "inline_common.h"
#include "memcpy.h"
#include "output.h"
#include "kernels.cuh"
#include "vector_types.h"
#ifdef MULTINODE
#include <mpi.h>
#include "multinode_memcpy.h"
#endif
#define WARMUP_COUNT 4
#define COOLDOWN_COUNT 4
#include <cassert>

MemcpyBuffer::MemcpyBuffer(size_t bufferSize): bufferSize(bufferSize), buffer(nullptr) {}

CUdeviceptr MemcpyBuffer::getBuffer() const {
    return (CUdeviceptr)buffer;
}

size_t MemcpyBuffer::getBufferSize() const {
    return bufferSize;
}

void xorshift2MBPattern(unsigned int* buffer, unsigned int seed) {
    unsigned int oldValue = seed;
    unsigned int n = 0;
    for (n = 0; n < (1024 * 1024 * 2) / sizeof(unsigned int); n++) {
        unsigned int value = oldValue;
        value = value ^ (value << 13);
        value = value ^ (value >> 17);
        value = value ^ (value << 5);
        oldValue = value;
        buffer[n] = oldValue;
    }
}

void memsetPatternHelper(CUstream stream, CUdeviceptr buffer, unsigned long long size, unsigned int seed, std::shared_ptr<NodeHelper> nodeHelper) {
    unsigned int* h_pattern;
    CUdeviceptr d_pattern;

    unsigned long long num_elements = size / sizeof(unsigned int);
    unsigned long long num_pattern_elements = _2MiB / sizeof(unsigned int);

    // Allocate 2MB of pattern
    CU_ASSERT(cuMemHostAlloc((void**)&h_pattern, sizeof(char) * _2MiB, CU_MEMHOSTALLOC_PORTABLE));
    xorshift2MBPattern(h_pattern, seed);

    // Copy the pattern to a device buffer
    CU_ASSERT(cuMemAlloc(&d_pattern, sizeof(char) * _2MiB));
    CU_ASSERT(cuMemcpyAsync(d_pattern, (CUdeviceptr)h_pattern, sizeof(char) * _2MiB, CU_STREAM_PER_THREAD));

    // Launch the memset kernel
    CU_ASSERT(memsetKernel(CU_STREAM_PER_THREAD, buffer, d_pattern, num_elements, num_pattern_elements));
    CU_ASSERT(nodeHelper->streamSynchronizeWrapper(CU_STREAM_PER_THREAD));

    CU_ASSERT(cuMemFreeHost((void*)h_pattern));
    CU_ASSERT(cuMemFree(d_pattern));
}

void memclearByWarpParity(CUstream stream, CUdeviceptr buffer, unsigned long long size, bool clearOddWarpIndexed, std::shared_ptr<NodeHelper> nodeHelper) {
    CU_ASSERT(memclearKernelByWarpParity(CU_STREAM_PER_THREAD, buffer, size, clearOddWarpIndexed));
    CU_ASSERT(nodeHelper->streamSynchronizeWrapper(CU_STREAM_PER_THREAD));
}

void MemcpyInitiatorCE::memsetPattern(MemcpyDispatchInfo &info) const {
    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        memsetPatternHelper(info.streams[i], info.dstBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xCAFEBABE, info.nodeHelper);
        memsetPatternHelper(info.streams[i], info.srcBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xBAADF00D, info.nodeHelper);
    }
}

void MemcpyInitiatorSM::memsetPattern(MemcpyDispatchInfo &info) const {
    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        memsetPatternHelper(info.streams[i], info.dstBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xCAFEBABE, info.nodeHelper);
        memsetPatternHelper(info.streams[i], info.srcBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xBAADF00D, info.nodeHelper);
    }
}

void MemcpyInitiatorMulticastWrite::memsetPattern(MemcpyDispatchInfo &info) const {
    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        memsetPatternHelper(info.streams[i], info.dstBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xCAFEBABE, info.nodeHelper);
        memsetPatternHelper(info.streams[i], info.srcBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xBAADF00D, info.nodeHelper);
    }
}

void MemcpyInitiatorSMSplitWarp::memsetPattern(MemcpyDispatchInfo &info) const {
    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        memsetPatternHelper(info.streams[i], info.dstBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xBAADF00D, info.nodeHelper);
        memsetPatternHelper(info.streams[i], info.srcBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xBAADF00D, info.nodeHelper);
        memclearByWarpParity(info.streams[i], info.dstBuffers[i]->getBuffer(), info.adjustedCopySizes[i], true /* clearOddWarpIndexed */, info.nodeHelper);
        memclearByWarpParity(info.streams[i], info.srcBuffers[i]->getBuffer(), info.adjustedCopySizes[i], false /* clearOddWarpIndexed */, info.nodeHelper);
    }
}

unsigned long long MemcpyInitiatorCE::getAdjustedBandwidth(unsigned long long bandwidth) {
    return bandwidth;
}

unsigned long long MemcpyInitiatorSM::getAdjustedBandwidth(unsigned long long bandwidth) {
    return bandwidth;
}

unsigned long long MemcpyInitiatorMulticastWrite::getAdjustedBandwidth(unsigned long long bandwidth) {
    return bandwidth;
}

unsigned long long MemcpyInitiatorSMSplitWarp::getAdjustedBandwidth(unsigned long long bandwidth) {
    // For split warp copies, we estimate bandwidth in each direction as 1/2 of measured bandwidth
    return bandwidth / 2;
}

// Add this new typedef for the comparison function pointer
typedef CUresult (*CompareKernelFunc)(CUstream, CUdeviceptr, CUdeviceptr, unsigned long long, unsigned int, CUdeviceptr);

void memcmpPatternHelper(CUstream stream, CUdeviceptr buffer, unsigned long long size, unsigned int seed, CompareKernelFunc compareKernel, std::shared_ptr<NodeHelper> nodeHelper) {
    unsigned int* h_pattern;
    CUdeviceptr d_pattern;
    int h_errorFlag = 0;
    CUdeviceptr d_errorFlag;
    unsigned long long num_elements = size / sizeof(unsigned int);
    unsigned long long num_pattern_elements = _2MiB / sizeof(unsigned int);

    // Allocate 2MB of pattern
    CU_ASSERT(cuMemHostAlloc((void**)&h_pattern, sizeof(char) * _2MiB, CU_MEMHOSTALLOC_PORTABLE));
    xorshift2MBPattern(h_pattern, seed);
    CU_ASSERT(cuMemAlloc(&d_pattern, sizeof(char) * _2MiB));
    CU_ASSERT(cuMemcpyAsync(d_pattern, (CUdeviceptr)h_pattern, sizeof(char) * _2MiB, CU_STREAM_PER_THREAD));

    // setup error flags
    CU_ASSERT(cuMemAlloc(&d_errorFlag, sizeof(int)));
    CU_ASSERT(cuMemcpyAsync(d_errorFlag, (CUdeviceptr)&h_errorFlag, sizeof(int), CU_STREAM_PER_THREAD));

    // launch kernel to compare
    CU_ASSERT(compareKernel(CU_STREAM_PER_THREAD, buffer, d_pattern, num_elements, num_pattern_elements, d_errorFlag));
    CU_ASSERT(nodeHelper->streamSynchronizeWrapper(CU_STREAM_PER_THREAD));
    CU_ASSERT(cuMemcpyAsync((CUdeviceptr)&h_errorFlag, d_errorFlag, sizeof(int), CU_STREAM_PER_THREAD));

    CU_ASSERT(cuMemFreeHost((void*)h_pattern));
    CU_ASSERT(cuMemFree(d_errorFlag));
    CU_ASSERT(cuMemFree(d_pattern));

    ASSERT(h_errorFlag == 0);
}

void MemcpyInitiatorCE::memcmpPattern(MemcpyDispatchInfo &info) const {
    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        memcmpPatternHelper(info.streams[i], info.dstBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xBAADF00D, memcmpKernel, info.nodeHelper);
    }
}

void MemcpyInitiatorSM::memcmpPattern(MemcpyDispatchInfo &info) const {
    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        memcmpPatternHelper(info.streams[i], info.dstBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xBAADF00D, memcmpKernel, info.nodeHelper);
    }
}

void MemcpyInitiatorMulticastWrite::memcmpPattern(MemcpyDispatchInfo &info) const {
    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        memcmpPatternHelper(info.streams[i], info.dstBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xBAADF00D, multicastMemcmpKernel, info.nodeHelper);
    }
}

void MemcpyInitiatorSMSplitWarp::memcmpPattern(MemcpyDispatchInfo &info) const {
    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        // src and dst buffer contents must match after the bidirectional split warp copy
        memcmpPatternHelper(info.streams[i], info.dstBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xBAADF00D, memcmpKernel, info.nodeHelper);
        memcmpPatternHelper(info.streams[i], info.srcBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xBAADF00D, memcmpKernel, info.nodeHelper);
    }
}

// Non-multinode MemcpyBuffers will always return Rank 0
// Semantically, we can assume that non-MPI runs will have world rank = 0 and world size = 1
int MemcpyBuffer::getMPIRank() const {
    return 0;
}

CUresult MemcpyBuffer::streamSynchronizeWrapper(CUstream stream) const {
    return cuStreamSynchronize(stream);
}

HostBuffer::HostBuffer(size_t bufferSize, int targetDeviceId): MemcpyBuffer(bufferSize) {
    CUcontext targetCtx;

    // Before allocating host memory, set correct NUMA affinity
    setOptimalCpuAffinity(targetDeviceId);
    CU_ASSERT(cuDevicePrimaryCtxRetain(&targetCtx, targetDeviceId));
    CU_ASSERT(cuCtxSetCurrent(targetCtx));

    CU_ASSERT(cuMemHostAlloc(&buffer, bufferSize, CU_MEMHOSTALLOC_PORTABLE));
}

HostBuffer::~HostBuffer() {
    if (isMemoryOwnedByCUDA(buffer)) {
        CU_ASSERT(cuMemFreeHost(buffer));
    } else {
        free(buffer);
    }
}

// Host nodes don't have a context, return null
CUcontext HostBuffer::getPrimaryCtx() const {
    return nullptr;
}

// Host buffers always return zero as they always represent one row in the bandwidth matrix
int HostBuffer::getBufferIdx() const {
    return 0;
}

std::string HostBuffer::getBufferString() const {
    return "Host";
}

DeviceBuffer::DeviceBuffer(size_t bufferSize, int deviceIdx): deviceIdx(deviceIdx), MemcpyBuffer(bufferSize) {
    CU_ASSERT(cuDevicePrimaryCtxRetain(&primaryCtx, deviceIdx));
    CU_ASSERT(cuCtxSetCurrent(primaryCtx));
    CU_ASSERT(cuMemAlloc((CUdeviceptr*)&buffer, bufferSize));
}

DeviceBuffer::~DeviceBuffer() {
    CU_ASSERT(cuCtxSetCurrent(primaryCtx));
    CU_ASSERT(cuMemFree((CUdeviceptr)buffer));
    CU_ASSERT(cuDevicePrimaryCtxRelease(deviceIdx));
}

CUcontext DeviceBuffer::getPrimaryCtx() const {
    return primaryCtx;
}

int DeviceBuffer::getBufferIdx() const {
    return deviceIdx;
}

std::string DeviceBuffer::getBufferString() const {
    return "Device " + std::to_string(deviceIdx);
}

bool DeviceBuffer::enablePeerAcess(const DeviceBuffer &peerBuffer) {
    int canAccessPeer = 0;
    CU_ASSERT(cuDeviceCanAccessPeer(&canAccessPeer, getBufferIdx(), peerBuffer.getBufferIdx()));
    if (canAccessPeer) {
        CUresult res;
        CU_ASSERT(cuCtxSetCurrent(peerBuffer.getPrimaryCtx()));
        res = cuCtxEnablePeerAccess(getPrimaryCtx(), 0);
        if (res != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
            CU_ASSERT(res);

        CU_ASSERT(cuCtxSetCurrent(getPrimaryCtx()));
        res = cuCtxEnablePeerAccess(peerBuffer.getPrimaryCtx(), 0);
        if (res != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
            CU_ASSERT(res);

        return true;
    }
    return false;
}

MemcpyDescriptor::MemcpyDescriptor(CUdeviceptr dst, CUdeviceptr src, CUstream stream, size_t copySize, unsigned long long loopCount) :
    dst(dst), src(src), stream(stream), copySize(copySize), loopCount(loopCount) {}

MemcpyOperation::MemcpyOperation(unsigned long long loopCount, MemcpyInitiator* memcpyInitiator, ContextPreference ctxPreference, BandwidthValue bandwidthValue) :
    MemcpyOperation(loopCount, memcpyInitiator, new NodeHelperSingle(), ctxPreference, bandwidthValue) {}

MemcpyOperation::MemcpyOperation(unsigned long long loopCount, MemcpyInitiator* memcpyInitiator, NodeHelper* nodeHelper, ContextPreference ctxPreference, BandwidthValue bandwidthValue) :
        loopCount(loopCount), memcpyInitiator(memcpyInitiator), nodeHelper(nodeHelper), ctxPreference(ctxPreference), bandwidthValue(bandwidthValue) {
    procMask = (size_t *)calloc(1, PROC_MASK_SIZE);
    PROC_MASK_SET(procMask, getFirstEnabledCPU());
}

MemcpyOperation::~MemcpyOperation() {
    PROC_MASK_CLEAR(procMask, 0);
}

double MemcpyOperation::doMemcpy(const MemcpyBuffer &srcBuffer, const MemcpyBuffer &dstBuffer) {
    std::vector<const MemcpyBuffer*> srcBuffers = {&srcBuffer};
    std::vector<const MemcpyBuffer*> dstBuffers = {&dstBuffer};
    return doMemcpy(srcBuffers, dstBuffers);
}

MemcpyDispatchInfo::MemcpyDispatchInfo(std::vector<const MemcpyBuffer*> srcBuffers, std::vector<const MemcpyBuffer*> dstBuffers, std::vector<CUcontext> contexts, std::vector<int> originalRanks) :
    srcBuffers(srcBuffers), dstBuffers(dstBuffers), contexts(contexts), originalRanks(originalRanks) {
}

NodeHelperSingle::NodeHelperSingle() {
    CU_ASSERT(cuMemHostAlloc((void **)&blockingVarHost, sizeof(*blockingVarHost), CU_MEMHOSTALLOC_PORTABLE));
}

NodeHelperSingle::~NodeHelperSingle() {
    CU_ASSERT(cuMemFreeHost((void*)blockingVarHost));
}

MemcpyDispatchInfo NodeHelperSingle::dispatchMemcpy(const std::vector<const MemcpyBuffer*> &srcBuffers, const std::vector<const MemcpyBuffer*> &dstBuffers, ContextPreference ctxPreference) {
    std::vector<CUcontext> contexts(srcBuffers.size());

    for (int i = 0; i < srcBuffers.size(); i++) {
        // prefer source context
        if (ctxPreference == PREFER_SRC_CONTEXT && srcBuffers[i]->getPrimaryCtx() != nullptr) {
            contexts[i] = srcBuffers[i]->getPrimaryCtx();
        } else if (dstBuffers[i]->getPrimaryCtx() != nullptr) {
            contexts[i] = dstBuffers[i]->getPrimaryCtx();
        }
    }

    return MemcpyDispatchInfo(srcBuffers, dstBuffers, contexts);
}

double NodeHelperSingle::calculateTotalBandwidth(double totalTime, double totalSize, size_t loopCount) {
    return (totalSize * loopCount * 1000ull * 1000ull) / totalTime;
}

double NodeHelperSingle::calculateSumBandwidth(std::vector<PerformanceStatistic> &bandwidthStats) {
    double sum = 0.0;
    for (auto stat : bandwidthStats) {
        sum += stat.returnAppropriateMetric() * 1e-9;
    }
    return sum;
}

double NodeHelperSingle::calculateFirstBandwidth(std::vector<PerformanceStatistic> &bandwidthStats) {
    return bandwidthStats[0].returnAppropriateMetric() * 1e-9;
}

std::vector<double> NodeHelperSingle::calculateVectorBandwidth(std::vector<double> &results, std::vector<int> originalRanks) {
    return results;
}

void NodeHelperSingle::synchronizeProcess() {
    // NOOP
}

CUresult NodeHelperSingle::streamSynchronizeWrapper(CUstream stream) const {
    return cuStreamSynchronize(stream);
}

void NodeHelperSingle::streamBlockerReset() {
    *blockingVarHost = 0;
}

void NodeHelperSingle::streamBlockerRelease() {
    *blockingVarHost = 1;
}

void NodeHelperSingle::streamBlockerBlock(CUstream stream) {
    // start the spin kernel on the stream
    CU_ASSERT(spinKernel(blockingVarHost, stream));
}

double MemcpyOperation::doMemcpy(const std::vector<const MemcpyBuffer*> &srcBuffers, const std::vector<const MemcpyBuffer*> &dstBuffers) {
    MemcpyDispatchInfo dispatchInfo = nodeHelper->dispatchMemcpy(srcBuffers, dstBuffers, ctxPreference);
    auto result = doMemcpyCore(dispatchInfo);
    return result[0];
}

std::vector<double> MemcpyOperation::doMemcpyVector(const std::vector<const MemcpyBuffer*> &srcBuffers, const std::vector<const MemcpyBuffer*> &dstBuffers) {
    MemcpyDispatchInfo dispatchInfo = nodeHelper->dispatchMemcpy(srcBuffers, dstBuffers, ctxPreference);
    auto results = doMemcpyCore(dispatchInfo);

    return nodeHelper->calculateVectorBandwidth(results, dispatchInfo.originalRanks);
}


// copy engine 测试带宽
// 有一条 master stream, 其他为 interference stream
// 每条流做 4x warmup memcpy
// master stream 在 warmup 后记录 start event
// interference stream 在 warmup 后, 需要等待 master stream 完成 start event 后开始 memcpy
// 如果计算 TOTAL_BW, 则 master stream 需要等待所有 interference stream 完成 memcpy 后才记录 end event
std::vector<double> MemcpyOperation::doMemcpyCore(MemcpyDispatchInfo &info) {
    std::vector<CUstream> streams(info.srcBuffers.size());
    std::vector<CUevent> startEvents(info.srcBuffers.size());
    std::vector<CUevent> endEvents(info.srcBuffers.size());
    std::vector<CUevent> warmupStartEvents(info.srcBuffers.size());
    std::vector<CUevent> warmupEndEvents(info.srcBuffers.size());
    std::vector<PerformanceStatistic> bandwidthStats(info.srcBuffers.size());
    std::vector<size_t> adjustedCopySizes(info.srcBuffers.size());
    PerformanceStatistic totalBandwidth;
    CUevent totalEnd;
    std::vector<size_t> finalCopySize(info.srcBuffers.size());

    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        // allocate the per simulaneous copy resources
        CU_ASSERT(cuStreamCreate(&streams[i], CU_STREAM_NON_BLOCKING));
        info.streams.push_back(streams[i]);
        CU_ASSERT(cuEventCreate(&startEvents[i], CU_EVENT_DEFAULT));
        CU_ASSERT(cuEventCreate(&endEvents[i], CU_EVENT_DEFAULT));
        CU_ASSERT(cuEventCreate(&warmupStartEvents[i], CU_EVENT_DEFAULT));
        CU_ASSERT(cuEventCreate(&warmupEndEvents[i], CU_EVENT_DEFAULT));
        // Get the final copy size that will be used.
        // CE and SM copy sizes will differ due to possible truncation
        // during SM copies.
        finalCopySize[i] = memcpyInitiator->getAdjustedCopySize(info.srcBuffers[i]->getBufferSize(), streams[i]);
        info.adjustedCopySizes.push_back(finalCopySize[i]);
    }
    info.nodeHelper = nodeHelper;

    if (info.contexts.size() > 0) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[0]));
    }
    // If no memcpy operations are happening on this node, let's still record a totalEnd event to simplify code
    CU_ASSERT(cuEventCreate(&totalEnd, CU_EVENT_DEFAULT));

    // This loop is for sampling the testcase (which itself has a loop count)
    for (unsigned int n = 0; n < averageLoopCount; n++) {
        nodeHelper->streamBlockerReset();
        nodeHelper->synchronizeProcess();

        memcpyInitiator->memsetPattern(info);
        // block stream, and enqueue copy
        for (int i = 0; i < info.srcBuffers.size(); i++) {
            CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));

            nodeHelper->streamBlockerBlock(info.streams[i]);

            // record warmup start event
            CU_ASSERT(cuEventRecord(warmupStartEvents[i], info.streams[i]));

            // warmup
            MemcpyDescriptor desc(info.dstBuffers[i]->getBuffer(), info.srcBuffers[i]->getBuffer(), info.streams[i], info.srcBuffers[i]->getBufferSize(), WARMUP_COUNT);
            memcpyInitiator->memcpyFunc(desc);

            // record warmup end event
            CU_ASSERT(cuEventRecord(warmupEndEvents[i], info.streams[i]));
        }

        if (info.srcBuffers.size() > 0) {
            CU_ASSERT(cuCtxSetCurrent(info.contexts[0]));
            CU_ASSERT(cuEventRecord(startEvents[0], info.streams[0]));
        }

        for (int i = 1; i < info.srcBuffers.size(); i++) {
            // ensure that all copies are launched at the same time
            CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
            CU_ASSERT(cuStreamWaitEvent(info.streams[i], startEvents[0], 0));
            CU_ASSERT(cuEventRecord(startEvents[i], info.streams[i]));
        }

        for (int i = 0; i < info.srcBuffers.size(); i++) {
            CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
            ASSERT(info.srcBuffers[i]->getBufferSize() == info.dstBuffers[i]->getBufferSize());
            MemcpyDescriptor desc(info.dstBuffers[i]->getBuffer(), info.srcBuffers[i]->getBuffer(), info.streams[i], info.srcBuffers[i]->getBufferSize(), loopCount);
            adjustedCopySizes[i] = memcpyInitiator->memcpyFunc(desc);
            CU_ASSERT(cuEventRecord(endEvents[i], info.streams[i]));
            if (bandwidthValue == BandwidthValue::TOTAL_BW && i != 0) {
                // make stream0 wait on the all the others so we can measure total completion time
                CU_ASSERT(cuStreamWaitEvent(info.streams[0], endEvents[i], 0));
            }
        }

        // record the total end - only valid if BandwidthValue::TOTAL_BW is used due to StreamWaitEvent above
        if (info.srcBuffers.size() > 0) {
            CU_ASSERT(cuCtxSetCurrent(info.contexts[0]));
            CU_ASSERT(cuEventRecord(totalEnd, info.streams[0]));
        }

        // unblock the streams
        nodeHelper->streamBlockerRelease();

        for (CUstream stream : info.streams) {
            CU_ASSERT(nodeHelper->streamSynchronizeWrapper(stream));
        }

        nodeHelper->synchronizeProcess();

        if (!skipVerification) {
            memcpyInitiator->memcmpPattern(info);
        }

        for (int i = 0; i < bandwidthStats.size(); i++) {
            float timeWithEvents = 0.0f;
            CU_ASSERT(cuEventElapsedTime(&timeWithEvents, startEvents[i], endEvents[i]));
            double elapsedWithEventsInUs = ((double) timeWithEvents * 1000.0);
            unsigned long long bandwidth = (adjustedCopySizes[i] * loopCount * 1000ull * 1000ull) / (unsigned long long) elapsedWithEventsInUs;

            bandwidth = memcpyInitiator->getAdjustedBandwidth(bandwidth);

            bandwidthStats[i]((double) bandwidth);

            // if (bandwidthValue == BandwidthValue::SUM_BW || BandwidthValue::TOTAL_BW || i == 0) {
            //     // Verbose print only the values that are used for the final output
            //     VERBOSE << "\tSample " << n << ": " << info.srcBuffers[i]->getBufferString() << " -> " << info.dstBuffers[i]->getBufferString() << ": " <<
            //         std::fixed << std::setprecision(2) << (double)bandwidth * 1e-9 << " GB/s\n";
            // }

            // Print timing information for each stream
            float warmupTime = 0.0f, startTime = 0.0f, totalTime = 0.0f;
            CU_ASSERT(cuEventElapsedTime(&warmupTime, warmupStartEvents[i], warmupEndEvents[i]));
            CU_ASSERT(cuEventElapsedTime(&startTime, warmupStartEvents[i], startEvents[i]));
            CU_ASSERT(cuEventElapsedTime(&totalTime, warmupStartEvents[i], endEvents[i]));

            if (bandwidthValue == BandwidthValue::SUM_BW || BandwidthValue::TOTAL_BW || i == 0) {
                VERBOSE << "\tSample " << n << ": " << info.srcBuffers[i]->getBufferString() << " -> " << info.dstBuffers[i]->getBufferString() << ": " <<
                    std::fixed << std::setprecision(2) << (double)bandwidth * 1e-9 << " GB/s, "
                    << "warmup end at: " << std::fixed << std::setprecision(3) << warmupTime << " ms,"
                    << "memcpy start at: " << std::fixed << std::setprecision(3) << startTime << " ms,"
                    << "memcpy end at: " << std::fixed << std::setprecision(3) << totalTime << " ms\n";
            }
        }

        if (bandwidthValue == BandwidthValue::TOTAL_BW) {
            float totalTime = 0.0f;
            if (startEvents.size() > 0) {
                CU_ASSERT(cuEventElapsedTime(&totalTime, startEvents[0], totalEnd));
            }
            double elapsedTotalInUs = ((double) totalTime * 1000.0);

            // get total bytes copied
            double totalSize = 0;
            for (double size : adjustedCopySizes) {
                totalSize += size;
            }

            double bandwidth = nodeHelper->calculateTotalBandwidth(elapsedTotalInUs, totalSize, loopCount);
            totalBandwidth(bandwidth);
            VERBOSE << "\tSample " << n << ": Total Bandwidth : " <<
                std::fixed << std::setprecision(2) << (double)bandwidth * 1e-9 << " GB/s\n";
        }
    }

    // cleanup
    CU_ASSERT(cuEventDestroy(totalEnd));

    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuStreamDestroy(info.streams[i]));
        CU_ASSERT(cuEventDestroy(startEvents[i]));
        CU_ASSERT(cuEventDestroy(endEvents[i]));
        CU_ASSERT(cuEventDestroy(warmupStartEvents[i]));
        CU_ASSERT(cuEventDestroy(warmupEndEvents[i]));
    }

    if (bandwidthValue == BandwidthValue::SUM_BW) {
        return {nodeHelper->calculateSumBandwidth(bandwidthStats)};
    } else if (bandwidthValue == BandwidthValue::TOTAL_BW) {
        return {totalBandwidth.returnAppropriateMetric() * 1e-9};
    } else if (bandwidthValue == BandwidthValue::VECTOR_BW) {
        std::vector<double> ret;
        for (auto stat : bandwidthStats) {
            ret.push_back(stat.returnAppropriateMetric() * 1e-9);
        }
        return ret;
    } else {
        return {nodeHelper->calculateFirstBandwidth(bandwidthStats)};
    }
}



// CustomMemcpyOperation constructors: takes ownership of each raw MemcpyInitiator* and stores in memcpyInitiators.
// Uses a CE initiator as the shared fallback for memset/memcmp pattern operations.
CustomMemcpyOperation::CustomMemcpyOperation(unsigned long long loopCount, std::vector<MemcpyInitiator*> initiators, ContextPreference ctxPreference, BandwidthValue bandwidthValue) :
    CustomMemcpyOperation(loopCount, initiators, new NodeHelperSingle(), ctxPreference, bandwidthValue) {}

CustomMemcpyOperation::CustomMemcpyOperation(unsigned long long loopCount, std::vector<MemcpyInitiator*> initiators, NodeHelper *_nodeHelper, ContextPreference ctxPreference, BandwidthValue bandwidthValue) :
        loopCount(loopCount), nodeHelper(_nodeHelper), ctxPreference(ctxPreference), bandwidthValue(bandwidthValue),
        fallbackInitiator(new MemcpyInitiatorCE()) {
    procMask = (size_t *)calloc(1, PROC_MASK_SIZE);
    PROC_MASK_SET(procMask, getFirstEnabledCPU());
    for (auto* init : initiators) {
        memcpyInitiators.push_back(std::shared_ptr<MemcpyInitiator>(init));
    }
}

CustomMemcpyOperation::~CustomMemcpyOperation() {
    PROC_MASK_CLEAR(procMask, 0);
}


double CustomMemcpyOperation::doMemcpy(const MemcpyBuffer &srcBuffer, const MemcpyBuffer &dstBuffer, InitiatorType type) {
    std::vector<const MemcpyBuffer*> srcBuffers = {&srcBuffer};
    std::vector<const MemcpyBuffer*> dstBuffers = {&dstBuffer};
    std::vector<InitiatorType> types = {type};
    return doMemcpy(srcBuffers, dstBuffers, types);
}


double CustomMemcpyOperation::doMemcpy(const std::vector<const MemcpyBuffer*> &srcBuffers, const std::vector<const MemcpyBuffer*> &dstBuffers, const std::vector<InitiatorType> &types) {
    MemcpyDispatchInfo dispatchInfo = nodeHelper->dispatchMemcpy(srcBuffers, dstBuffers, ctxPreference);
    auto result = doMemcpyCore(dispatchInfo, types);
    return result[0];
}


std::vector<double> CustomMemcpyOperation::doMemcpyVector(const std::vector<const MemcpyBuffer*> &srcBuffers, const std::vector<const MemcpyBuffer*> &dstBuffers, const std::vector<InitiatorType> &types) {
    MemcpyDispatchInfo dispatchInfo = nodeHelper->dispatchMemcpy(srcBuffers, dstBuffers, ctxPreference);
    std::vector<double> results = doMemcpyCore(dispatchInfo, types);
    return nodeHelper->calculateVectorBandwidth(results, dispatchInfo.originalRanks);
}


// Custom multi-stream memcpy with warmup/test/cooldown phases.
// Each stream dispatches via its own initiator (from memcpyInitiators[i]) if set,
// otherwise falls back to the fallbackInitiator.
// Supports CONCURRENT_BW (per-GPU aggregate) and VECTOR_BW (per-stream) return modes.
std::vector<double> CustomMemcpyOperation::doMemcpyCore(MemcpyDispatchInfo &info, const std::vector<InitiatorType> &types) {
    std::vector<CUstream> streams(info.srcBuffers.size());
    std::vector<CUevent> warmupStartEvents(info.srcBuffers.size());
    std::vector<CUevent> testStartEvents(info.srcBuffers.size());
    std::vector<CUevent> cooldownStartEvents(info.srcBuffers.size());
    std::vector<CUevent> cooldownEndEvents(info.srcBuffers.size());
    std::vector<PerformanceStatistic> bandwidthStats(info.srcBuffers.size());
    // std::vector<PerformanceStatistic> aggregateStats(gpuIds.size());
    std::vector<size_t> adjustedCopySizes(info.srcBuffers.size());
    PerformanceStatistic totalBandwidth;
    CUevent totalEnd;
    std::vector<size_t> finalCopySize(info.srcBuffers.size());

    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        // allocate the per simulaneous copy resources
        CU_ASSERT(cuStreamCreate(&streams[i], CU_STREAM_NON_BLOCKING));
        info.streams.push_back(streams[i]);
        CU_ASSERT(cuEventCreate(&warmupStartEvents[i], CU_EVENT_DEFAULT));
        CU_ASSERT(cuEventCreate(&testStartEvents[i], CU_EVENT_DEFAULT));
        CU_ASSERT(cuEventCreate(&cooldownStartEvents[i], CU_EVENT_DEFAULT));
        CU_ASSERT(cuEventCreate(&cooldownEndEvents[i], CU_EVENT_DEFAULT));
        // Get the final copy size per stream; CE returns size unchanged, SM may truncate.
        finalCopySize[i] = memcpyInitiators[static_cast<size_t>(types[i])]->getAdjustedCopySize(info.srcBuffers[i]->getBufferSize(), streams[i]);
        info.adjustedCopySizes.push_back(finalCopySize[i]);
    }
    info.nodeHelper = nodeHelper;

    if (info.contexts.size() > 0) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[0]));
    }
    // If no memcpy operations are happening on this node, let's still record a totalEnd event to simplify code
    CU_ASSERT(cuEventCreate(&totalEnd, CU_EVENT_DEFAULT));

    // This loop is for sampling the testcase (which itself has a loop count)
    for (unsigned int n = 0; n < averageLoopCount; n++) {
        nodeHelper->streamBlockerReset();
        nodeHelper->synchronizeProcess();

        // Use the fallback (CE) initiator for pattern setup/verification; pattern logic is identical
        // across CE and SM initiators so this is correct for mixed-initiator scenarios.
        fallbackInitiator->memsetPattern(info);

        // ========== Phase 1: Warmup ==========
        for (int i = 0; i < (int)info.srcBuffers.size(); i++) {
            CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));

            nodeHelper->streamBlockerBlock(info.streams[i]);

            CU_ASSERT(cuEventRecord(warmupStartEvents[i], info.streams[i]));

            MemcpyDescriptor warmupDesc(info.dstBuffers[i]->getBuffer(), info.srcBuffers[i]->getBuffer(), info.streams[i], info.srcBuffers[i]->getBufferSize(), WARMUP_COUNT);
            memcpyInitiators[static_cast<size_t>(types[i])]->memcpyFunc(warmupDesc);
        }

        // ========== Phase 2: Test ==========
        // All streams are synchronized by streamBlockerRelease(), no master stream needed
        for (int i = 0; i < (int)info.srcBuffers.size(); i++) {
            CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
            ASSERT(info.srcBuffers[i]->getBufferSize() == info.dstBuffers[i]->getBufferSize());

            CU_ASSERT(cuEventRecord(testStartEvents[i], info.streams[i]));

            MemcpyDescriptor testDesc(info.dstBuffers[i]->getBuffer(), info.srcBuffers[i]->getBuffer(), info.streams[i], info.srcBuffers[i]->getBufferSize(), loopCount);
            adjustedCopySizes[i] = memcpyInitiators[static_cast<size_t>(types[i])]->memcpyFunc(testDesc);
        }

        // ========== Phase 3: Cooldown ==========
        for (int i = 0; i < (int)info.srcBuffers.size(); i++) {
            CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));

            CU_ASSERT(cuEventRecord(cooldownStartEvents[i], info.streams[i]));
            MemcpyDescriptor cooldownDesc(info.dstBuffers[i]->getBuffer(), info.srcBuffers[i]->getBuffer(), info.streams[i], info.srcBuffers[i]->getBufferSize(), COOLDOWN_COUNT);
            memcpyInitiators[static_cast<size_t>(types[i])]->memcpyFunc(cooldownDesc);
            CU_ASSERT(cuEventRecord(cooldownEndEvents[i], info.streams[i]));
        }

        if (info.srcBuffers.size() > 0) {
            CU_ASSERT(cuCtxSetCurrent(info.contexts[0]));
            CU_ASSERT(cuEventRecord(totalEnd, info.streams[0]));
        }

        // unblock the streams
        nodeHelper->streamBlockerRelease();

        for (CUstream stream : info.streams) {
            CU_ASSERT(nodeHelper->streamSynchronizeWrapper(stream));
        }

        nodeHelper->synchronizeProcess();

        if (!skipVerification) {
            fallbackInitiator->memcmpPattern(info);
        }

        // ========== Bandwidth calculation: only measure Test phase ==========
        std::vector<unsigned long long> aggregate(gpuIds.size(), 0);
        for (int i = 0; i < (int)bandwidthStats.size(); i++) {
            float warmTime = 0.0f, testTime = 0.0f, coolTime = 0.0f, totalTime = 0.0f;
            CU_ASSERT(cuEventElapsedTime(&warmTime, warmupStartEvents[i], testStartEvents[i]));
            CU_ASSERT(cuEventElapsedTime(&testTime, testStartEvents[i], cooldownStartEvents[i]));
            CU_ASSERT(cuEventElapsedTime(&coolTime, cooldownStartEvents[i], cooldownEndEvents[i]));
            CU_ASSERT(cuEventElapsedTime(&totalTime, warmupStartEvents[i], cooldownEndEvents[i]));
            double elapsedTestInUs = ((double) testTime * 1000.0);
            unsigned long long bandwidth = (adjustedCopySizes[i] * loopCount * 1000ull * 1000ull) / (unsigned long long) elapsedTestInUs;

            bandwidth = memcpyInitiators[static_cast<size_t>(types[i])]->getAdjustedBandwidth(bandwidth);

            bandwidthStats[i]((double) bandwidth);

            VERBOSE << "\tSample " << n << ": " << info.srcBuffers[i]->getBufferString() << " -> " << info.dstBuffers[i]->getBufferString() << ": "
                << "stream " << i << ": "
                << std::fixed << std::setprecision(2) << (double)bandwidth * 1e-9 << " GB/s, "
                << "warm: " << std::fixed << std::setprecision(3) << warmTime << " ms, "
                << "test: " << std::fixed << std::setprecision(3) << testTime << " ms, "
                << "cool: " << std::fixed << std::setprecision(3) << coolTime << " ms, "
                << "total: " << std::fixed << std::setprecision(3) << totalTime << " ms\n";

            // if (bandwidthValue == BandwidthValue::CONCURRENT_BW) {
            //     aggregate[i / streamCount] += bandwidth;
            //     if (i % streamCount == streamCount - 1) {
            //         aggregateStats[i / streamCount]((double)(aggregate[i / streamCount] / streamCount));
            //         VERBOSE << "\t\tAggregate bandwidth: "
            //             << std::fixed << std::setprecision(2) << (double)aggregate[i / streamCount] * 1e-9 << " GB/s\n";
            //     }
            // }
        }
    }

    // cleanup
    CU_ASSERT(cuEventDestroy(totalEnd));

    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuStreamDestroy(info.streams[i]));
        CU_ASSERT(cuEventDestroy(warmupStartEvents[i]));
        CU_ASSERT(cuEventDestroy(testStartEvents[i]));
        CU_ASSERT(cuEventDestroy(cooldownStartEvents[i]));
        CU_ASSERT(cuEventDestroy(cooldownEndEvents[i]));
    }

    if (bandwidthValue == BandwidthValue::VECTOR_BW) {
        double total = 0.0;
        std::vector<double> ret;
        for (auto stat : bandwidthStats) {
            ret.push_back(stat.returnAppropriateMetric() * 1e-9);
        }
        return ret;
    }
    return {0};
}

size_t MemcpyInitiatorSM::memcpyFunc(MemcpyDescriptor &desc) {
    return copyKernel(desc);
}

size_t MemcpyInitiatorSM::getAdjustedCopySize(size_t size, CUstream stream) {
    CUdevice dev;
    CUcontext ctx;

    CU_ASSERT(cuStreamGetCtx(stream, &ctx));
    CU_ASSERT(cuCtxGetDevice(&dev));
    int numSm;
    CU_ASSERT(cuDeviceGetAttribute(&numSm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    unsigned int totalThreadCount = numSm * numThreadPerBlock;
    // We want to calculate the exact copy sizes that will be
    // used by the copy kernels.
    if (size < (smallBufferThreshold * _MiB)) {
        // copy size is rounded down to 16 bytes
        int numUint4 = size / sizeof(uint4);
        return numUint4 * sizeof(uint4);
    }
    // adjust size to elements (size is multiple of MB, so no truncation here)
    size_t sizeInElement = size / sizeof(uint4);
    // this truncates the copy
    sizeInElement = totalThreadCount * (sizeInElement / totalThreadCount);
    return sizeInElement * sizeof(uint4);
}

size_t MemcpyInitiatorCE::memcpyFunc(MemcpyDescriptor &desc) {
    for (unsigned int l = 0; l < desc.loopCount; l++) {
        CU_ASSERT(cuMemcpyAsync(desc.dst, desc.src, desc.copySize, desc.stream));
    }

    return desc.copySize;
}

size_t MemcpyInitiatorCE::getAdjustedCopySize(size_t size, CUstream stream) {
    // CE does not change/truncate buffer size
    return size;
}

size_t MemcpyInitiatorMulticastWrite::memcpyFunc(MemcpyDescriptor &desc) {
    return multicastCopy(desc.dst, desc.src, desc.copySize, desc.stream, desc.loopCount);
}

size_t MemcpyInitiatorMulticastWrite::getAdjustedCopySize(size_t size, CUstream stream) {
    size = size / sizeof(unsigned);
    return size * sizeof(unsigned);
}

size_t MemcpyInitiatorSMSplitWarp::memcpyFunc(MemcpyDescriptor &desc) {
    return copyKernelSplitWarp(desc);
}

MemPtrChaseOperation::MemPtrChaseOperation(unsigned long long loopCount) : loopCount(loopCount) {
    cudaDeviceProp prop;
    CUDA_ASSERT(cudaGetDeviceProperties(&prop, 0));
    smCount = prop.multiProcessorCount;
}

double MemPtrChaseOperation::doPtrChase(const int srcId, const MemcpyBuffer &peerBuffer) {
    double lat = 0.0;
    lat = latencyPtrChaseKernel(srcId, (void*)peerBuffer.getBuffer(), peerBuffer.getBufferSize(), latencyMemAccessCnt, smCount);
    return lat;
}
