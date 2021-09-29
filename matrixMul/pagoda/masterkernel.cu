#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "../../common/para.h"
#include "../../common/para.cuh"
#include "kernel.cuh"
#include "runtime.cuh"

__global__ void masterKernel(volatile int *done, volatile int *totalScheTasks, volatile gTaskStruct *gTaskPool)
{
	int warpIdxx = (threadIdx.x / warpSize);
	__shared__ volatile int barID;						 // the ID for bar.sync
	__shared__ volatile int smStartIndx;				 // the start pointer for free memory region of shared memory
	__shared__ volatile int doneCtr[BP_NUM];			 // number of warp in a task
	__shared__ volatile gWarpStruct warpPoolDev[BP_NUM]; // warpPool
	int taskPointer;									 //local pointer of task table
	int taskStartP;										 //global pointer of task table
	__shared__ volatile int barIDArray[syncNum];
	__shared__ volatile int sharedTree[SH_TREE_SIZE]; //shared mem data structure
	__shared__ volatile int warpCtr;
	__shared__ volatile int warpId;
	__shared__ volatile int exit;
	extern __shared__ volatile int shared_mem[];
	int i;
	int threadDone;

	// Init warp pool
	if ((threadIdx.x & 0x1f) != 0)
		warpPoolDev[(threadIdx.x & 0x1f)].exec = 0;
	else
		warpPoolDev[(threadIdx.x & 0x1f)].exec = -1;

	taskPointer = 0;
	exit = 0;
	doneCtr[(threadIdx.x & 0x1f)] = 0;

	__threadfence_block();

	// MTB 中第一个 warp, 也就是 scheduler warp
	// scheduler warp 中的每一个线程负责 tasktable column 中的一个 entry
	if (threadIdx.x < warpSize)
	{
		while (!(*done))
		{
			// 其实就是二维数组转成一维数组的后的下标
			// 行号为 taskPointer，列号为 blockIdx.x
			taskStartP = (taskPointer * BK_NUM + blockIdx.x);
			__threadfence_block();
			// 当前Task不是第一个Task，即readyId不是-1
			if (gTaskPool[taskStartP].readyId != -1 && doneCtr[taskPointer] == 0)
			{
				// 当前Task的上一个Task
				if (gTaskPool[gTaskPool[taskStartP].readyId].done == 1)
				{
					barID = -1;
					smStartIndx = -1;
					doneCtr[taskPointer] = gTaskPool[gTaskPool[taskStartP].readyId].thread *
										   gTaskPool[gTaskPool[taskStartP].readyId].block / warpSize;
					warpCtr = doneCtr[taskPointer];
					warpId = 0;
					//parallel scheduling
					while (1)
					{
						threadDone = 1;
						if (threadIdx.x > 0)
						{
							threadDone = 0;
							if (warpPoolDev[threadIdx.x].exec == 0)
							{
								// warpCtr中保存着还需要调度的warp数量
								if (atomicSub((int *)&warpCtr, 1) > 0)
								{
									warpPoolDev[threadIdx.x].warpId = atomicAdd((int *)&warpId, 1) * warpSize;
									// global pointer of task table
									warpPoolDev[threadIdx.x].bufferNum = gTaskPool[taskStartP].readyId;
									warpPoolDev[threadIdx.x].SMindex = smStartIndx; // shared mem. index
									warpPoolDev[threadIdx.x].barId = barID;			// index of threadblock
																					// num. of thread
									warpPoolDev[threadIdx.x].threadNum = gTaskPool[gTaskPool[taskStartP].readyId].thread;
									warpPoolDev[threadIdx.x].taskId = taskPointer; // local pointer of task table
									__threadfence_block();						   // To make sure the exec. is worked after fence
									warpPoolDev[threadIdx.x].exec = 1;
									__threadfence_block();
								} // End atomic
							}
						}
						// 当前Task所需的warp已经全部分配完了
						if (warpCtr <= 0)
							threadDone = 1;
						// 保证 scheduler warp 中的32个线程都全部执行完毕
						if (__all(threadDone == 1) != 0)
						{
							break;
						}

					} // End while(1)

					//gTaskPool[taskStartP-1].done = 0; // reset flag whenever task scheduling has been done
					gTaskPool[gTaskPool[taskStartP].readyId].done = 0;
					gTaskPool[taskStartP].readyId = -1;

				} // End if ready flag
			}

			// taskPoint即行号，行号++，循环该Column的每一行
			taskPointer++; // renew the local pointer of task table
			if (taskPointer == BP_NUM)
				taskPointer = 0;

		} // End while done
		exit = 1;
		__threadfence_block();
	} // End if thread < 32

#if 1
	// executor warp
	else
	{
		//while(!(*done)){
		while (!exit)
		{
			// 当前 executor warp 的 exec 标志位被设置
			// 意味着该 executor warp 已经被 scheduler warp 安排了任务，需要执行
			if (warpPoolDev[warpIdxx].exec == 1)
			{
				// kernel running here
#if 1
				mult_gpu((float *)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[0],
						 (float *)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[1],
						 (float *)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[2],
						 (int)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[3],
						 (int)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[4],
						 warpPoolDev[warpIdxx].warpId);
#endif
				// 每个 executor warp 中的第一个线程负责在当前 warp 执行完毕后，将计数器的值减一
				// 该计数器统计当前 task 未完成的 warp 数量，当计数器的值为0时，该 task 执行完毕
				if ((threadIdx.x & 0x1f) == 0)
				{
					// 如果当前 warp 是该 task 的最后一个warp，那么该 warp 执行完成后要处理一些最后的工作
					if (atomicSub((int *)&doneCtr[warpPoolDev[warpIdxx].taskId], 1) == 1)
					{ // when all warps in a task have been done
						__threadfence_system();
						gTaskPool[warpPoolDev[warpIdxx].bufferNum].ready = 0; //unset the ready flag

						atomicAdd((int *)totalScheTasks, 1); //update the global task counter
					}

					warpPoolDev[warpIdxx].exec = 0;
					__threadfence_block();

				} // End if threadIdx.x
			}	  // End if exec
		}		  // End while done
	}			  // End else
#endif
}
