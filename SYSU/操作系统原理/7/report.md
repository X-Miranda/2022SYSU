# **操作系统原理 实验七**

## 个人信息

【院系】计算机学院

【专业】计算机科学与技术

【学号】22336259

【姓名】   谢宇桐

## 实验题目

内存管理

## 实验目的

1. 实现二级分页机制，并能够在虚拟机地址空间中进行内存管理，包括内存的申请和释放等。
2. 实现动态分区算法。
3. 实现页面置换算法。
4. 掌握和实现虚拟内存管理技术。

## 实验要求

1. 实现二级分页机制，进行内存管理
2. 实现动态分区算法
3. 实现页面置换算法
4. 撰写实验报告。

##  实验方案

本次实验有4个任务：

一、复现参考代码，实现二级分页机制，并在虚拟机地址空间中进行内存管理，包括内存的申请和释放等。

二、实现动态分区算法

在这里使用的是`first-fit`算法，其基本思想：要求空闲区按地址递增的次序排列。当进行内存分配时，从空闲区表头开始顺序查找，直到找到第一个能滿足其大小要求的空闲区为止。分一块给请求者，余下部分仍留在空闲区中。

三、实现页面置换

在这里我们使用FIFO算法。具体来说，FIFO算法维护一个先进先出的页面队列，当一个页面需要调入主存储器时，它被加入队列的末尾。当需要置换页面时，FIFO算法选择队列头部的页面进行置换。这样，最先进入主存储器的页面总是最先被置换出去，而最后进入主存储器的页面总是保留在主存储器中。

四、复现“虚拟页内存管理”代码，结合代码分析虚拟页内存分配的三步过程和虚拟页内存释放，并构造测试例子来分析虚拟页内存管理的实现是否存在bug。



## 实验过程

一、

![企业微信截图_17185194472204](企业微信截图_17185194472204.png)

复现代码后可以得到与教程相同结果。我们也可以在下面一段显示的第三行中看见对内存大小的输出。

然后我们来实现对内存的申请和释放。利用代码里的申请内存函数，修改`setup.cpp`：

```c++
void first_thread(void *arg)
{
    // 第1个线程不可以返回
    // stdio.moveCursor(0);
    // for (int i = 0; i < 25 * 80; ++i)
    // {
    //     stdio.print(' ');
    // }
    // stdio.moveCursor(0);

    // 申请三个内核页 
    int addr[3]; 
    for (int i = 0; i < 3; ++i) 
    { 
       addr[i] = memoryManager.allocatePhysicalPages(KERNEL, 1); 
       printf("allocating 1 page, address: 0x%x\n", addr[i]); 
    }
    // 释放三个内核页 
    for (int i = 0; i < 3; ++i) 
    { 
       memoryManager.releasePhysicalPages(KERNEL, addr[i], 1); 
       printf("releasing 1 page, address: 0x%x\n", addr[i]); 
    } 
    asm_halt();
}
```

结果如下，表示申请了三个内存并释放：

![1718536244644](1718536244644.png)



二、

我们使用`first-fit`算法，直接修改`setup.cpp`即可：

```c++
void first_thread(void *arg)
{
    // 第1个线程不可以返回
    // stdio.moveCursor(0);
    // for (int i = 0; i < 25 * 80; ++i)
    // {
    //     stdio.print(' ');
    // }
    // stdio.moveCursor(0);

    char *page0 = (char *)memoryManager.allocatePhysicalPages(AddressPoolType::KERNEL, 64);
    printf("Allocated 64 pages for page0, starting at %d.\n", page0);

    char *page1 = (char *)memoryManager.allocatePhysicalPages(AddressPoolType::KERNEL, 32);
    printf("Allocated 32 pages for page1, starting at %d.\n", page1);

    char *page2 = (char *)memoryManager.allocatePhysicalPages(AddressPoolType::KERNEL, 16);
    printf("Allocated 16 pages for page2, starting at %d.\n", page2);

    char *page3 = (char *)memoryManager.allocatePhysicalPages(AddressPoolType::KERNEL, 8);
    printf("Allocated 8 pages for page3, starting at %d.\n", page3);

    memoryManager.releasePhysicalPages(AddressPoolType::KERNEL, int(page0), 64);
    printf("Released 64 pages for page0.\n");

    memoryManager.releasePhysicalPages(AddressPoolType::KERNEL, int(page2), 16);
    printf("Released 16 pages for page2.\n");

    char *page4 = (char *)memoryManager.allocatePhysicalPages(AddressPoolType::KERNEL, 16);
    printf("Allocated 16 pages for page4, starting at %d.\n", page4);

    char *page5 = (char *)memoryManager.allocatePhysicalPages(AddressPoolType::KERNEL, 16);
    printf("Allocated 16 pages for page5, starting at %d.\n", page5);


    asm_halt();
}
```

结果如下：

![1718544355964](1718544355964.png)

如图，我们进行分配内存和释放内存，再申请内存操作。因为是首次适应，所以在释放完page0和page2时申请page4和page5时，会放在原先page0地址处。由此表明，我们成功实现了首次适应动态分配算法。



三、

我们选择运用FIFO算法进行页面置换，首先修改`memory.cpp`：

```c++
#include "memory.h"
#include "os_constant.h"
#include "stdlib.h"
#include "asm_utils.h"
#include "stdio.h"
#include "program.h"
#include "os_modules.h"
int t=0;
MemoryManager::MemoryManager()
{
    initialize();
}

void MemoryManager::initialize()
{
    this->totalMemory = 0;
    this->totalMemory = getTotalMemory();
    for(int i=0;i<20;i++){
    	mem_address[i]=0;
    	mem_len[i]=0;
    	time[i]=0;
    	mem_num = 0;
    }

    // 预留的内存
    int usedMemory = 256 * PAGE_SIZE + 0x100000;
    if (this->totalMemory < usedMemory)
    {
        printf("memory is too small, halt.\n");
        asm_halt();
    }
    // 剩余的空闲的内存
    int freeMemory = this->totalMemory - usedMemory;

    int freePages = freeMemory / PAGE_SIZE;
    int kernelPages = freePages / 2;
    int userPages = freePages - kernelPages;

    int kernelPhysicalStartAddress = usedMemory;
    int userPhysicalStartAddress = usedMemory + kernelPages * PAGE_SIZE;

    int kernelPhysicalBitMapStart = BITMAP_START_ADDRESS;
    int userPhysicalBitMapStart = kernelPhysicalBitMapStart + ceil(kernelPages, 8);
    int kernelVirtualBitMapStart = userPhysicalBitMapStart + ceil(userPages, 8);

    kernelPhysical.initialize(
        (char *)kernelPhysicalBitMapStart,
        kernelPages,
        kernelPhysicalStartAddress);

    userPhysical.initialize(
        (char *)userPhysicalBitMapStart,
        userPages,
        userPhysicalStartAddress);

    kernelVirtual.initialize(
        (char *)kernelVirtualBitMapStart,
        kernelPages,
        KERNEL_VIRTUAL_START);

    printf("total memory: %d bytes ( %d MB )\n",
           this->totalMemory,
           this->totalMemory / 1024 / 1024);

    printf("kernel pool\n"
           "    start address: 0x%x\n"
           "    total pages: %d ( %d MB )\n"
           "    bitmap start address: 0x%x\n",
           kernelPhysicalStartAddress,
           kernelPages, kernelPages * PAGE_SIZE / 1024 / 1024,
           kernelPhysicalBitMapStart);

    printf("user pool\n"
           "    start address: 0x%x\n"
           "    total pages: %d ( %d MB )\n"
           "    bit map start address: 0x%x\n",
           userPhysicalStartAddress,
           userPages, userPages * PAGE_SIZE / 1024 / 1024,
           userPhysicalBitMapStart);

    printf("kernel virtual pool\n"
           "    start address: 0x%x\n"
           "    total pages: %d  ( %d MB ) \n"
           "    bit map start address: 0x%x\n",
           KERNEL_VIRTUAL_START,
           userPages, kernelPages * PAGE_SIZE / 1024 / 1024,
           kernelVirtualBitMapStart);
}

int MemoryManager::allocatePhysicalPages(enum AddressPoolType type, const int count)
{
    int start = -1;

    if (type == AddressPoolType::KERNEL)
    {
        start = kernelPhysical.allocate(count);
    }
    else if (type == AddressPoolType::USER)
    {
        start = userPhysical.allocate(count);
    }

    return (start == -1) ? 0 : start;
}

void MemoryManager::releasePhysicalPages(enum AddressPoolType type, const int paddr, const int count)
{
    if (type == AddressPoolType::KERNEL)
    {
        kernelPhysical.release(paddr, count);
    }
    else if (type == AddressPoolType::USER)
    {

        userPhysical.release(paddr, count);
    }
}

int MemoryManager::getTotalMemory()
{

    if (!this->totalMemory)
    {
        int memory = *((int *)MEMORY_SIZE_ADDRESS);
        // ax寄存器保存的内容
        int low = memory & 0xffff;
        // bx寄存器保存的内容
        int high = (memory >> 16) & 0xffff;

        this->totalMemory = low * 1024 + high * 64 * 1024;
    }

    return this->totalMemory;
}

void MemoryManager::openPageMechanism()
{
    // 页目录表指针
    int *directory = (int *)PAGE_DIRECTORY;
    //线性地址0~4MB对应的页表
    int *page = (int *)(PAGE_DIRECTORY + PAGE_SIZE);

    // 初始化页目录表
    memset(directory, 0, PAGE_SIZE);
    // 初始化线性地址0~4MB对应的页表
    memset(page, 0, PAGE_SIZE);

    int address = 0;
    // 将线性地址0~1MB恒等映射到物理地址0~1MB
    for (int i = 0; i < 256; ++i)
    {
        // U/S = 1, R/W = 1, P = 1
        page[i] = address | 0x7;
        address += PAGE_SIZE;
    }

    // 初始化页目录项

    // 0~1MB
    directory[0] = ((int)page) | 0x07;
    // 3GB的内核空间
    directory[768] = directory[0];
    // 最后一个页目录项指向页目录表
    directory[1023] = ((int)directory) | 0x7;

    // 初始化cr3，cr0，开启分页机制
    asm_init_page_reg(directory);

    printf("open page mechanism\n");
}

int MemoryManager::allocatePages(enum AddressPoolType type, const int count)
{
    // 第一步：从虚拟地址池中分配若干虚拟页
    t++;
    int virtualAddress = allocateVirtualPages(type, count);
    if (!virtualAddress){
    	printf("set %d pages fail\n",count);
        do{
            virtualAddress = allocateVirtualPages(type, count);
            if(!virtualAddress){
            	int min = 100000;
            	int min_idx = 0;
            	for(int i=0;i<mem_num;i++){
            	    if(time[i]<min){
            	    	min = time[i];
            	    	min_idx = i;
            	    }
            	}
            	releasePages(type,mem_address[min_idx],mem_len[min_idx]);
            	printf("Release a block address : %x\n",mem_address[min_idx]);
            	printf("\n");
            	for(int i=min_idx;i<mem_num-1;i++){
            	    mem_address[i] = mem_address[i+1];
            	    time[i] = time[i+1];
            	    mem_len[i] = mem_len[i+1];
            	}
            }
            else break;
        }while(1);
    }

    bool flag;
    int physicalPageAddress;
    int vaddress = virtualAddress;

    // 依次为每一个虚拟页指定物理页
    for (int i = 0; i < count; ++i, vaddress += PAGE_SIZE)
    {
        flag = false;
        // 第二步：从物理地址池中分配一个物理页
        physicalPageAddress = allocatePhysicalPages(type, 1);
        if (physicalPageAddress)
        {
            //printf("allocate physical page 0x%x\n", physicalPageAddress);

            // 第三步：为虚拟页建立页目录项和页表项，使虚拟页内的地址经过分页机制变换到物理页内。
            flag = connectPhysicalVirtualPage(vaddress, physicalPageAddress);
        }
        else
        {
            flag = false;
        }

        // 分配失败，释放前面已经分配的虚拟页和物理页表
        if (!flag)
        {
            // 前i个页表已经指定了物理页
            releasePages(type, virtualAddress, i);
            // 剩余的页表未指定物理页
            releaseVirtualPages(type, virtualAddress + i * PAGE_SIZE, count - i);
            return 0;
        }
    }
    mem_address[mem_num] = virtualAddress;
    mem_len[mem_num] = count;
    time[mem_num] = t;
    mem_num++;
    
    return virtualAddress;
}

int MemoryManager::allocateVirtualPages(enum AddressPoolType type, const int count)
{
    int start = -1;

    if (type == AddressPoolType::KERNEL)
    {
        start = kernelVirtual.allocate(count);
    }

    return (start == -1) ? 0 : start;
}

bool MemoryManager::connectPhysicalVirtualPage(const int virtualAddress, const int physicalPageAddress)
{
    // 计算虚拟地址对应的页目录项和页表项
    int *pde = (int *)toPDE(virtualAddress);
    int *pte = (int *)toPTE(virtualAddress);

    // 页目录项无对应的页表，先分配一个页表
    if(!(*pde & 0x00000001)) 
    {
        // 从内核物理地址空间中分配一个页表
        int page = allocatePhysicalPages(AddressPoolType::KERNEL, 1);
        if (!page)
            return false;

        // 使页目录项指向页表
        *pde = page | 0x7;
        // 初始化页表
        char *pagePtr = (char *)(((int)pte) & 0xfffff000);
        memset(pagePtr, 0, PAGE_SIZE);
    }

    // 使页表项指向物理页
    *pte = physicalPageAddress | 0x7;

    return true;
}

int MemoryManager::toPDE(const int virtualAddress)
{
    return (0xfffff000 + (((virtualAddress & 0xffc00000) >> 22) * 4));
}

int MemoryManager::toPTE(const int virtualAddress)
{
    return (0xffc00000 + ((virtualAddress & 0xffc00000) >> 10) + (((virtualAddress & 0x003ff000) >> 12) * 4));
}

void MemoryManager::releasePages(enum AddressPoolType type, const int virtualAddress, const int count)
{
    int vaddr = virtualAddress;
    int *pte;
    for (int i = 0; i < count; ++i, vaddr += PAGE_SIZE)
    {
        // 第一步，对每一个虚拟页，释放为其分配的物理页
        releasePhysicalPages(type, vaddr2paddr(vaddr), 1);

        // 设置页表项为不存在，防止释放后被再次使用
        pte = (int *)toPTE(vaddr);
        *pte = 0;
    }
    mem_num--;
    mem_address[mem_num] = 0;
    mem_len[mem_num] = 0;
    time[mem_num] = 0;
    // 第二步，释放虚拟页
    releaseVirtualPages(type, virtualAddress, count);
}

int MemoryManager::vaddr2paddr(int vaddr)
{
    int *pte = (int *)toPTE(vaddr);
    int page = (*pte) & 0xfffff000;
    int offset = vaddr & 0xfff;
    return (page + offset);
}

void MemoryManager::releaseVirtualPages(enum AddressPoolType type, const int vaddr, const int count)
{
    if (type == AddressPoolType::KERNEL)
    {
        kernelVirtual.release(vaddr, count);
    }
}
```

再修改`setup.cpp`：

```c++
void first_thread(void *arg)
{
    // 第1个线程不可以返回
    // stdio.moveCursor(0);
    // for (int i = 0; i < 25 * 80; ++i)
    // {
    //     stdio.print(' ');
    // }
    // stdio.moveCursor(0);

    char *p0 = (char *)memoryManager.allocatePages(AddressPoolType::KERNEL, 100);
    printf("page0 addresss : %x %d\n", p0,100);

    char *p1 = (char *)memoryManager.allocatePages(AddressPoolType::KERNEL, 100);
    printf("page1 addresss : %x %d\n", p1,100);

    char *p2 = (char *)memoryManager.allocatePages(AddressPoolType::KERNEL, 100);
    printf("page2 addresss : %x %d\n", p2,100);

    char *p3 = (char *)memoryManager.allocatePages(AddressPoolType::KERNEL, 15400);
    printf("page3 addresss : %x %d\n", p3,15400);

    char *p4 = (char *)memoryManager.allocatePages(AddressPoolType::KERNEL, 100);
    printf("page4 addresss : %x %d\n", p4,100);

    char *p5 = (char *)memoryManager.allocatePages(AddressPoolType::KERNEL, 100);
    printf("page5 addresss : %x %d\n", p5,100);

    char *p6 = (char *)memoryManager.allocatePages(AddressPoolType::KERNEL, 100);
    printf("page6 addresss : %x %d\n", p6,100);

    char *p7 = (char *)memoryManager.allocatePages(AddressPoolType::KERNEL, 100);
    printf("page7 addresss : %x %d\n", p7,100);


    asm_halt();
}
```

`memory.h`：

```c++
#ifndef MEMORY_H
#define MEMORY_H

#include "address_pool.h"

enum AddressPoolType
{
    USER,
    KERNEL
};

class MemoryManager
{
public:
    // 可管理的内存容量
    int totalMemory;
    // 内核物理地址池
    AddressPool kernelPhysical;
    // 用户物理地址池
    AddressPool userPhysical;
    // 内核虚拟地址池
    AddressPool kernelVirtual;
    
    //申请的块内存地址（最多申请20块）
    int mem_address[20];
    //申请的块内存长度
    int mem_len[20];
    int time[20];
    //申请的块内存数量
    int mem_num;

public:
    MemoryManager();

    // 初始化地址池
    void initialize();

    // 从type类型的物理地址池中分配count个连续的页
    // 成功，返回起始地址；失败，返回0
    int allocatePhysicalPages(enum AddressPoolType type, const int count);

    // 释放从paddr开始的count个物理页
    void releasePhysicalPages(enum AddressPoolType type, const int startAddress, const int count);

    // 获取内存总容量
    int getTotalMemory();

    // 开启分页机制
    void openPageMechanism();

    // 页内存分配
    int allocatePages(enum AddressPoolType type, const int count);

    // 虚拟页分配
    int allocateVirtualPages(enum AddressPoolType type, const int count);

    // 建立虚拟页到物理页的联系
    bool connectPhysicalVirtualPage(const int virtualAddress, const int physicalPageAddress);

    // 计算virtualAddress的页目录项的虚拟地址
    int toPDE(const int virtualAddress);

    // 计算virtualAddress的页表项的虚拟地址
    int toPTE(const int virtualAddress);

    // 页内存释放
    void releasePages(enum AddressPoolType type, const int virtualAddress, const int count);    

    // 找到虚拟地址对应的物理地址
    int vaddr2paddr(int vaddr);

    // 释放虚拟页
    void releaseVirtualPages(enum AddressPoolType type, const int vaddr, const int count);
};

#endif
```

别忘了`os_constant.h`：

```c++
#define BITMAP_START_ADDRESS 0x10000
```

结果如下：

![1718545354984](1718545354984.png)

我们可以看到分配 `page6`和`page7`时，内存不够了，需要进行虚拟页面的替换，第一个分配的页`page0`被替换成`page6`，`page1`被`page7`替换，至此虚拟页面替换的FIFO算法实现完毕。



四、

复现代码结果如下：

![1718547231143](1718547231143.png)

虚拟页内存分配通常包括以下三个步骤：

1. **分配虚拟内存**:
   - 在这一步中，操作系统的内存管理单元（MMU）或内存管理器会从虚拟地址空间中选择一个或多个空闲的虚拟页。这通常涉及到查找一个空闲的页表条目（PTE）或页目录条目（PDE），并将它们标记为已使用。
2. **分配物理内存**:
   - 分配虚拟页后，就为这些虚拟页分配相应的物理页。
3. **建立虚拟-物理映射**:
   - 最后一步是将分配的虚拟页与物理页关联起来，这通过更新页表条目来实现。每个虚拟页的页表条目会被设置为指向相应的物理页帧，同时确保页表条目包含正确的访问权限和状态信息。

虚拟页内存释放过程则通常包括以下步骤：

1. **断开虚拟-物理映射**:
   - 在释放虚拟页之前，需要先断开虚拟页和物理页之间的映射关系。这通常通过将相关的页表条目（PTE）清零或设置为无效状态来完成。
2. **标记物理页为空闲**:
   - 一旦虚拟-物理映射被断开，物理页就可以被标记为空闲了。这涉及到更新物理内存的位图或其它数据结构，以反映这些页现在可以被重新分配。
3. **更新内存管理数据结构**:
   - 最后，操作系统需要更新其内存管理的数据结构，以确保内存的状态是最新的。这可能包括更新任何内存使用统计信息，以及可能的内存压缩或碎片整理操作，以优化内存的使用。

在提供的代码示例中，虚拟页内存分配和释放的过程大致如下：

- **分配虚拟页** (`allocateVirtualPages`): 从内核虚拟内存池中分配虚拟页。

- **分配物理页** (`allocatePhysicalPages`): 为虚拟页分配物理页。

- **建立映射** (`connectPhysicalVirtualPage`): 将物理页地址写入页表条目，建立虚拟地址到物理地址的映射。

- **释放虚拟页** (`releaseVirtualPages`): 将虚拟页标记为空闲，释放虚拟页表条目。

- **释放物理页** (`releasePhysicalPages`): 将物理页标记为空闲，更新物理内存的位图。

- **断开映射**: 在`releasePages`函数中，通过将页表条目设置为0来断开虚拟地址和物理地址之间的映射。

  

## 实验总结

通过本次实验，我对操作系统的内存管理机制有了更深入的理解。我学会了如何实现和优化内存分配与释放策略，并通过实际代码的编写和测试，提高了我的编程和问题解决能力。

实验一中，通过复现参考代码，我实现了二级分页机制。这个过程加深了我对虚拟内存和物理内存之间映射关系的理解。在虚拟机环境中进行内存管理，我学会了如何申请和释放内存，使我对操作系统的内存管理有了更直观的认识。

在实验二中，实现`first-fit`动态分区算法时，我遇到了一些挑战。计算地址空间是一个比较关键且有挑战性的任务，且空闲区的维护需要有序，这要求我仔细考虑数据结构的选择和实现方式，说明选择合适的数据结构对于算法效率至关重要。

实验三中，我实现了FIFO页面置换算法。这个算法更加简单易懂，在之前的实验中，我们也接触过先进先出的实验。

实验四要求我复现“虚拟页内存管理”代码，并分析内存分配和释放的过程。在这个过程中，我构造了多个测试例子，以确保代码的正确性。这个过程锻炼了我的调试和测试能力，也让我意识到了在实际开发中，编写测试用例的重要性。

本次实验通过实现和分析内存管理的关键技术，使我们加深对操作系统内存管理机制的理解，包括二级分页机制、动态分区算法、页面置换算法以及虚拟页内存管理。也使我们对计算机内存这个概念有了进一步的认识。

## 参考文献

[首次适应（FirstFit）算法（空闲区地址递增）]([首次适应（FirstFit）算法（空闲区地址递增）_空闲分区以地址递增次序排列的方法-CSDN博客](https://blog.csdn.net/qq_39026548/article/details/78008326?ops_request_misc=%7B%22request%5Fid%22%3A%22171853488716800222847006%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=171853488716800222847006&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-1-78008326-null-null.142^v100^pc_search_result_base3&utm_term=firstfit算法代码&spm=1018.2226.3001.4187))

[页面置换算法（FIFO,OPT,LRU)_fifo算法-CSDN博客](https://blog.csdn.net/weixin_64066303/article/details/130780210?ops_request_misc=%7B%22request%5Fid%22%3A%22171853522916800211546501%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=171853522916800211546501&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-130780210-null-null.142^v100^pc_search_result_base3&utm_term=FIFO算法&spm=1018.2226.3001.4187)

