<font size =6>**操作系统原理 实验五**</font>

## 个人信息

【院系】计算机学院

【专业】计算机科学与技术

【学号】22336259

【姓名】谢宇桐

## 实验题目

内核线程

## 实验目的

1. 学习C语言的可变参数机制的实现方法，实现可变参数机制，以及实现一个较为简单的printf函数。
2. 实现内核线程的实现。
3. 重点理解`asm_switch_thread`是如何实现线程切换的，体会操作系统实现并发执行的原理。
4. 实现基于时钟中断的时间片轮转(RR)调度算法，并实现一个线程调度算法。

## 实验要求

1. 实现了可变参数机制及printf函数。
2. 自行实现PCB，实现线程。
3. 了解线程调度，并实现一个调度算法。
4. 撰写实验报告。

## 实验方案

本次实验有四个任务：

一、实现一个能够进行基本的格式化输出的printf

​	这个任务需要我们使用C语言的可变参数机制，先定义一个具有可变参数的函数，用来输出若干个整数的函数。然后在函数内部引用可变参数列表中的参数。实现可变参数机制后实现printf即可。

二、自行设计PCB，根据你的PCB来实现线程。

​	先向内存申请一个PCB，再创建线程，实现线程调度。

三、编写若干个线程函数，使用gdb跟踪函数，观察线程调度切换时，前后栈、寄存器、PC等变化，并说明：

- 一个新创建的线程是如何被调度然后开始执行的。
- 一个正在执行的线程是如何被中断然后被换下处理器的，以及换上处理机后又是如何从被中断点开始执行的。

四、实现线程调度算法，在下面的调度算法选一执行：

- 先来先服务。
- 最短作业（进程）优先。
- 响应比最高者优先算法。
- 优先级调度算法。
- 多级反馈队列调度算法。

我选的是先来先服务，在 `program.cpp` 中编写先进先出函数，并在其他地方作出修改。



## 实验过程

一、

这个任务教程已有例子，我们只需了解其原理，了解C语言的可变参数机制，并复现即可，复现结果如下：

![企业微信截图_17149653741038](企业微信截图_17149653741038.png)

二、

第二个任务在教程中也有非常详细的例子，我们只需了解PCB的设计原理，并学会创建线程“Hello World!”，按照教程实现即可。结果如下：

![企业微信截图_17155592777161](企业微信截图_17155592777161.png)

三、

首先我们先删掉setup.cpp里的注释符号

```c++
void first_thread(void *arg)
{
   
    printf("pid %d name \"%s\": Hello World!\n", programManager.running->pid, programManager.running->name);
    if (!programManager.running->pid)
    {
        programManager.executeThread(second_thread, nullptr, "second thread", 1);
        programManager.executeThread(third_thread, nullptr, "third thread", 1);//这里原先被注释掉了，我们将注释删掉
    }
    asm_halt();
}
```

然后我们先说明一个新创建的线程是如何被调度然后开始执行的：

我们用`make && make debug` 打开gdb调试后，给源码打上断点，并运行到`setup_kernel`函数，即创建第一个线程的地方：

```shell
layout src 
b schedule 
b setup_kernel 
b c_time_interrupt_handler 
b asm_switch_thread 
c
```

![企业微信截图_17161047015973](企业微信截图_17161047015973.png)

通过代码我们看到会首先会通过 `executeThread` 方法创 建一个线程，然后通过调用 `asm_switch_thread` 函数把第一个线程调度上去执行。

再继续运行，进入`asm_switch_thread`，通过`info registers`查看寄存器状态

![企业微信截图_17161086758128](企业微信截图_17161086758128.png)

![企业微信截图_1716108795578](企业微信截图_1716108795578.png)

我们可以看到`asm_switch_thread` 函数会先将 esp 的值到线程的 `PCB::statck` 中，用做下次恢复。随后，这个 函数会将 `next->stack` 的值写入到 esp 中，从而完成线程栈的切换。到此，我们已经切换到创建的第一个线程 `first_thread`。

接下来我们看一个正在执行的线程是如何被中断然后被换下处理器的，以及换上处理机后又是如何从被中断点开始执行的。

用c指令继续运行，运行到时钟中断处理函数 `c_time_interrupt_handler`处，使用`p cur->name` `p cur->ticks`指令，可以观察到现在执行的线程是 `first_thread`，其会不断消耗分配给他的时间片。

![企业微信截图_17161120655294](企业微信截图_17161120655294.png)

接下来进入到schedule中，代码分析如下： 

```c++
void ProgramManager::schedule()
{
    bool status = interruptManager.getInterruptStatus();
    interruptManager.disableInterrupt();

    if (readyPrograms.size() == 0)
    {
        interruptManager.setInterruptStatus(status);
        return;
    }

    if (running->status == ProgramStatus::RUNNING)
    {
        running->status = ProgramStatus::READY;//首先将现在执行的线程first_thread的状态从 RUNNING 修改为 READY，并且放到就绪队列中。
        running->ticks = running->priority * 10;
        readyPrograms.push_back(&(running->tagInGeneralList));
    }
    else if (running->status == ProgramStatus::DEAD)
    {
        releasePCB(running);
    }

    ListItem *item = readyPrograms.front();
    PCB *next = ListItem2PCB(item, tagInGeneralList);
    PCB *cur = running;
    next->status = ProgramStatus::RUNNING;
    running = next;
    readyPrograms.pop_front();

    asm_switch_thread(cur, next);//接下来，调用 asm_switch_thread 函数，把 first_thread 换下处理器，然后把之前新创建的第二个线程 second_thread 换上处理器。

    interruptManager.setInterruptStatus(status);
}
```

在 `c_time_interrupt_handler` 函数中，`third_thread` 的时间片消耗完毕后，函数会调用 `schedule` 方法，进而调用 `asm_switch_thread` 函数，将线程切换回之前被换下处理器的 `first_thread`。最后，我们执行 `ret` 返回。`function` 会被加载进 `eip`，从而使得 CPU 跳转到这个 函数中执行。至此讨论完毕。



四、

在教程给的几个调度算法中，我们选择最简单的先来先服务进行实现。

在 `program.cpp` 中，编写 `FCFS_schedule()`函数：

```c++
void ProgramManager::FCFS_schedule() 
{ 
    bool status = interruptManager.getInterruptStatus(); 
    interruptManager.disableInterrupt(); 
 
    if (readyPrograms.size() == 0)  // 没有就绪程序，恢复中断并返回 
    { 
	interruptManager.setInterruptStatus(status); 
	return; 
    } 

    if (running->status == ProgramStatus::RUNNING)   // 有 RUNNING 程序，改为 READY，并放回就绪队列 
    { 
	 running->status = ProgramStatus::READY; 
	 readyPrograms.push_back(&(running->tagInGeneralList)); 
    } 

    else if (running->status == ProgramStatus::DEAD)  // 如果程序为 DEAD，释放其 PCB 
    { 
	 releasePCB(running); 
    } 
 
 // 从就绪队列中取出下一个要运行的程序 
    ListItem *item = readyPrograms.front(); 
    PCB *next = ListItem2PCB(item, tagInGeneralList); 
    PCB *cur = running; 
 // 更新下一个程序的状态为 RUNNING，并更新 running 指针 
    next->status = ProgramStatus::RUNNING; 
    running = next; 
    readyPrograms.pop_front();
// 进行线程切换 
    asm_switch_thread(cur, next); 
 
 // 恢复中断状态 
    interruptManager.setInterruptStatus(status); 
} 
```

 修改`program_exit()`函数，使线程结束时调用 `FCFS_schedule` 函数。：

```C++
void program_exit()
{
    PCB *thread = programManager.running;
    thread->status = ProgramStatus::DEAD;

    if (thread->pid)
    {
        //programManager.schedule();在此处修改
	programManager.FCFS_schedule(); 
    }
    else
    {
        interruptManager.disableInterrupt();
        printf("halt\n");
        asm_halt();
    }
}
```

`interrupt.cpp` 中的 `c_time_interrupt_handler`也需要修改：

```c++
// 中断处理函数
extern "C" void c_time_interrupt_handler()
{
    PCB *cur = programManager.running;

    if (cur->ticks)
    {
        --cur->ticks;
        ++cur->ticksPassedBy;
    }
    else
    {
        programManager.schedule();
    }
    programManager.FCFS_schedule(); //在中断到来时调用先进先出算法
}
```

最后，在program.h里加入声明：

```c++
class ProgramManager
{
public:
//...
void FCFS_schedule(); //添加声明
};

void program_exit();

#endif
```

最后运行，观察结果：

![3b7bd298e95009822338e9337af9dea](3b7bd298e95009822338e9337af9dea.png)

可以看到，符合先来先服务的思想，该调度算法完成。

## 实验总结

本次实验包括实现一个能够进行基本的格式化输出的printf、自行设计PCB并实现线程、编写多个线程函数进行调度、实现线程调度算法。通过这些任务的完成，我对操作系统的相关知识有了更深入的了解。

在第一个任务中，我们学习了C语言的可变参数机制，并通过实现printf函数加深了对该机制的理解。通过定义一个具有可变参数的函数，我们可以输出若干个整数。通过这个任务，我学会了如何使用C语言的可变参数机制来实现格式化输出。

在第二个任务中，我们自行设计PCB，并根据该PCB来实现线程。通过这个任务，我了解了如何设计PCB以及如何实现线程调度。

在第三个任务中，我们编写了多个线程函数，并使用gdb跟踪函数，观察线程调度切换时前后栈、寄存器、PC等变化。通过观察线程的切换过程，我们深入理解了线程调度的原理和机制。我个人感觉这个任务最难，有很多操作还不够熟练，还需要多练习如何debug，多学习gdb的命令。

在第四个任务中，我们实现了线程调度算法——先进先出。通过该算法，我们按照线程创建的顺序进行调度，并依次执行每个线程。通过这个任务，我学会了如何实现线程调度算法，并了解了不同调度算法的特点和适用场景。

总体而言，本次实验使我对操作系统的原理和机制有了更深入的理解。我了解了如何使用C语言的可变参数机制、如何设计PCB并实现线程、如何编写多个线程函数进行调度以及如何实现线程调度算法。这些技能对于我的编程和计算机科学知识的提升都有很大的帮助。

## 参考文献

