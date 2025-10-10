<font size =6>**操作系统原理 实验八**</font>

## 个人信息

【院系】计算机学院

【专业】计算机科学与技术

【学号】22336259

【姓名】  谢宇桐

## 实验题目

从内核态到用户态

## 实验目的

1. 理解区分内核态和用户态的必要性。
2. 编写一个系统调用，并分析系统调用前后栈的变化。
3. 掌握用户态与内核态之间切换的过程。
4. 掌握进程创建的过程。
5. 分析fork/exit/wait指令执行过程。
6. 学习如何通过分页机制隔离进程间的地址空间。

## 实验要求

1. 编写一个系统调用。
2. 掌握用户态与内核态之间切换的过程。
3. 实现fork/exit/wait指令。
3. 实现进程之间的隔离。
4. 撰写实验报告。

##  实验方案

一、

1.1、编写一个系统调用，然后在进程中调用之，并根据gdb来分析执行系统调用后的栈的变化情况、说明TSS在系统调用执行过程中的作用。

1.2、创建第一个进程，在进程中使用`printf`会发生闪退。因为我们在启动进程时设置只有特权级0下的代码才可以访问显存，而进程运行在特权级3下。解决这个问题我们可以通过系统调用的方式，从用户态进入内核态，然后在内核态将需要显示的字符写入显存，最后返回到用户态即可。

1.3、在第一个进程创建过程中，我们在找创建的PCB时存在一个风险语句，编写一个`findProgramByPid`函数使得通过`pid`找PCB。

二、实现fork函数，根据gdb来分析子进程的跳转地址、数据寄存器和段寄存器的变化，并比较上述过程和父进程执行完`ProgramManager::fork`后的返回过程的异同，最后解释fork是如何保证子进程的`fork`返回值是0，而父进程的`fork`返回值是子进程的pid。

上述实验可以通过代码逻辑和gdb调试进行：从子进程第一次被调度执行时开始，逐步跟踪子进程的执行流程一直到子进程从`fork`返回。

三、实现wait函数和exit函数，实现回收僵尸进程的有效方法。



## 实验过程

一、

我们在教程代码中的`setup.cpp`中加入对0号系统的调用：

```c++
extern "C" void setup_kernel()
{
   // 设置0号系统调用
    systemService.setSystemCall(0, (int)syscall_0);

    int ret; 
    // 测试系统调用 
    ret = asm_system_call(0, 1, 2, 3, 4, 5); 
    printf("return value: %d\n", ret); 
//...
}
```

![1719402889008](1719402889008.png)

测试有输出`return value: 15`，说明我们成功实现在进程中调用系统调用。

接下来使用gdb进行调试分析执行系统调用后的栈的变化情况。我们先用`break setup_kernel`在这里打上断点，可以看到调用前的栈帧只有 setup_kernel：

![a99db50e9e1b308eec227c407e211d4](a99db50e9e1b308eec227c407e211d4.png)

用n移到`ret = asm_system_call(0, 1, 2, 3, 4, 5);` 代码处，使用si命令进入，再用bt指令查看栈帧：

![17a7eda789d80a7339f6025c00f38cd](17a7eda789d80a7339f6025c00f38cd.png)

可以看到setup_kernel 上的 栈帧变成了 asm_system_call，说明我们在这实现了系统调用。

接下来继续使用n移到`int 0x80`处，使用si命令进入，再用bt指令查看栈帧：

![1719402656544](1719402656544.png)

我们可以看到，`setup_kernel` 上的 栈帧变成了 `asm_system_call_handler`。  这是 `int 0x80` 中断的处理函数。 

接下来继续使用n移到`call dword [system_call_table + eax * 4]`处，使用si命令进入，再用bt指令查看栈帧：

![1719402771210](1719402771210.png)

可以看到栈顶为`syscall_0`，这就是我们调用的系统函数。系统调用完成。



接下来我们根据gdb来说明TSS在系统调用执行过程中的作用。

依据要求，我们先

 TSS（Task State Segment）任务状态段描述符用于描述保存任务重要信息的系统段，权限发生变化要借用TSS。任务状态段寄存器TR的可见部分为当前任务状态段描述符的选择子，不可见部分是当前任务状态段的段基地址和段界限等信息，TR只装载一次，`TR.Base`指向的地址即TSS。

 操作系统通过TSS实现任务的挂起和恢复，在切换任务的过程中，处理器中的各寄存器的当前值会被自动地保存到TR指定的TSS中，接着下一个任务的TSS的选择子被装入TR，最后从TR所指定的TSS中取出各寄存器的值送到处理器的各寄存器中。

首先给 first_thread 打上断点并进入，我们查看当前的 TSS 信息。

![1719403105849](1719403105849.png)

我们只需关注 `esp0` 以及 `ss0`，由于当前还没有进入到系统调用， esp0 是 0；而由于我们在 `bootloader` 中放入了 SS 的选择子，ss0 是 0x10。然后在first_process打断点，进入此处：

![1719403485315](1719403485315.png)

TSS 中的 esp0 发生了变化，因为在这里中我们调用了系统调用 `asm_system_call`。



1.2

我们在`first_process`加入并实现printf：

首先，为了通过中断进入内核态，我们专门为printf写一个系统调用：

```assembly
asm_system_call_printf:
    push ebp
    mov ebp, esp

    push ebx
    push ecx
    push edx
    push esi
    push edi

    mov eax, [ebp + 2 * 4]
    mov ebx, [ebp + 3 * 4]
    mov ecx, [ebp + 4 * 4]
    mov edx, [ebp + 5 * 4]
    mov esi, [ebp + 6 * 4]
    mov edi, [ebp + 7 * 4]

    int 0x80

    pop edi
    pop esi
    pop edx
    pop ecx
    pop ebx
    pop ebp

    ret
```

再在setup.cpp中加入函数：

```c++
void first_process()
{
    asm_system_call_printf(1,"Hello World!");
    asm_system_call(0, 132, 324, 12, 124);
    asm_halt();
}
//...
int syscall_printf(char *x){
    printf("%s\n",x);
    return 0;
}
//...
extern "C" void setup_kernel()
{//...
systemService.setSystemCall(1, (int)syscall_printfstr);
    //...
}
```

这样我们就能实现printf了：

![1719411549525](1719411549525.png)



1.3

代码如下：

```c++
PCB* ProgramManager::findProgramByPid(int pid){
    ListItem *item;
    PCB *x;
    item = programManager.allPrograms.head.next;
    while (item){
        x = ListItem2PCB(item, tagInAllList);
        if(x -> pid == pid)break;
        item = item->next;
    }
    return x;
}
```

在program.h中加入：

```c++
PCB* findProgramByPid(int pid);
```

![企业微信截图_17185851591908](企业微信截图_17185851591908-1719414754470-21.png)

成功输出，说明我们函数正确。





二、

对于fork的实现，我们直接复现教程例子：

![企业微信截图_17185920994608](企业微信截图_17185920994608.png)

我们结合代码来分析fork的实现过程：

```c++
int ProgramManager::fork()
{
    bool status = interruptManager.getInterruptStatus(); //获取当前的中断状态。
	interruptManager.disableInterrupt(); //禁用中断，以确保fork操作的原子性，防止在创建进程的过程中发生中断。

    // 禁止内核线程调用
    PCB *parent = this->running;//获取当前正在运行的进程的控制块（PCB）
    if (!parent->pageDirectoryAddress)//说明当前进程没有页目录地址，可能是内核线程或其他不应该被fork的进程。
    {
        interruptManager.setInterruptStatus(status);//恢复中断状态并返回错误码-1。
        return -1;
    }

    // 创建子进程
    int pid = executeProcess("", 0);//调用executeProcess函数创建一个新的进程。参数为空字符串和0可能表示使用当前进程的映像来创建新进程
    if (pid == -1)//创建进程失败
    {
        interruptManager.setInterruptStatus(status);//子进程的状态设置为DEAD，恢复中断状态
        return -1;
    }

    // 初始化子进程
    PCB *child = ListItem2PCB(this->allPrograms.back(), tagInAllList);//从进程列表中获取新创建的子进程的控制块

    bool flag = copyProcess(parent, child);//调用copyProcess函数复制父进程的资源到子进程。这可能包括内存、寄存器状态、打开的文件等

    if (!flag)//复制失败
    {
        child->status = ProgramStatus::DEAD;//将子进程的状态设置为DEAD，恢复中断状态
        interruptManager.setInterruptStatus(status);
        return -1;
    }

    interruptManager.setInterruptStatus(status);//恢复中断状态，以保持系统的一致性。
    return pid;
}
```



接下来，我们用gdb调试来进行分析：

我们给`asm_switch_thread`和`asm_start_process`打上断点，并用c开始运行到`asm_start_process`，查看寄存器：

![1719407657445](1719407657445.png)

从这开始不断使用n和info registers进行对寄存器的查看：

![1719407704596](1719407704596.png)

如此我们可以跟踪查看到子进程的执行流程。在 asm_start_process 中， ProcessStartStack 的起始地址会被送入 esp，然后更新寄存器。最后执行 iret 后，特权级 3 的选择子被放入到段寄存器中，代码会跳转到进程的起始处执行。 并看到最后子进程的 fork 返回值为 0。

接下来我们看父进程的返回过程：

首先在 first_process 和fork处打上断点并进入，并不断n进入return pid，查看pid的值：

![1719408025536](1719408025536.png)

最后 fork 会返回 pid=2，说明当前父进程调用 fork 时，会返回子进程的 pid。

继续n以查看父进程的返回过程：

![1719408184203](1719408184203.png)

![1719408469759](1719408469759.png)

我们能看到父进程的返回过程为 asm_system_call_handler->asm_system_call->fork()->first_process

接下来看子进程的返回过程：

因为子进程的是通过 asm_start_process 启动的，所以我们在此打断点，单步执行到iret：

![1719408786329](1719408786329.png)

![1719409009866](1719409009866.png)

可以看到执行iret前后eip的值有变化，这是为了保证在执行 fork 后，父子进程从相同的点返回，asm_start_process 最后的 iret 会将上面说到的保存在 0 特权级栈的父进程的 eip 的内容送入到 eip 中。 

且在上图中，我们也能清楚的看见子进程的返回过程为：asm_start_process->asm_system_call->fork()->first_process。



我们在上面过程中已经用gdb看到了父进程的返回值，接下来我们用代码进行分析：

```c++
int ProgramManager::fork()
{
    //1.保存中断状态并禁用中断
    bool status = interruptManager.getInterruptStatus(); //获取当前的中断状态。
	interruptManager.disableInterrupt(); //禁用中断，以确保fork操作的原子性，防止在创建进程的过程中发生中断。

    // 禁止内核线程调用
    //2.检查父进程的合法性
    PCB *parent = this->running;
    if (!parent->pageDirectoryAddress)//检查当前运行的进程（parent）是否有有效的页目录地址（pageDirectoryAddress）。如果没有，说明这是一个内核线程或其他不应被 fork 的进程，此时恢复中断状态并返回错误码 -1。
    {
        interruptManager.setInterruptStatus(status);
        return -1;
    }

    // 3.创建子进程
    int pid = executeProcess("", 0);//调用executeProcess函数创建一个新的进程，该进程的优先级为 0
    if (pid == -1)//创建进程失败
    {
        interruptManager.setInterruptStatus(status);//子进程的状态设置为DEAD，恢复中断状态
        return -1;
    }

    // 4.初始化子进程的PCB
    PCB *child = ListItem2PCB(this->allPrograms.back(), tagInAllList);//从进程列表中获取新创建的子进程的控制块

    bool flag = copyProcess(parent, child);//5.调用copyProcess函数复制父进程的资源到子进程。这可能包括内存、寄存器状态、打开的文件等

    if (!flag)//复制失败
    {
        child->status = ProgramStatus::DEAD;//将子进程的状态设置为DEAD，恢复中断状态
        interruptManager.setInterruptStatus(status);
        return -1;
    }

    interruptManager.setInterruptStatus(status);//6.恢复中断状态，以保持系统的一致性。
    return pid;//7.返回子进程的 PID，如果一切顺利，fork 函数将返回新创建的子进程的 PID 给父进程。
}
```

关键点在于第五步`copyProcess` 函数中的操作：

```c++
// 复制进程0级栈
    ProcessStartStack *childpss =
        (ProcessStartStack *)((int)child + PAGE_SIZE - sizeof(ProcessStartStack));
    ProcessStartStack *parentpss =
        (ProcessStartStack *)((int)parent + PAGE_SIZE - sizeof(ProcessStartStack));
    memcpy(parentpss, childpss, sizeof(ProcessStartStack));
    // 设置子进程的返回值为0
    childpss->eax = 0;
```

- 在复制父进程的栈到子进程的栈之后，`childpss->eax` 被设置为 0。这是因为在 x86 架构中，`eax` 寄存器通常用于存储函数的返回值。
- 子进程的初始化代码 `asm_start_process` 将使用这个栈来启动新进程的执行。当子进程开始执行时，它将返回到这个栈所设置的地址，并且 `eax` 寄存器的值将是 0。

父进程在 fork 调用结束后会直接得到子进程的 PID，因为 fork 函数的返回值已经在父进程的上下文中设置好了。



**三、**

exit：

```c++
void ProgramManager::exit(int ret)
{
    // 关中断，确保退出过程不会被其他中断或进程切换打断。
    interruptManager.disableInterrupt();
    
    // 第一步，标记PCB状态为`DEAD`并放入返回值，表示该进程将不再被调度执行。
    PCB *program = this->running;
    program->retValue = ret;
    program->status = ProgramStatus::DEAD;

    int *pageDir, *page;
    int paddr;

    // 第二步，如果PCB标识的是进程，则释放进程所占用的物理页、页表、页目录表和虚拟地址池bitmap的空间。
    if (program->pageDirectoryAddress)
    {
        pageDir = (int *)program->pageDirectoryAddress;
        for (int i = 0; i < 768; ++i)//通过双重循环遍历页目录表和页表，释放所有被标记为已使用的物理页。
        {
            if (!(pageDir[i] & 0x1))
            {
                continue;
            }

            page = (int *)(0xffc00000 + (i << 12));

            for (int j = 0; j < 1024; ++j)
            {
                if(!(page[j] & 0x1)) {
                    continue;
                }

                paddr = memoryManager.vaddr2paddr((i << 22) + (j << 12));
                memoryManager.releasePhysicalPages(AddressPoolType::USER, paddr, 1);
            }

            paddr = memoryManager.vaddr2paddr((int)page);//将虚拟地址转换为物理地址，然后调用
            memoryManager.releasePhysicalPages(AddressPoolType::USER, paddr, 1);
        }

        memoryManager.releasePages(AddressPoolType::KERNEL, (int)pageDir, 1);//释放页表页和页目录表页
        
        int bitmapBytes = ceil(program->userVirtual.resources.length, 8);
        int bitmapPages = ceil(bitmapBytes, PAGE_SIZE);

        memoryManager.releasePages(AddressPoolType::KERNEL, (int)program->userVirtual.resources.bitmap, bitmapPages);

    }

    // 第三步，立即执行线程/进程调度，以启动另一个就绪状态的进程。
    schedule();
}
```

![企业微信截图_17185922568173](企业微信截图_17185922568173.png)

进程退出后能够隐式地调用exit和此时的exit返回值是0的原因：
**进程正常退出**：

- 在代码中，`exit` 函数被设计为接收一个整数参数 `ret`，这个参数表示进程的退出状态码。当进程需要正常退出时，它可以调用 `exit` 函数并传递一个状态码给父进程。如果进程是正常退出，通常状态码是 0。

**隐式调用**：

- 代码中的 `program_exit` 函数在进程的栈中被设置为进程启动的返回地址。这意味着当进程执行的 `load_process` 函数完成时，控制流将返回到 `program_exit`。这可以视为一种隐式调用，因为 `program_exit` 是在进程启动时设置的，而不是在进程的主体代码中显式调用。

**退出状态码设置为 0**：

- 在 `load_process` 函数中，用户栈被设置，其中 `userStack[1]` 和 `userStack[2]` 被初始化为 0。这意味着如果进程没有显式地调用 `exit` 并传递一个退出状态码，当 `program_exit` 被调用时，它将使用栈中的 0 作为退出状态码。

**系统调用约定**：

- 在许多操作系统中，`exit` 函数通常遵循特定的系统调用约定，其中 `eax` 寄存器用于传递退出状态码。在代码中，`childpps->eax` 被设置为 0，这符合这种约定，并将导致子进程的 `fork` 返回值为 0。

综上所述，进程退出后隐式调用 `exit` 是因为进程启动时栈已经被设置，而 `exit` 返回值是 0 是因为进程没有显式地传递其他退出状态码，且栈中的默认值被用作退出状态码。这种设计确保了进程可以干净地退出，并且父进程可以接收到子进程的退出状态。



wait：

```c++
int ProgramManager::wait(int *retval)
{
    PCB *child;
    ListItem *item;
    bool interrupt, flag;

    while (true)
    {
        interrupt = interruptManager.getInterruptStatus();//禁用中断，确保 wait 函数在执行过程中不会被其他中断或进程切换打断。
        interruptManager.disableInterrupt();

        item = this->allPrograms.head.next;

        // 查找子进程
        flag = true;
        while (item)//进入一个无限循环，不断检查是否存在已经终止的子进程。
        {//使用 item 指针遍历 allPrograms 列表，查找所有属于当前运行进程（running）的子进程
            child = ListItem2PCB(item, tagInAllList);
            if (child->parentPid == this->running->pid)
            {
                flag = false;
                if (child->status == ProgramStatus::DEAD)
                {
                    printf("child %d status DEAD\n",child->pid);//找到了，退出循环
                    break;
                }
            }
            item = item->next;
        }

        if (item) // 找到一个可返回的子进程
        {
            if (retval)//如果 retval 指针非空，则将子进程的退出状态码 retValue 复制到 retval 指向的位置。
            {
                *retval = child->retValue;
            }

            int pid = child->pid;
            releasePCB(child);//释放子进程的 PCB。
            interruptManager.setInterruptStatus(interrupt);//恢复中断状态，并返回子进程的 PID。
            return pid;
        }
        else //在所有子进程中没有找到已经终止的子进程
        {
            if (flag) // 子进程已经返回
            {
                
                interruptManager.setInterruptStatus(interrupt);
                return -1;
            }
            else // 存在子进程，但子进程的状态不是DEAD
            {
                interruptManager.setInterruptStatus(interrupt);
                schedule();
            }
        }
    }
}
```

![企业微信截图_17185923053171](企业微信截图_17185923053171.png)

僵尸进程：如果一个父进程先于子进程退出，那么子进程在退出之前会被称为孤儿进程。子进程在退出后，从状态被标记为`DEAD`开始到被回收，子进程会被称为僵尸进程。

教程中的代码已经存在回收僵尸进程的方法，我们将它标注说明一下，并在找到僵尸进程和释放PCB时给出输出，让其更直观：

```c++
void ProgramManager::releasePCB(PCB *program)
{
    int index = ((int)program - (int)PCB_SET) / PCB_SIZE;
    PCB_SET_STATUS[index] = false;
    this->allPrograms.erase(&(program->tagInAllList));
    printf("Release child %d PCB\n",program->pid);//输出release标注
}
```



```c++
int ProgramManager::wait(int *retval)
{
    //...
        // 查找子进程
        flag = true; //表示尚未找到目标子进程
        while (item)
        {
            child = ListItem2PCB(item, tagInAllList);//将列表项转换为进程控制块
            if (child->parentPid == this->running->pid)//检查当前进程是否是正在运行的进程的子进程。
            {
                flag = false;
                if (child->status == ProgramStatus::DEAD)
                {
                    printf("child %d status DEAD\n",child->pid);//是则输出进程状态
                    break;   //跳出循环
                }
            }
            item = item->next;//继续遍历进程列表
        }

        if (item) // 找到一个可返回的子进程
        {
            if (retval)
            {
                *retval = child->retValue;//将子进程的返回值赋值给retval指针指向的内存地址。
            }

            int pid = child->pid;
            releasePCB(child);//释放子进程的PCB
            interruptManager.setInterruptStatus(interrupt);//恢复中断状态
            return pid;
        }
        else //没有找到僵尸进程
        {
            //...
            }
        }
    }
}
```

结果如下：

![1719060109096](1719060109096.png)

僵尸进程回收完成。




## 实验总结

​	不得不说这次的实验难度很大且繁琐，但因此学到了很多。

​	通过实验一我学到了如何从用户态切换到内核态，以及系统调用的实现原理。理解了系统调用号的分配和系统调用服务例程的作用，理解了函数调用过程中栈帧的分配和回收，也了解到任务状态段（TSS）在任务切换中的作用。

​	通过实验二我深入理解了`fork`系统调用的工作机制，理解了子进程如何获得独立的地址空间和执行环境。学习了`fork`系统调用如何保证在子进程中返回0，而在父进程中返回子进程的PID，并理解了操作系统如何调度进程，以及上下文切换过程中对寄存器和栈的保存与恢复。

​	通过实验三我学习了`exit`函数如何正确终止一个进程，包括资源的清理和退出状态的设置。理解了父进程如何等待子进程结束，并进行资源回收，防止僵尸进程的产生。

​	通过本次实验，我们不仅学习了操作系统中进程管理的关键概念，如系统调用、TSS、fork、wait和exit，而且还通过GDB调试深入理解了这些概念在实际代码中的实现和工作机制。实验过程中遇到的问题和挑战，如进程的栈变化、特权级切换、资源回收等，都促使我更深入地思考和探索。我发现操作系统设计中的许多细节问题需要仔细考虑，如栈的变化、上下文保存、特权级管理等。这些问题的解决不仅需要对操作系统原理的理解，还需要对硬件和汇编语言有一定的掌握。我也更加了解了gdb进行debug的一些重要步骤和环节，运用此工具更加熟练。

## 参考文献

[TSS 任务状态段描述符](https://blog.dsb.ink/2022/04/15/study/os/tss/#:~:text=TSS 任务状态段描述符 TSS（Task State,Segment）任务状态段描述符用于描述保存任务重要信息的系统段，权限发生变化要借用TSS。 任务状态段寄存器TR的可见部分为当前任务状态段描述符的选择子，不可见部分是当前任务状态段的段基地址和段界限等信息，TR只装载一次， TR.Base 指向的地址即TSS。 操作系统通过TSS实现任务的挂起和恢复，在切换任务的过程中，处理器中的各寄存器的当前值会被自动地保存到TR指定的TSS中，接着下一个任务的TSS的选择子被装入TR，最后从TR所指定的TSS中取出各寄存器的值送到处理器的各寄存器中。)