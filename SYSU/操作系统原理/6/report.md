**操作系统原理 实验六**

## 个人信息

【院系】计算机学院

【专业】计算机科学与技术

【学号】22336259

【姓名】谢宇桐

## 实验题目

并发与锁机制

## 实验目的

1. 学习自旋锁以及使用自旋锁实现信号量，并在原子指令层面理解其实现过程。
2. 实现自定义的锁机制。
3. 学习使用信号量机制解决同步互斥问题。
4. 学习死锁的解决方法。

## 实验要求

1. 复现自旋锁和信号量机制。
2. 实现自定义的锁机制。
3. 掌握几种经典的同步互斥问题解决方法。
4. 撰写实验报告。

## 实验方案

本次实验需要完成三个部分：

1.复现教程中自旋锁和信号量的实现方法解决消失的芝士汉堡问题，并再实现一个与本教程的实现方式不完全相同的锁机制。我们选择教程中提到的原子质量`bts`和`lock`前缀来实现锁机制。

其工作原理如下：

`bts`（Bit Test and Set）指令是用于处理器中的位操作，其测试并设置指定的位（Bit） 为 1。  

`bts` 指令首先会测试在指定的位地址上的位值（即该位是 1 还是 1）。然后，无论该位原来的值是什么，`bts` 都会将其设置为 1。在执行操作后，`bts` 指令还会影响处理器的标志寄存器。具体来讲，它会设置进位标志（Carry Flag，CF）为该位操作前的值。如果操作前该位是 1，那么 CF 将被设置为 1；如果是 0，那么 CF 将被设置为 0。这允许我们得到该位在被设置之前的状态。 

`lock` 前缀确保指令的执行是原子的，即在多核处理器环境中，保证指令的操作在完成之前不会被其他处理器的操作干扰。

2.不使用任何同步互斥的工具创建多个线程来模拟生产者-消费者问题，这会导致线程产生冲突，呈现这个场景并解决。在这里我们模拟读者写者冲突的场景，通过对输出的更新展现错误场景，并使用信号量加锁、解锁逻辑解决这个问题

3.创建多个线程来模拟哲学家就餐的场景，并结合信号量来实现理论课上关于哲学家就餐问题的方法，但此方法可能导致死锁，演示此场景并解决它。

理论课中的哲学家就餐问题解决方法：

·哲学家 i(i ∈ 1,2,3,4,5) 会持续尝试进餐；

·哲学家 i 准备进餐时，会尝试拿起两侧的筷子；

·如果可以拿起筷子，则开始吃饭。否则等待；

·如果成功拿起筷子吃饭，吃完饭放下两侧筷子。 

导致死锁的场景：若所有哲学家同时饥饿，并都拿起左边的筷子，那么当哲学家试图拿起右边的筷子时，会被永远推迟，也即产生死锁。

我们采用以下方案解决： 单号哲学家先拿起左边的筷子，再拿起右边的筷子；双号的哲学家反之。

包括：硬件或虚拟机配置方法、软件工具与作用、方案的思想、相关原理、程序流程、算法和数据结构、程序关键模块，结合代码与程序中的位置位置进行解释。不得抄袭，否则按作弊处理。

## 实验过程

一、

1.1 我们将教程的代码依次复现即可得：

![企业微信截图_17166452471412](企业微信截图_17166452471412.png)

在没有加锁时，可以看到所有的汉堡都被儿子偷吃了。也就是说这里的`cheese_burger`是共享的，而我们并没有采取任何措施来协调线程之间对这个共享变量的访问顺序。因此会产生race condition。

![企业微信截图_17166455409476](企业微信截图_17166455409476.png)

上锁之后，可以看到汉堡成功存活。

用信号量也可以达到同样的效果：

![企业微信截图_17166477744204](企业微信截图_17166477744204.png)

1.2

```assembly
global asm_atomic_exchange
;......
asm_atomic_exchange:
    push ebp
    mov ebp, esp
    pushad

    ; 获取参数 
    mov ecx, [ebp + 4 * 2] ; register 
    mov ebx, [ebp + 4 * 3] ; memory 
    mov edx, 0 
 
    ; 设置 memory 的第 0 位为 1 
    lock bts [ebx], edx 
 
    ; 取出 CF 标志位到 dl 
    setc dl 
    mov [ecx], edx ; 作为结果修改 register 

    popad
    pop ebp
    ret
```

这样写就能交换 key 和 bolt（即 register 和 memory）的值，实现自旋锁的目的。

别忘了在`asm_utils.h` 加上声明：

```assembly
extern "C" void my_asm_atomic_exchange(uint32 *reg, uint32 *mem); 
```

结果如下，可见汉堡没被偷吃，证明我们实现成功：

![2c302c5f3e2a24cd2a50f915983f77d](2c302c5f3e2a24cd2a50f915983f77d.png)



二、

2.1

在`setup.cpp`中创建多个线程，并在全局变量中补充：

```c++
char *quote1 = "Quote 1" ; 
char *quote2 = "Quote 2" ; 
char *quote3 = "Quote 3" ; 

//......

void readFirstQuote(void *arg) 
{ 
 printf("First Quote: %s\n", quote1); 
} 
 
void readSecondQuote(void *arg) 
{ 
 printf("Second Quote: %s\n", quote2); 
} 
 
void readThirdQuote(void *arg) 
{ 
 printf("Third Quote: %s\n", quote3); 
} 
 
void writeThirdQuote(void *arg) 
{ 
 // 写等待时间，大于 RRschedule 的时间片 
 int wait = 0x3f3f3f3f; 
 while(wait) wait--; 
 quote3 = "Quote 3 new"; 
} 

void writeSecondQuote(void *arg) 
{ 
 // 写等待时间，大于 RRschedule 的时间片 
 int wait = 0x3f3f3f3f; 
 while(wait) wait--; 
 quote2 = "Quote 2 new"; 
} 

void first_thread(void *arg) 
{ 
 // 第 1 个线程不可以返回 
 stdio.moveCursor(0); 
 for (int i = 0; i < 25 * 80; ++i) 
 { 
 stdio.print(' '); 
 } 
 stdio.moveCursor(0); 
 
cheese_burger = 0;
    aLock.initialize();

    programManager.executeThread(a_mother, nullptr, "second thread", 1);
    programManager.executeThread(a_naughty_boy, nullptr, "third thread", 1);

 //模拟读错误 
 //创建线程读3 条记录 
 programManager.executeThread(readFirstQuote, nullptr, "fourth thread", 1); 
 programManager.executeThread(readSecondQuote, nullptr, "fifth thread", 1); 
 programManager.executeThread(readThirdQuote, nullptr, "sixth thread", 1); 
 //创建线程，修改第 2 条和第 3 条记录为较长内容 
 //由于写时间较长，写线程运行时间大于 RRschedule 的 time quantum 
 programManager.executeThread(writeSecondQuote, nullptr, "seventh thread", 1); 
 programManager.executeThread(writeThirdQuote, nullptr, "eighth thread", 1); 
 //创建线程读第 2 条和第 3 条记录 
 programManager.executeThread(readSecondQuote, nullptr, "ninth thread", 1); 
 programManager.executeThread(readThirdQuote, nullptr, "tenth thread", 1); 
 
 asm_halt(); 
} 
```

![1717350027101](1717350027101.png)

可以看到结果没有读到修改后的`Quote 2 new`、`Quote 3 new`，而是输出了初始项 

2.2 

首先我们先把教程中用信号量解决的方法写入`sync.cpp`中：

```c++
//......
Semaphore::Semaphore()
{
    initialize(0);
}

void Semaphore::initialize(uint32 counter)
{
    this->counter = counter;
    semLock.initialize();
    waiting.initialize();
}

void Semaphore::P()
{
    PCB *cur = nullptr;

    while (true)
    {
        semLock.lock();
        if (counter > 0)
        {
            --counter;
            semLock.unlock();
            return;
        }

        cur = programManager.running;
        waiting.push_back(&(cur->tagInGeneralList));
        cur->status = ProgramStatus::BLOCKED;

        semLock.unlock();
        programManager.schedule();
    }
}

void Semaphore::V()
{
    semLock.lock();
    ++counter;
    if (waiting.size())
    {
        PCB *program = ListItem2PCB(waiting.front(), tagInGeneralList);
        waiting.pop_front();
        semLock.unlock();
        programManager.MESA_WakeUp(program);
    }
    else
    {
        semLock.unlock();
    }
}
```

再对setup.cpp的线程创建过程进行修改：

```c++
//在每次读写操作进行加锁解锁操作
void readFirstQuote(void *arg) 
{ 
 // 加锁 
 semaphore.P(); 
 printf("First Quote: %s\n", quote1); 
 // 解锁 
 semaphore.V(); 
} 
 
void readSecondQuote(void *arg) 
{ 
 // 加锁 
 semaphore.P(); 
 printf("Second Quote: %s\n", quote2); 
 // 解锁 
 semaphore.V(); 
} 
 
void readThirdQuote(void *arg) 
{  
 semaphore.P(); 
 printf("Third Quote: %s\n", quote3); 
 semaphore.V(); 
} 

void writeSecondQuote(void *arg) 
{ 
 semaphore.P(); 
 int wait = 0x3f3f3f3f; 
 while(wait) wait--; 
 quote2 = "Quote 2 new"; 
 semaphore.V(); 
} 
 
void writeThirdQuote(void *arg) 
{ 
 semaphore.P(); 
 int wait = 0x3f3f3f3f; 
 while(wait) wait--; 
 quote3 = "Quote 3 new"; 
 semaphore.V(); 
} 

void first_thread(void *arg) 
{  
 stdio.moveCursor(0); 
 for (int i = 0; i < 25 * 80; ++i) 
 { 
 stdio.print(' '); 
 } 
 stdio.moveCursor(0); 
 
cheese_burger = 0;
    //aLock.initialize();
    semaphore.initialize(1); //将创建第一个线程开始时将其初始化为1
    programManager.executeThread(a_mother, nullptr, "second thread", 1);
    programManager.executeThread(a_naughty_boy, nullptr, "third thread", 1);
    
printf("---------\n");

 //......
 asm_halt(); 
} 
```

![1717350915451](1717350915451.png)

可以看到结果中成功输出了修改后的`Quote 2 new`、`Quote 3 new`，说明我们成功解决了线程间的读写者冲突。



三、

在`setup.cpp`中进行修改：

```c++
Semaphore chop[5]; //先声明无双筷子
//......
void philosopher(int left_chop, int right_chop) 
{ 
 // 哲学家持续尝试进餐 
 for(;;) 
 { 
 // 思考 
 printf("Philosopher at %d thinking.\n", left_chop + 1); 
 chop[left_chop].P(); // 尝试拿起左边的筷子 
 chop[right_chop].P(); // 尝试拿起右边的筷子 
 
 // 吃饭 
 printf("Philosopher at %d eating with chopsticks %d and %d\n", 
left_chop + 1, left_chop + 1, right_chop + 1); 
 
 int wait = 0x3f3f3f3f; 
 while(wait)wait--; 
 
 chop[left_chop].V(); // 放下左边的筷子 
 chop[right_chop].V(); // 放下右边的筷子 
 } 
} 
//五个哲学家的行动
void first_philosopher(void *arg) { 
 philosopher(0, 1); 
} 
 
void second_philosopher(void *arg) { 
 philosopher(1, 2); 
} 
 
void third_philosopher(void *arg) { 
 philosopher(2, 3); 
} 
 
void fourth_philosopher(void *arg) { 
 philosopher(3, 4); 
} 
 
void fifth_philosopher(void *arg) { 
 philosopher(4, 0); 
}

void first_thread(void *arg) 
{ 
 stdio.moveCursor(0); 
 for (int i = 0; i < 25 * 80; ++i) 
 { 
 stdio.print(' '); 
 } 
 stdio.moveCursor(0); 
 
 // 初始化信号量 
 for (int i = 0; i < 5; ++i) { 
 chop[i].initialize(1); 
 } 
 
 // 创建五个哲学家线程 
 programManager.executeThread(first_philosopher, nullptr, "first philosopher", 1); 
 programManager.executeThread(second_philosopher, nullptr, "second philosopher", 1); 
 programManager.executeThread(third_philosopher, nullptr, "third philosopher", 1); 
 programManager.executeThread(fourth_philosopher, nullptr, "fourth philosopher", 1); 
 programManager.executeThread(fifth_philosopher, nullptr, "fifth philosopher", 1); 
 
 asm_halt(); 
} 

```

![1717351430992](1717351430992.png)

与实际推算符合，当哲学家1吃饭时，哲学家2和5须等待，当哲学家3吃饭时，哲学家2和4必须等待，以此类推。

3.2

方案中提到的死锁问题实现：

```c++
void philosopher(int left_chop, int right_chop) {
 for(;;) {  
 printf("Philosopher at %d thinking.\n", left_chop + 1); 
 chop[left_chop].P(); // 尝试拿起左边的筷子 
 printf("Philosopher at %d taking chopsticks %d\n", left_chop + 1, left_chop + 1); 
 
 int wait = 0x3f3f3f3f; //在哲学家拿起左边的筷子后，添加等待时间，只要等待时间足够长，哲学家们的操作就会趋近于同时进行，相当于五个哲学家同时拿起左边的筷子。 

 while(wait)wait--; 
 
 chop[right_chop].P(); // 尝试拿起右边的筷子 
 printf("Philosopher at %d taking chopsticks %d\n", left_chop + 1, right_chop + 1); 
 
 // 吃饭 
 printf("Philosopher at %d eating with chopsticks %d and %d\n", left_chop + 1, left_chop + 1, right_chop + 1); 
 
 wait = 0x3f3f3f3f;  
 while(wait)wait--; 
 
 chop[left_chop].V(); // 放下左边的筷子 
 chop[right_chop].V(); // 放下右边的筷子 
 } 
} 
```

![1717351805004](1717351805004.png)

由图可以看见每个哲学家都拿起了左边的筷子，但没有右边的筷子，导致不能吃饭而死锁。

解决时依然修改setup.cpp：

```c++
void philosopher(int left_chop, int right_chop, int index) { 
    for(;;){  
	printf("Philosopher %d thinking.\n", index + 1); 
 
	int wait; 
 
	// 偶数哲学家 
	if (index % 2 == 0) { 
 		chop[left_chop].P(); // 先拿左边的筷子 
		printf("Philosopher %d taking left chopstick %d\n", index + 1, left_chop + 1); 
 
		wait = 0x3f3f3f3f; 
		while(wait) wait--; 
 
 		chop[right_chop].P(); // 再拿右边的筷子 
 		printf("Philosopher %d taking right chopstick %d\n", index + 1, right_chop + 1); 
 
 		wait = 0x3f3f3f3f; 
 		while(wait) wait--; 
 	} 
 	// 奇数哲学家 
 	else { 
 		chop[right_chop].P(); // 先拿右边的筷子 
 		printf("Philosopher %d taking right chopstick %d\n", index + 1, right_chop + 1); 
 
 		wait = 0x3f3f3f3f; 
 		while(wait) wait--; 
 
 		chop[left_chop].P(); // 再拿左边的筷子 
 		printf("Philosopher %d taking left chopstick %d\n", index + 1, left_chop + 1); 
 
 		wait = 0x3f3f3f3f; 
 		while(wait) wait--; 
 	} 
 

 	printf("Philosopher %d eating with chopsticks %d and %d\n", index + 1, left_chop + 1, right_chop + 1); 
 
 	wait = 0x3f3f3f3f; 
	while(wait)wait--; 
 
 	chop[left_chop].V(); 
 	chop[right_chop].V(); 
    } 

} 

void first_philosopher(void *arg) { 
 philosopher(0, 1, 0); 
} 
 
void second_philosopher(void *arg) { 
 philosopher(1, 2, 1); 
} 
 
void third_philosopher(void *arg) { 
 philosopher(2, 3, 2);                                                         
} 
 
void fourth_philosopher(void *arg) { 
 philosopher(3, 4, 3); 
} 
 
void fifth_philosopher(void *arg) { 
 philosopher(4, 0, 4); 
}
```

![1717352153762](1717352153762.png)

可以看到，2 4哲学家成功吃上饭，破除死锁，解决完成。

## 实验总结

本次实验主要涉及三个部分：自旋锁和信号量的实现，解决生产者-消费者问题以及哲学家就餐问题。

在第一部分中，我们使用自旋锁和信号量解决了消失的芝士汉堡问题。自旋锁使用了原子指令`bts`和`lock`前缀来实现。通过这种方式，我们可以保证在多核处理器环境中，指令的操作在完成之前不会被其他处理器的操作干扰。

在第二部分中，我们不使用任何同步互斥的工具来创建多个线程，模拟了生产者-消费者问题。通过使用信号量来控制对共享资源的访问，我们可以保证每次只有一个线程访问资源，避免了冲突。在实验代码输出中，我本意用输出”------“分割线来使线程读写操作更明显，却发现它会最先出现，不在原芝士汉堡调用之后。咨询同学后得知线程调度是随机的，出现的位置比较任意。

在第三部分中，我们使用了信号量来解决哲学家就餐问题。哲学家就餐问题中存在潜在的死锁场景，即所有哲学家同时饥饿并且拿起左边的筷子，导致无法获取右边的筷子，从而产生死锁。为了解决这个问题，我们采用了一个方案：单号哲学家先拿起左边的筷子，再拿起右边的筷子；双号的哲学家则反之。通过这种策略，我们避免了所有哲学家同时只拿起同一侧筷子的情况，从而避免了死锁的发生。

通过实验，我们了解到了不同线程之间的竞态条件和死锁问题，并学会了使用各种同步机制来解决这些问题。这些同步机制对于并发编程非常重要，可以确保多个线程之间的正确协作和资源共享。我们更加深入理解了并发编程的挑战以及如何应对和解决这些挑战。

## 参考文献

[BTS指令](https://blog.csdn.net/misterliwei/article/details/3950973)

[并发相关指令](https://zhuanlan.zhihu.com/p/674786168)

[经典的哲学家就餐问题](https://blog.csdn.net/vincent_wen0766/article/details/108695806)