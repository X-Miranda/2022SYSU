<font size =6>**操作系统原理 实验四**</font>

## 个人信息

【院系】计算机学院

【专业】计算机科学与技术

【学号】22336259

【姓名】谢宇桐

## 实验题目

从实模式到保护模式

## 实验目的

1. 学习C代码变成C程序的过程。
2. 学习C/C++项目的组织方法。
3. 学习makefile的使用。
4. 学习C和汇编混合编程。
5. 学习保护模式中断处理机制。
6. 学习8259A可编程中断处理部件。
7. 学习时钟中断的处理。

## 实验要求

1. 学习混合编程基本思路
2. 使用C/C++编写内核
3. 学习对中断的处理
3. 实现时钟中断
4. 撰写实验报告

## 实验方案

本次实验有四个任务：

一、混合编程：复现例1：

- 在文件`c_func.c`中定义C函数`function_from_C`。

- 在文件`cpp_func.cpp`中定义C++函数`function_from_CPP`。

- 在文件`asm_func.asm`中定义汇编函数`function_from_asm`，在`function_from_asm`中调用`function_from_C`和`function_from_CPP`。

- 在文件`main.cpp`中调用汇编函数`function_from_asm`。

  将这4个文件统一编译成可重定位文件即`.o`文件，然后将这些`.o`文件链接成一个可执行文件。

  若我们改用make执行，则需要编写一个makefile进行统一编译并执行。

  

二、使用C/C++编写内核：复现例二，并将输出hello world改为输出自己的学号：

  1.首先先转变工作目录环境，假设项目的放置的文件夹是`project`，则各子文件夹含义如下：

- `project/build`。存放Makefile，make之后生成的中间文件如`.o`，`.bin`等会放置在这里，目的是防止这些文件混在代码文件中。
- `project/include`。存放`.h`等函数定义和常量定义的头文件等。
- `project/run`。存放gdb配置文件，硬盘映像`.img`文件等。
- `project/src`。存放`.c`，`.cpp`等函数实现的文件。

  2.在bootloader的最后加上读取内核的代码

  3.常量的定义放置在`boot.inc`中

  4.定义内核进入点在`entry.asm`中

  5.`setup_kernel`的定义在`setup.cpp`中

  6.将汇编函数放在`asm_utils.h`中，并声明所有的汇编函数

  7.进行编译运行。



三、中断处理：复现例3：

(该任务代码在第二个任务的基础上进行修改)

  1.定义一个类，称为中断管理器`InterruptManager`，其定义放在`interrupt.h`中

  2.在`include/os_type.h`定义基本的数据类型的别名

  3.在`interrupt.cpp`中初始化IDT，并调用`setInterruptDescriptor`放入256个默认的中断描述符

  4.在`asm_utils.asm`汇编代码中实现能够将IDT的信息放入到IDTR的函数`asm_lidt`，并定义一个默认的中断处理函数`asm_interrupt_empty_handler`

  5.在函数`setup_kernel.cpp`中定义并初始化中断处理器，并在`os_modules.h`中声明这个实例

  6.最后，将一些常量统一定义在`os_constant.h`下

  7.在qemu的debug模式下加载运行，并查看中断是否被正常调用



四、时钟中断：复现例4，并封装自己的类来实现时钟中断处理过程。并通过时钟中断实现一个跑马灯显示学号和英文名：

(该任务代码在第三个任务的基础上进行修改)

1.在`interrupt.cpp`中为中断控制器`InterruptManager`加入一些成员变量和函数，并初始化8259A芯片。

2.在`asm_utils.asm`的`asm_out_port`中实现对`out`指令的封装，`asm_in_port`中实现对`in`指令的封装，并实现一个完整的时钟中断处理函数

3.封装一个能够处理屏幕输出和滚屏的类`STDIO`，声明放置在`stdio.h`中

4.在`stdio.cpp`实现代码。

5.定义中断处理函数`c_time_interrupt_handler`。

6.封装开启和关闭时钟中断的函数

7.在`setup_kernel`中定义`STDIO`的实例`stdio`，最后初始化内核的组件，然后开启时钟中断和开中断。

8.在`os_modules.h`声明这个实例。

9.进行编译运行



## 实验过程

一、

在文件`cpp_func.cpp`中定义C++函数`function_from_CPP`，代码如下：

```c++
#include <iostream>

extern "C" void function_from_CPP() {
    std::cout << "This is a function from C++." << std::endl;
}
```

我们来解释一下extern "C"的用法：**在C++中，extern"C"修饰的语句是按照C语言的方式进行编译的**。

即这个`function_from_CPP()`函数，即使是在C++中，也是按照C的方法编译的，因为它被extern"C"修饰过了。

在文件`asm_func.asm`中定义汇编函数`function_from_asm`，在`function_from_asm`中调用`function_from_C`和`function_from_CPP`：

```assembly
[bits 32]
global function_from_asm
extern function_from_C
extern function_from_CPP

function_from_asm:
    call function_from_C
    call function_from_CPP
    ret
```

这里的global用来**声明本变量或函数可以被外部调用，由内用到外**

extern是用来**调用外部模块的函数或变量，导入外面模块的使用，由外用到内**

即在这里，global是为了声明`function_from_asm`可以被外部调用，而extern关键字是为了调用外部的`function_from_C`和`function_from_CPP`

编写完所有代码后将这4个文件统一编译成可重定位文件即`.o`文件，然后将这些`.o`文件链接成一个可执行文件，编译命令分别如下：

```shell
gcc -o c_func.o -m32 -c c_func.c
g++ -o cpp_func.o -m32 -c cpp_func.cpp 
g++ -o main.o -m32 -c main.cpp
nasm -o asm_func.o -f elf32 asm_func.asm
g++ -o main.out main.o c_func.o cpp_func.o asm_func.o -m32
```

运行得出：

![adbe36477874bdd4367e426c5a537d6](adbe36477874bdd4367e426c5a537d6.png)

若要改用make进行编译，则我们需要编写makefile文件，命令如下：

```
main.out: main.o c_func.o cpp_func.o asm_func.o
	g++ -o main.out main.o c_func.o cpp_func.o asm_func.o -m32

c_func.o: c_func.c
	gcc -o c_func.o -m32 -c c_func.c

cpp_func.o: cpp_func.cpp
	g++ -o cpp_func.o -m32 -c cpp_func.cpp 

main.o: main.cpp
	g++ -o main.o -m32 -c main.cpp

asm_func.o: asm_func.asm
	nasm -o asm_func.o -f elf32 asm_func.asm
clean:
	rm *.o
```

然后用`make`指令进行编译，再运行，结果如下，与直接编译运行结果一致：

![481302f34821b6ae3b830c91df22318](481302f34821b6ae3b830c91df22318.png)



二、

关键代码展示：汇编函数`asm_utils.asm`：

```assembly
[bits 32]

global asm_hello_world

asm_hello_world:
    push eax
    xor eax, eax

    mov ah, 0x03 ;青色
    mov al, 'H'
    mov [gs:2 * 0], ax

    mov al, 'e'
    mov [gs:2 * 1], ax

    mov al, 'l'
    mov [gs:2 * 2], ax

    mov al, 'l'
    mov [gs:2 * 3], ax

    mov al, 'o'
    mov [gs:2 * 4], ax

    mov al, ' '
    mov [gs:2 * 5], ax

    mov al, 'W'
    mov [gs:2 * 6], ax

    mov al, 'o'
    mov [gs:2 * 7], ax

    mov al, 'r'
    mov [gs:2 * 8], ax

    mov al, 'l'
    mov [gs:2 * 9], ax

    mov al, 'd'
    mov [gs:2 * 10], ax

    pop eax
    ret
```

文件都编译完后进入`build`文件夹下开始编译，我们首先编译MBR、bootloader。

```shell
nasm -o mbr.bin -f bin -I../include/ ../src/boot/mbr.asm
nasm -o bootloader.bin -f bin -I../include/ ../src/boot/bootloader.asm
```

其中，`-I`参数指定了头文件路径，`-f`指定了生成的文件格式是二进制的文件。

接着，我们编译`setup.cpp`。

```shell
g++ -g -Wall -march=i386 -m32 -nostdlib -fno-builtin -ffreestanding -fno-pic -I../include -c ../src/kernel/setup.cpp
```

最后我们链接生成的可重定位文件为两个文件：只包含代码的文件`kernel.bin`，可执行文件`kernel.o`。

```shell
ld -o kernel.o -melf_i386 -N entry.obj setup.o asm_utils.o -e enter_kernel -Ttext 0x00020000
ld -o kernel.bin -melf_i386 -N entry.obj setup.o asm_utils.o -e enter_kernel -Ttext 0x00020000 --oformat binary
```

链接后我们使用dd命令将`mbr.bin bootloader.bin kernel.bin`写入硬盘即可，如下所示。

```shell
dd if=mbr.bin of=../run/hd.img bs=512 count=1 seek=0 conv=notrunc
dd if=bootloader.bin of=../run/hd.img bs=512 count=5 seek=1 conv=notrunc
dd if=kernel.bin of=../run/hd.img bs=512 count=200 seek=6 conv=notrunc
```

在`run`目录下启动：

```shell
qemu-system-i386 -hda ../run/hd.img -serial null -parallel stdio -no-reboot
```

结果如下：

![270791a251b189309354a116a7e06ad](270791a251b189309354a116a7e06ad.png)



将“hello world"改为自己学号输出，我们只需要将汇编函数`asm_utils.asm`进行修改即可：

```assembly
[bits 32]

global asm_hello_world

asm_hello_world:
    push eax
    xor eax, eax

    mov ah, 0x03 ;青色
    mov al, '2'
    mov [gs:2 * 0], ax

    mov al, '2'
    mov [gs:2 * 1], ax

    mov al, '3'
    mov [gs:2 * 2], ax

    mov al, '3'
    mov [gs:2 * 3], ax

    mov al, '6'
    mov [gs:2 * 4], ax

    mov al, '2'
    mov [gs:2 * 5], ax

    mov al, '5'
    mov [gs:2 * 6], ax

    mov al, '9'
    mov [gs:2 * 7], ax


    pop eax
    ret
```

运行结果如下：

![4c9575d9eb6fa868c2f9f610f8183b7](4c9575d9eb6fa868c2f9f610f8183b7.png)



三、

按照例子和上面的实验方案进行复现后，使用`make`编译后用`make debug`在qemu的debug模式下加载运行，并在gdb下使用`x/256gx 0x8880`命令可以查看我们是否已经放入默认的中断描述符，如图所示：

![9066c22148fb03172e462638623b255](9066c22148fb03172e462638623b255.png)

这里的数字和例子不太一样是因为偏移量有些偏差，但实验结果是对的，因此我们忽略不计。

第一次做的时候进入gbd我直接使用`x/256gx 0x8880`命令，但显示全为0，如下图：

![3729ca7664cfbd4a61a92393f874a26](3729ca7664cfbd4a61a92393f874a26.png)

后面询问同学得知，因为没有让代码运行，所以其地址都显示0，所以在进入gbd中要先用c使代码运行，再ctrl+c退出运行后再使用`x/256gx 0x8880`命令即可。

需要将以上全部关掉后，再重新进行编译运行即可出现如下正常调用中断的界面（若不关掉debug界面直接进行`make run`，这个界面不会显示）：

![7934fd8c35d18a9ed305da63186af74](7934fd8c35d18a9ed305da63186af74.png)

如图显示，中断被正常调用了。



四、

关键代码展示：

在`interrupt.cpp`的中断处理函数`c_time_interrupt_handler`如下所示：

```c++
// 中断处理函数
extern "C" void c_time_interrupt_handler()
{
    // 清空屏幕
    for (int i = 0; i < 80; ++i)
    {
        stdio.print(0, i, ' ', 0x07);
    }

    // 输出中断发生的次数
    ++times;
    char str[] = "interrupt happend: ";
    char number[10];
    int temp = times;

    // 将数字转换为字符串表示
    for(int i = 0; i < 10; ++i ) {
        if(temp) {
            number[i] = temp % 10 + '0';
        } else {
            number[i] = '0';
        }
        temp /= 10;
    }

    // 移动光标到(0,0)输出字符
    stdio.moveCursor(0);
    for(int i = 0; str[i]; ++i ) {
        stdio.print(str[i]);
    }

    // 输出中断发生的次数
    for( int i = 9; i > 0; --i ) {
        stdio.print(number[i]);
    }
}
```

在`asm_utils.asm`中一个完整的时钟中断处理函数如下所示：

```assembly
asm_time_interrupt_handler:
    pushad
    
    nop ; 否则断点打不上去
    ; 发送EOI消息，否则下一次中断不发生
    mov al, 0x20
    out 0x20, al
    out 0xa0, al
    
    call c_time_interrupt_handler

    popad
    iret
```

用`make && make run`编译运行代码如下:

![92d07f75dfa224705281785ca6a9353](92d07f75dfa224705281785ca6a9353.png)

需要将我们的学号和姓名用跑马灯在时钟后显示出来，我们需要改一下中断处理函数`c_time_interrupt_handler`：

```c
extern "C" void c_time_interrupt_handler()
{
    // 清屏
    for (int i = 0; i < 80; ++i)
    {
        stdio.print(0, i, ' ', 0x07);
    }

    // 输出中断发生的次数
    ++times;
    char str[] = "interrupt happend: ";
    char name[] = "22336259 xyt" ;  //在这里创造字符串显示姓名学号
    char number[10];
    int temp = times;

    // 将数字转换为字符串表示
    for(int i = 0; i < 10; ++i ) {
        if(temp) {
            number[i] = temp % 10 + '0';
        } else {
            number[i] = '0';
        }
        temp /= 10;
    }

    // 移动光标到(0,0)输出字符
    stdio.moveCursor(0);
    for(int i = 0; str[i]; ++i ) {
        stdio.print(str[i]);
    }

    // 输出中断发生的次数
    for( int i = 9; i > 0; --i ) {
        stdio.print(number[i]);
    }
    
    int idx = times % 12;//用于确定当前要显示的姓名学号字符串的起始位置
    int pos = stdio.getCursor() + idx + 3; //计算要移动到的光标位置 pos，确保在清除了之前的内容后，光标能够正确移动到姓名学号字符串的起始位置。
    stdio.moveCursor(pos);//将光标移动到之前计算的位置 pos，准备开始打印姓名学号字符串。
    stdio.print(name[idx], idx % 10 + 2);//打印姓名学号字符串的一部分，从索引 idx 处开始，打印 idx 除以 10 的余数加上 2 个字符。这样就实现了跑马灯效果
}
```

再进行编译运行，即可成功显示姓名学号的跑马灯效果：

![1714824168328](1714824168328.png)

![1714824189175](1714824189175.png)



## 实验总结

这次的实验内容较多，涵盖了混合编程、内核开发、中断处理以及时钟中断等多个方面，且语言也包含C、C++和汇编语言等，通过逐步实现不同的任务，我初步全面了解操作系统内核的构建过程以及中断处理的机制，并且初步了解如何将它们结合起来实现系统功能。

在混合编程任务中，我学会了如何在不同的语言文件中定义函数，并且通过链接器将它们组合成一个可执行文件。并了解了extern和global关键字的用法，以及extern “C”语句的含义，学会了定义全局变量和在不同文件调用函数和变量。

在内核开发任务中，通过引导加载器加载内核，初始化各种数据结构和硬件设备，最终启动了内核。在这个过程中，我大概了解了引导加载过程、内核的组织结构以及与硬件交互的方法。

在中断处理任务中，我大概了解了如何初始化、配置中断描述符表（IDT），并实现中断处理函数。中断是操作系统中一个重要的部分，是操作系统与外部环境交互的重要方式，包括硬件设备的响应、系统调用等。在前面的实验中我们也初步接触了，可见它的重要性。

最后在时钟中断任务中，我进一步扩展了中断处理的功能，实现了一个简单的时钟中断处理器，并结合屏幕输出实现了一个跑马灯效果。展示了操作系统内核如何利用时钟中断来进行任务调度和时间管理。

通过这个实验，我提升了对计算机体系结构和底层硬件的理解。同时，我也解决了一些遇到的挑战和问题，比如调试、设计代码等方面。总的来说，这个实验覆盖全面，帮助我深入理解了操作系统内核的实现细节，并培养了解决问题和动手实践的能力。

## 参考文献

[C/C++中的 extern 和extern“C“关键字的理解和使用（对比两者的异同）](https://blog.csdn.net/m0_46606290/article/details/119973574)

[extern、static、.global的用法](https://blog.csdn.net/a1809032425/article/details/101725981)