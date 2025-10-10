<font size =6>**操作系统原理 实验三**</font>

## 个人信息

【院系】计算机学院

【专业】计算机科学与技术

【学号】22336259

【姓名】   谢宇桐

## 实验题目

从实模式到保护模式

## 实验目的

1. 从16位的实模式跳转到32位的保护模式，然后在平坦模式下运行32位程序。
2. 学习如何使用I/O端口和硬件交互，为后面保护模式编程打下基础。

## 实验要求

1. 从外存中加载程序到内存中运行，用LBA方式读写硬盘。
2. 从实模式进入到保护模式。
3. 在保护模式下运行自定义的汇编程序。
4. 撰写实验报告。

## 实验方案

本次实验需要完成三个任务。

一、关于bootloader。加载

1.1、复现例一

要求：将输出Hello World的代码放入到bootloader中，然后在MBR中加载bootloader到内存，并跳转到bootloader的起始地址执行。

方案：我们在`mbr.asm`处放入使用LBA模式读取硬盘的代码，然后在MBR中加载bootloader到地址0x7e00。将bootloader读取到起始位置为0x7e00的内存后，我们执行远跳转到0x7e00。编译运行。

1.2、切换读取方式

要求：将1.1中LBA28读取硬盘的方式换成CHS读取，同时给出逻辑扇区号向CHS的转换公式。

方案：BIOS提供了实模式下读取硬盘的中断，不需要关心具体的I/O端口，只需要给出逻辑扇区号对应的磁头（Heads）、扇区（Sectors）和柱面（Cylinder）即可



二、进入保护模式

复现例二，在bootloader中进入保护模式，并在进入保护模式后在显示屏上输出`protect mode`。并使用gdb在进入保护模式的4个重要步骤上设置断点，并查看寄存器状态。

需要我们先将常量定义在一个独立的文件boot.inc中。重写bootloader和改造mbr，再进行调试。



三、在保护模式后执行自定义的汇编程序

改造lab2中汇编小程序代码为32位代码，在保护模式下运行。

包括：硬件或虚拟机配置方法、软件工具与作用、方案的思想、相关原理、程序流程、算法和数据结构、程序关键模块，结合代码与程序中的位置位置进行解释。不得抄袭，否则按作弊处理。

## 实验过程

一、

1.1

按照例子编辑`bootloader.asm`和`mbr.asm`代码，然后用下面的命令分别编译运行。

将`bootloader.asm`写入硬盘起始编号为1的扇区，共有5个扇区：

```
nasm -f bin bootloader.asm -o bootloader.bin
dd if=bootloader.bin of=hd.img bs=512 count=5 seek=1 conv=notrunc
```

`mbr.asm`重新编译并写入硬盘起始编号为0的扇区：

```
nasm -f bin mbr.asm -o mbr.bin
dd if=mbr.bin of=hd.img bs=512 count=1 seek=0 conv=notrunc
```

使用qemu运行：

```
qemu-system-i386 -hda hd.img -serial null -parallel stdio 
```

运行结果如下：

![e719c76ff897a2d93888a935506b538](e719c76ff897a2d93888a935506b538.png)



1.2

将读取硬盘的LBA28方式改为CHS模式，我们需要把`mbr.asm`文件进行改造。

查阅资料得知，LBA与CHS转换公式如下：

**逻辑编号(即LBA地址)=(柱面编号x磁头数+磁头编号)x扇区数+扇区编号-1**

（磁头数为硬盘磁头的总数，扇区数为每磁道的扇区数）

进而推导出从LBA计算CHS的公式：

**S = (LBA + 1) % 每磁道扇区数**

**H = (LBA + 1) / 每磁道扇区数 % 总磁头数**

**C = (LBA + 1) / 每磁道扇区数 / 总磁头数**

（LBA+1可以表示第几个扇区，因为LBA是从0开始编号的）

代码如下：

```assembly
org 0x7c00
[bits 16]
xor ax, ax ; eax = 0
; 初始化段寄存器, 段地址全部设为0
mov ds, ax
mov ss, ax
mov es, ax
mov fs, ax
mov gs, ax

; 初始化栈指针
mov sp, 0x7c00
mov ax, 1                ; 逻辑扇区号第0~15位
mov cx, 0                ; 逻辑扇区号第16~31位
mov bx, 0x7e00           ; bootloader的加载地址
load_bootloader:
    call asm_read_hard_disk  ; 读取硬盘
    inc ax
    cmp ax, 5
    jle load_bootloader
jmp 0x0000:0x7e00        ; 跳转到bootloader

jmp $ ; 死循环

asm_read_hard_disk: 

  mov ch, 0 ; 柱面号
  mov dh, 0 ; 磁头号
  mov dl, 80h
  
  mov cl, al ; lba
  inc cl ; 扇区=lba+1

  mov ah, 02h
  mov al, 1
  int 13h  ;int13h中断
  add bx, 512 ; 缓冲区首地址+=512
  ret 

times 510 - ($ - $$) db 0
db 0x55, 0xaa

```

再用1.1的`bootloader.asm`和此`mbr1_2.asm`文件用1.1中后面的命令编译写进磁盘即可。运行结果与1.1一致：

![e246f3b1da30f962ca8cd2b0e4b3616](e246f3b1da30f962ca8cd2b0e4b3616-1713450284740-2.png)



二、

首先先按照教程复现例二，编译`bootloader2.asm`和`mbr2.asm`文件并将它们写入磁盘中。但是需注意，在这里我们用的磁盘不能再是之前的磁盘，若是接着用同一个磁盘，会错误显示，前面的部分bootloader会显示并遮盖bootloader2中的显示，导致错误。所以我们应创建新的`hd2.img`磁盘，编译并写入`bootloader2.asm`和`mbr2.asm`文件，即可得到正确运行：

![039d8170134fa585974d4b2136e4754](039d8170134fa585974d4b2136e4754.png)

运行成功后，我们使用gdb进行调试

首先我们需要换nasm版本。在Ubuntu 18.04虚拟机上，nasm默认是2.13版本，但如果我们使用了`2.13`版本的nasm生成的文件来debug，那么我们无法在`src`窗口下显示汇编代码，所以我们需要2.15版本。

换完版本后，我们先生成符号表：

```
nasm -o mbr2.o -g -f elf32 mbr2.asm 
```

然后我们为可重定位文件`mbr2.o`指定起始地址`0x7c00`，分别链接生成可执行文件`mbr2.symbol`和`mbr2.bin`。在这里可能会有警告报错，但并不影响实验完成：

```
ld -o mbr2.symbol -melf_i386 -N mbr2.o -Ttext 0x7c00
ld -o mbr2.bin -melf_i386 -N mbr2.o -Ttext 0x7c00 --oformat binary
```

注意，在`bootloader2.asm`中，我们需要将第二行代码注释，否则调试时会报错，即：

```assembly
%include "boot.inc"
;org 0x7e00  这行代码需要注释
[bits 16]
mov ax, 0xb800
;...
```

后续操作也一样：

```
nasm -o bootloader2.o -g -f elf32 bootloader2.asm 
ld -o bootloader2.symbol -melf_i386 -N bootloader2.o -Ttext 0x7e00
ld -o bootloader2.bin -melf_i386 -N bootloader2.o -Ttext 0x7e00 --oformat binary
```

然后将它们写入和上面一样的磁盘位置，使用qemu加载hd2.img，就可以开始debug了。

根据教程设置四个断点并进行debug，运行出来结果如下：

第一个断点：

![a6b7414d82daf57067fa23c3b6e1a52](a6b7414d82daf57067fa23c3b6e1a52.png)

使用`si`进行单步指令运行，并查看寄存器状态：

![dd084744d99f2eb4efcb6a9acda678b](dd084744d99f2eb4efcb6a9acda678b.png)

第二个断点：

![171e10c80d16e7451d32e28f408f800](171e10c80d16e7451d32e28f408f800-1713543051289-3.png)

第三个断点：

![1713544800313](1713544800313.png)

第四个断点及GDT内容：

![1713544930581](1713544930581.png)

三、

在这个任务中，我们需将小程序的显示代码接到`bootloader2.asm`的`output_protect_mode_tag`后并做些修改：

1.扩展寄存器：我们需要将原本16为的小程序代码改为32位，所以我们需要改变一些寄存器号

2.去掉终端：因为**在保护模式下不能直接调用中断**，所以我们需要注释掉原来的终端调用

最终具体代码如下：

```assembly
%include "boot.inc"
org 0x7e00
[bits 16]

mov ax, 3; clear the screen
int 10h

mov ax, 0xb800
mov gs, ax
mov ah, 0x03 ;青色
mov ecx, bootloader_tag_end - bootloader_tag
xor ebx, 60;ebx
mov esi, bootloader_tag
output_bootloader_tag:
    mov al, [esi]
    mov word[gs:bx], ax
    inc esi
    add ebx,2
    loop output_bootloader_tag

;空描述符
mov dword [GDT_START_ADDRESS+0x00],0x00
mov dword [GDT_START_ADDRESS+0x04],0x00  

;创建描述符，这是一个数据段，对应0~4GB的线性地址空间
mov dword [GDT_START_ADDRESS+0x08],0x0000ffff    ; 基地址为0，段界限为0xFFFFF
mov dword [GDT_START_ADDRESS+0x0c],0x00cf9200    ; 粒度为4KB，存储器段描述符 

;建立保护模式下的堆栈段描述符      
mov dword [GDT_START_ADDRESS+0x10],0x00000000    ; 基地址为0x00000000，界限0x0 
mov dword [GDT_START_ADDRESS+0x14],0x00409600    ; 粒度为1个字节

;建立保护模式下的显存描述符   
mov dword [GDT_START_ADDRESS+0x18],0x80007fff    ; 基地址为0x000B8000，界限0x07FFF 
mov dword [GDT_START_ADDRESS+0x1c],0x0040920b    ; 粒度为字节

;创建保护模式下平坦模式代码段描述符
mov dword [GDT_START_ADDRESS+0x20],0x0000ffff    ; 基地址为0，段界限为0xFFFFF
mov dword [GDT_START_ADDRESS+0x24],0x00cf9800    ; 粒度为4kb，代码段描述符 

;初始化描述符表寄存器GDTR
mov word [pgdt], 39      ;描述符表的界限   
lgdt [pgdt]
      
in al,0x92                         ;南桥芯片内的端口 
or al,0000_0010B
out 0x92,al                        ;打开A20

cli                                ;中断机制尚未工作
mov eax,cr0
or eax,1
mov cr0,eax                        ;设置PE位
      
;以下进入保护模式
jmp dword CODE_SELECTOR:protect_mode_begin

;16位的描述符选择子：32位偏移
;清流水线并串行化处理器
[bits 32]           
protect_mode_begin:                              

mov eax, DATA_SELECTOR                     ;加载数据段(0..4GB)选择子
mov ds, eax
mov es, eax
mov eax, STACK_SELECTOR
mov ss, eax
mov eax, VIDEO_SELECTOR
mov gs, eax

mov ecx, protect_mode_tag_end - protect_mode_tag
mov ebx, 80 * 2 + 58
mov esi, protect_mode_tag
mov ah, 0x3
output_protect_mode_tag:
    mov al, [esi]
    mov word[gs:ebx], ax
    add ebx, 2
    inc esi
    loop output_protect_mode_tag
    
    


;------------------
;print name section
mov ah, 0x03 
mov al, '2'
mov [gs:2 * 30], ax

mov al, '2'
mov [gs:2 * 31], ax

mov al, '3'
mov [gs:2 * 32], ax

mov al, '3'
mov [gs:2 * 33], ax

mov al, '6'
mov [gs:2 * 34], ax

mov al, '2'
mov [gs:2 * 35], ax

mov al, '5'
mov [gs:2 * 36], ax

mov al, '9'
mov [gs:2 * 37], ax

mov al, 'X'
mov [gs:2 * 39], ax

mov al, 'i'
mov [gs:2 * 40], ax

mov al, 'e'
mov [gs:2 * 41], ax

mov al, 'Y'
mov [gs:2 * 43], ax

mov al, 'u'
mov [gs:2 * 44], ax

mov al, 't'
mov [gs:2 * 45], ax

mov al, 'o'
mov [gs:2 * 46], ax

mov al, 'n'
mov [gs:2 * 47], ax

mov al, 'g'
mov [gs:2 * 48], ax

;mov ah, 0x00                     
;int 16h

;mov ax, 3; 清屏                  将这两行代码去掉，保护模式下不能直接调用中断
;int 10h

mov bl,0x47     ;set the color
mov bh,0


flag_r dw 1
flag_c dw 1; saving for whether add or sub
row dw 0
col dw 2
;----字符弹射------
loop: ;
    mov cx, [row]
    mov dx, [col]; initialize the register

;------------------------------------
;boundary check
    push ax
    cmp dx, 0
        je boundary_colomn_0
    cmp dx, 79
        je boundary_colomn_80
boundary_colomn_end:
    cmp cx, 0
        je boundary_row_0 
    cmp cx, 23
        je boundary_row_24
boundary_row_end:
    pop ax

;------------------------------------
; move the cursor
    add cx, [flag_r]
    add dx, [flag_c]
    mov ah, dl
    mov al, cl
    and al, 0x7
    add al, 0x30
    mov [row], cx; update to the memory
    mov [col], dx

;------------------------------------
; print the cursor
    imul bx, cx, 80
    add bx, dx ; cor = 80 * row + col
    imul bx, 2    
    mov [gs:bx], ax

;-----------------------------------
; print the character symmetrically
    mov bx, 24
    sub bx, cx
    mov cx, bx
    mov bx, 80
    sub bx, dx
    mov dx, bx

    imul bx, cx, 80
    add bx, dx ; cor = 80 * row + col
    imul bx, 2    
    mov [gs:bx], ax
; -----------------------
; this section is used for delay
;therefore it can print the character slower
    push cx
    mov cx, 0x7fff
loop_delay_1:   
    push cx     
    mov cx, 0x007f
    loop_delay_2:
        loop loop_delay_2
    pop cx
    loop loop_delay_1
    pop cx

    jmp loop




boundary_row_0:
    mov ax, 1
    mov [flag_r], ax
    jmp boundary_row_end
boundary_colomn_0:
    mov ax, 1
    mov [flag_c], ax
    jmp boundary_colomn_end
boundary_row_24:
    mov ax, -1
    mov [flag_r], ax
    jmp boundary_row_end
boundary_colomn_80:
    mov ax, -1
    mov [flag_c], ax
    jmp boundary_colomn_end
loop_end:

jmp $ ; 死循环

pgdt dw 0
     dd GDT_START_ADDRESS

bootloader_tag db 'run bootloader'
bootloader_tag_end:

protect_mode_tag db 'enter protect mode'
protect_mode_tag_end:

```

结果如下：

![fd4db8a3278d05ad1bccb61b93fe4f1](fd4db8a3278d05ad1bccb61b93fe4f1.png)

（在这里的小程序弹射时快时慢，并不是代码有问题。在虚拟机运行过程中可能出现运行进程较多导致运行卡顿等情况）

包括：主要工具安装使用过程及截图结果、程序过程中的操作步骤、测试数据、输入及输出说明、遇到的问题及解决情况、关键功能或操作的截图结果。不得抄袭，否则按作弊处理。

## 实验总结

这次实验涉及了从实模式到保护模式的转换，以及在保护模式下执行自定义的汇编程序。这次实验和上次的lab2实验让我进一步了解到实模式和保护模式的切换以及保护模式的运行。

实验一我们先复现了一个简单的例子，然后通过 MBR 加载 bootloader 到内存并执行。在个过程中，我们了解了如何在引导扇区中加载代码，并且明白了如何利用 BIOS 提供的 LBA 模式读取硬盘数据。进一步，我们切换了读取方式，将 LBA28 读取硬盘的方式换成了 CHS 读取，同时给出了逻辑扇区号向 CHS 的转换公式。这一步让我们加深了对磁头、扇区和柱面的理解，以及学习了LBA28和CHS的转换。

实验二中我们进入了保护模式，并通过使用 GDB 在进入保护模式的关键步骤设置断点进行调试，学习了调试方法以及加深了对计算机底层运行机制的理解。

实验三我们将实验中的汇编小程序改造为32位代码，在保护模式下成功运行。更加加深了我们对保护模式的理解。

本次实验中遇到的问题较多，且我认为debug难度最大，但也收获颇丰，尤其加深了对保护模式的理解。

## 参考文献

[CHS和LBA的换算小记]([CHS和LBA的换算小记 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/608292324))

