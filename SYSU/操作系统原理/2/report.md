<font size =6>**操作系统原理 实验二**</font>

## 个人信息

【院系】计算机学院

【专业】计算机科学与技术

【学号】22336259

【姓名】   谢宇桐

## 实验题目

计算机开机启动及中断调用

## 实验目的

1. 熟悉计算机开机过程并编写代码。
1. 熟悉操作系统中断机制，通过汇编语言进行实现。
2. 掌握基础汇编语言，可以使用汇编语言写简单的程序。
3. 掌握处理器寻址方式。
4. 掌握常用的寄存器用法。
5. 熟练掌握qemu+gdb调试方法。

## 实验方案

此次实验包含四个任务：

1.编写MBR代码，使计算机显示写定位置和颜色的hello world和学号

2.探索实模式中断，利用中断完成光标位置获取、输出学号和实现键盘输入并回显

3.将伪代码转换为汇编代码，运行assignment程序输出hello world

4.编写字符弹射小程序，并使用键盘中断使其先显示姓名学号再显示小程序

本次实验都需要用到汇编代码、汇编代码转换为对应二进制

`nasm -f bin mbr.asm -o mbr.bin #用mbr文件作例子`

将二进制文件写入磁盘

`qemu-img create [filename] [size] #创建磁盘`

`dd if=[filename] of=[name].img bs=512 count=1 seek=0 conv=notrunc #将文件写入磁盘首扇区` 

再用qemu模拟虚拟机启动

`qemu-system-i386 -hda [name].img -serial null -parallel stdio `



## 实验过程

一.1.1

我们只需要复现教程例子，将例子创建为mbr.asm文件，根据方案后半部分转换为二进制代码，将文件写入磁盘首扇区，运行qemu即可（后续都是这四步操作）

![8f0576f49bc36413a0364ebeb2b9faf](8f0576f49bc36413a0364ebeb2b9faf.png)

1.2

此任务需要我们在使得MBR被加载到0x7C00后在(12,12)处开始输出我的学号，w在1.1的代码的基础上在输出hello world后面加

```assembly
mov ah, 0x03 ;颜色选为亮蓝色
mov al, '2'
mov [gs:2 * (12*80 + 12)], ax 

mov al, '2'
mov [gs:2 * (12*80 + 13)], ax

;...显示自己学号

mov al, '9'
mov [gs:2 * (12*80 + 19)], ax
```

`mov [gs:2 * (12*80 + 12)], ax` 将寄存器ax中的值存储到计算得到的内存地址中。`[gs:2 * (12*80 + 13)]`  是内存地址的表示方式。`gs`  是段寄存器，用于指定内存段的起始地址。 `2 * (12*80 + 13)`  是偏移量，通过计算得到具体的内存地址。`ax` 是16位寄存器，存储着一个16位的值。

将其创建为mbr2.asm，并四步操作后成功显示：

![09b8a80814ffc961cfa3368a7f3f450](09b8a80814ffc961cfa3368a7f3f450.png)



二、2.1

这个任务需要我们请探索实模式下的光标中断，利用中断实现光标的位置获取和光标的移动

编写2_1.asm代码如下：

```assembly
org 0x7c00
[bits 16]
xor ax, ax ; eax = 0
; 初始化段寄存器, 段地址全部设为0
mov ds, ax
mov ss, ax
mov es, ax
; 初始化栈指针
mov sp, 0x7c00

mov ah,0x02    ;功能号
mov bh,0x00    ;第0页
mov dh,0x05    ;dh中放行号，这里光标位置设为5行12列
mov dl,0x12    ;dl中放列号
int 0x10

mov ah,0x03 ; 输入 3 子功能是获取光标位置，需要存入 ah 寄存器
mov bh,0   ; bh 寄存器是待获取的光标的页号

 int 0x10 ; 输出： ch=光标开始行，cl=光标结束行
    			; dh = 光标所在行号，dl= 光标所在列号
```

由此，经过四步后显示如下（图中最后一行Disk上方闪烁的光标即为光标位置）：

![e52023945cdfa8f99d3a9aee259c146](C:/Users/85013/Documents/WeChat Files/wxid_9juhscjdbrkj22/FileStorage/Temp/e52023945cdfa8f99d3a9aee259c146.png)



但任务并不是一帆风顺的，在刚开始时，我的代码初始化部分和任务1.1是一样的，即：

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
mov ax, 0xb800
mov gs, ax
```

（由于提交限制，就不放图了）运行结果虽然成功显示了光标的位置，却也把任务一的部分显示出来了。在咨询了同学后才知道**`mov ax 0xb800`和`mov gs ax`这两个初始化是直接操作到显存了**，而我们需要用实模式中断输出，将它们删除后即可恢复正常显示。



2.2

此任务需要我们修改1.2的代码，使用实模式下的中断来输出你的学号。代码修改为2_2.asm如下：

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

mov ah, 0x0E ; 功能号，显示字符
mov al, '2' ; 要显示的字符
mov bh, 0 ; 页号
mov bl, 0x07 ; 字符颜色
int 0x10

mov ah, 0x0E ; 功能号，显示字符
mov al, '2' ; 要显示的字符
mov bh, 0 ; 页号
mov bl, 0x07 ; 字符颜色
int 0x10

mov ah, 0x0E ; 功能号，显示字符
mov al, '3' ; 要显示的字符
mov bh, 0 ; 页号
mov bl, 0x07 ; 字符颜色
int 0x10

mov ah, 0x0E ; 功能号，显示字符
mov al, '3' ; 要显示的字符
mov bh, 0 ; 页号
mov bl, 0x07 ; 字符颜色
int 0x10

mov ah, 0x0E ; 功能号，显示字符
mov al, '6' ; 要显示的字符
mov bh, 0 ; 页号
mov bl, 0x07 ; 字符颜色
int 0x10

mov ah, 0x0E ; 功能号，显示字符
mov al, '2' ; 要显示的字符
mov bh, 0 ; 页号
mov bl, 0x07 ; 字符颜色
int 0x10

mov ah, 0x0E ; 功能号，显示字符
mov al, '5' ; 要显示的字符
mov bh, 0 ; 页号
mov bl, 0x07 ; 字符颜色
int 0x10

mov ah, 0x0E ; 功能号，显示字符
mov al, '9' ; 要显示的字符
mov bh, 0 ; 页号
mov bl, 0x07 ; 字符颜色
int 0x10

jmp $ ; 无限循环

times 510-($-$$) db 0 ; 填充剩余空间为0
dw 0xAA55 ; 结束标志

```

结果显示如下

![f1cfc5803e4d6d482f75c378f485f61](f1cfc5803e4d6d482f75c378f485f61.png)

但我们可以发现上面这个代码比较繁琐，有很多重复，所以我想或许可以将所有一样的提一起，即改为下面这样，将功能号、字符颜色、页号提前：

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

mov ah, 0x0E ; 功能号，显示字符
mov bl, 0x07 ; 字符颜色
mov bh, 0 ; 页号

mov al, '2' ; 要显示的字符
int 0x10

mov al, '2' ; 要显示的字符
int 0x10

mov al, '3' ; 要显示的字符
int 0x10

mov al, '3' ; 要显示的字符
int 0x10

mov al, '6' ; 要显示的字符
int 0x10

mov al, '2' ; 要显示的字符
int 0x10

mov al, '5' ; 要显示的字符
int 0x10

mov al, '9' ; 要显示的字符
int 0x10

jmp $ ; 无限循环

times 510-($-$$) db 0 ; 填充剩余空间为0
dw 0xAA55 ; 结束标志

```

也能得到正确运行。



2.3

这个任务需要我们在2.1和2.2的基础上探索实模式的键盘中断，利用键盘中断实现键盘输入并回显，代码如下：

```assembly
org 0x7c00
[bits 16]
xor ax, ax ; eax = 0
; 初始化段寄存器, 段地址全部设为0
mov ds, ax
mov ss, ax
mov es, ax
; 初始化栈指针
mov sp, 0x7c00

main_loop:

 mov ah, 0x00 ; 等待并读取按键（阻塞模式），等待用户按下一个键，然后返回按键的 ASCII 码在 AL 寄存器中，扫描码在 AH 寄存器中。

 int 16h ; 功能号，调用键盘中断

 mov ah, 0x0E ; TTY 模式下输出字符
 int 10h ; 调用视频中断来回显字符
 
 jmp main_loop 
 
times 510 - ($ - $$) db 0 
db 0x55, 0xAA 
```

得到结果，我们打出 ”how are you" 来看看 ：

![4db0af411eb71e50a4874d7f5b4375b](4db0af411eb71e50a4874d7f5b4375b.png)

得到正确结果。



三、

在这个任务中首先需要把教程中给的代码转换为伪代码，并按格式将其写为student.asm文件，student.asm文件如下：

```assembly
; If you meet compile error, try 'sudo apt install gcc-multilib g++-multilib' first

%include "head.include"
; you code here

your_if:
; put your implementation here
    mov eax, [a1] 
    cmp eax, 12
    jl lt12       ; a1 < 12
    cmp eax, 24
    jl lt24       ; 12 <= a1 < 24
                  ; a1 >= 24 
    mov eax, [a1]
    shl eax, 4    ; a1 * 16 
    jmp end_if 
lt12:
    mov eax, [a1]
    shr eax, 1    ; a1 / 2 
    add eax, 1    ; a1 + 1 
    jmp end_if 
lt24:
    mov eax, [a1]
    mov ebx, 24
    sub ebx, eax  ; 24 - a1 
    imul ebx, eax ; (24 - a1) * a1 
    mov eax, ebx 
    jmp end_if 
end_if:
    mov [if_flag], eax


your_while:
; put your implementation here
    mov ebx, [a2]             
    cmp ebx, 12
    jl end_while
loop:
    call my_random        
 ; 计算 while_flag 数组的偏移地址
    sub ebx, 12               ; 计算偏移量（a2 - 12）
    mov ecx, [while_flag]   
    mov [ecx + ebx], al       ; while_flag[ebx] = while_flag[a2 - 12] <- al 
    dec ebx                   ; a2--
    mov [a2], ebx             ; 更新 a2 
    cmp ebx, 12
    jge loop                  ; 如果 a2>=12， 继续循环
end_while:

%include "end.include"

your_function: 
    pusha                   

    ;xor ecx, ecx            ; 循环计数器清零
    mov edi, [your_string]
    
loop1:
    mov al, [edi]
    test al, al             ; 检查当前字符是否为'\0'
    jz end_loop         
    
    push eax           
    call print_a_char       ; 不需要手动压栈。c语言会自动压栈
    ; print_a_char会使用eax寄存器。记得push/pop 保存和恢复
    pop eax
    
    inc edi                 ; 循环计数器自增
    jmp loop1

end_loop:
    popa                    ; 恢复所有寄存器的值
    ret                     ; 返回

```

用下列命令将student.asm文件转换为student.o文件：

`nasm -f elf32 student.asm -o student.o`

将其放进assignment中，输入make run使其运行，运行结果如下：

![1711872751378](1711872751378.png)



四、汇编小程序

这个任务需要我们编写一个字符弹射程序，其从点(2,0)(2,0)处开始向右下角45度开始射出，遇到边界反弹，反弹后按45度角射出，方向视反弹位置而定。额外任务需要我们先显示学号姓名，然后使用键盘回显再显示字符弹射程序。

程序如下：

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
mov ax, 0xb800
mov gs, ax

mov ax, 3; 清屏
int 10h

;------------------
;print name section 居中显示姓名学号
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

mov ah, 0x00 ;键盘回显
int 16h      ;将这两行即下面两行去掉即为原本任务

mov ax, 3    ;清屏
int 10h

mov dl, 2      
mov dh, 0      
mov ah, 02h
int 10h       


flag_r dw 1
flag_c dw 1
row dw 0
col dw 2
;----字符弹射------
loop: ;
    mov cx, [row]
    mov dx, [col]

;------------------------------------

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

times 510 - ($ - $$) db 0
db 0x55, 0xaa

```

最后执行文件，我们可得结果：

![1711875529580](1711875529580.png)

以上是原本任务，额外任务如下：

![1711875450899](1711875450899.png)

先显示姓名学号，按键后显示：

![1711875488231](1711875488231.png)



## 实验总结

在完成这个实验的过程中，我深入了解了计算机的开机启动过程以及中断调用的机制。通过编写汇编代码，我成功实现了四个任务，并且掌握了相关的工具和调试方法。

首先，在编写MBR代码的过程中，我学会了如何编写简单的引导程序，并实现了在屏幕上显示指定位置和颜色的文本信息。这个过程中，我深入理解了计算机启动时的引导过程以及硬件相关的知识。

其次，通过探索实模式中断，我了解了中断的概念以及在实模式下如何利用中断完成一些功能，比如获取光标位置、输出信息以及实现键盘输入和回显。这一部分的实践让我更加熟悉了操作系统中断机制的原理和实现方式。

接着，将伪代码转换为汇编代码并运行assignment程序输出hello world，这个任务让我对汇编语言的语法和编程风格有了更深入的了解。通过将伪代码转换为汇编代码，我能够更清晰地理解程序的运行逻辑和算法设计。

最后，编写字符弹射小程序，并使用键盘中断使其先显示姓名学号再显示小程序 。这个任务让我学会了如何处理键盘中断以及如何编写简单的字符动画程序。通过这个实践，我不仅提高了对汇编语言的熟练程度，还锻炼了解决问题的能力和编程思维。

在本次实验中，我遇到了很多问题，都已在上述实验过程中展出。例如不知道功能号的用法，不清楚为中断的实现原理，在翻译伪代码时总是出错，无法跟教程给出的文件匹配，assignment只运行成功一半却无法输出等等问题。好在之后查资料、咨询同学后已经解决了。

总的来说，通过这个实验，我不仅学到了理论知识，还通过实践加深了对计算机底层原理和汇编语言的理解。同时，我也掌握了一些常用的调试工具和方法，提高了自己的编程能力和实践能力。

## 参考文献

[将.asm文件编译成.o文件](https://blog.csdn.net/miyali123/article/details/80506873)

[BIOS](https://wiki.osdev.org/BIOS)

[键盘I/O中断调用](https://blog.csdn.net/deniece1/article/details/103447413)

[键盘扫描码](http://blog.sina.com.cn/s/blog_1511e79950102x2b0.html)

