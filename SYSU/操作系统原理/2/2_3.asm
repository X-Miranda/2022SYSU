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

 mov ah, 0x00 ; 等待并读取按键
 int 16h ; 调用键盘中断

 mov ah, 0x0E ; TTY 模式下输出字符
 int 10h ; 调用视频中断来回显字符
 
 jmp main_loop 
 
times 510 - ($ - $$) db 0 
db 0x55, 0xAA 

