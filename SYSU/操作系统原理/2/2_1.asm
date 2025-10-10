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
mov dh,0x05    ;dh中放行号
mov dl,0x12    ;dl中放列号
int 0x10

mov ah,0x03 ; 输入 3 子功能是获取光标位置，需要存入 ah 寄存器
mov bh,0   ; bh 寄存器是待获取的光标的页号

 int 0x10 ; 输出： ch=光标开始行，cl=光标结束行
    			; dh = 光标所在行号，dl= 光标所在列号