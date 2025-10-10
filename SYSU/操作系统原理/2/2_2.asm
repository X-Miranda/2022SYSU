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