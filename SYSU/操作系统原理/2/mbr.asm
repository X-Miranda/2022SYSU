org 0x7c00
[bits 16]
xor ax, ax ; eax = 0
;初始化段寄存器, 段地址全部设为0
mov ds, ax
mov ss, ax
mov es, ax
mov fs, ax
mov gs, ax

;初始化栈指针
mov sp, 0x7c00
mov ax, 0xb800
mov gs, ax


mov ah, 0x01 ;蓝色
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

jmp $ ; 死循环

times 510 - ($ - $$) db 0
db 0x55, 0xaa

mov ah,2    ;功能号
mov bh,0    ;第0页
mov dh,5    ;dh中放行号
mov dl,12    ;dl中放列号
int 10h

mov ah,3 ; 输入 3 子功能是获取光标位置，需要存入 ah 寄存器
mov bh,0   ; bh 寄存器是待获取的光标的页号

 int 0x10 ; 输出： ch=光标开始行，cl=光标结束行
    			; dh = 光标所在行号，dl= 光标所在列号
