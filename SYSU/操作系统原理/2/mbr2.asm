org 0x7c00
[bits 16]
xor ax, ax ; eax = 0

mov ds, ax
mov ss, ax
mov es, ax
mov fs, ax
mov gs, ax

mov sp, 0x7c00
mov ax, 0xb800
mov gs, ax


mov ah, 0x01 
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

mov ah, 0x03 
mov al, '2'
mov [gs:2 * (12*80 + 12)], ax

mov al, '2'
mov [gs:2 * (12*80 + 13)], ax

mov al, '3'
mov [gs:2 * (12*80 + 14)], ax

mov al, '3'
mov [gs:2 * (12*80 + 15)], ax

mov al, '6'
mov [gs:2 * (12*80 + 16)], ax

mov al, '2'
mov [gs:2 * (12*80 + 17)], ax

mov al, '5'
mov [gs:2 * (12*80 + 18)], ax

mov al, '9'
mov [gs:2 * (12*80 + 19)], ax


jmp $ 

times 510 - ($ - $$) db 0
db 0x55, 0xaa
