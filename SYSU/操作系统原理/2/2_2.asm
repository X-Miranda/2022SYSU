org 0x7c00
[bits 16]
xor ax, ax ; eax = 0
; ��ʼ���μĴ���, �ε�ַȫ����Ϊ0
mov ds, ax
mov ss, ax
mov es, ax
mov fs, ax
mov gs, ax
; ��ʼ��ջָ��
mov sp, 0x7c00

mov ah, 0x0E ; ���ܺţ���ʾ�ַ�
mov al, '2' ; Ҫ��ʾ���ַ�
mov bh, 0 ; ҳ��
mov bl, 0x07 ; �ַ���ɫ
int 0x10

mov ah, 0x0E ; ���ܺţ���ʾ�ַ�
mov al, '2' ; Ҫ��ʾ���ַ�
mov bh, 0 ; ҳ��
mov bl, 0x07 ; �ַ���ɫ
int 0x10

mov ah, 0x0E ; ���ܺţ���ʾ�ַ�
mov al, '3' ; Ҫ��ʾ���ַ�
mov bh, 0 ; ҳ��
mov bl, 0x07 ; �ַ���ɫ
int 0x10

mov ah, 0x0E ; ���ܺţ���ʾ�ַ�
mov al, '3' ; Ҫ��ʾ���ַ�
mov bh, 0 ; ҳ��
mov bl, 0x07 ; �ַ���ɫ
int 0x10

mov ah, 0x0E ; ���ܺţ���ʾ�ַ�
mov al, '6' ; Ҫ��ʾ���ַ�
mov bh, 0 ; ҳ��
mov bl, 0x07 ; �ַ���ɫ
int 0x10

mov ah, 0x0E ; ���ܺţ���ʾ�ַ�
mov al, '2' ; Ҫ��ʾ���ַ�
mov bh, 0 ; ҳ��
mov bl, 0x07 ; �ַ���ɫ
int 0x10

mov ah, 0x0E ; ���ܺţ���ʾ�ַ�
mov al, '5' ; Ҫ��ʾ���ַ�
mov bh, 0 ; ҳ��
mov bl, 0x07 ; �ַ���ɫ
int 0x10

mov ah, 0x0E ; ���ܺţ���ʾ�ַ�
mov al, '9' ; Ҫ��ʾ���ַ�
mov bh, 0 ; ҳ��
mov bl, 0x07 ; �ַ���ɫ
int 0x10

jmp $ ; ����ѭ��

times 510-($-$$) db 0 ; ���ʣ��ռ�Ϊ0
dw 0xAA55 ; ������־