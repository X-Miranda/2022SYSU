org 0x7c00
[bits 16]
xor ax, ax ; eax = 0
; ��ʼ���μĴ���, �ε�ַȫ����Ϊ0
mov ds, ax
mov ss, ax
mov es, ax
; ��ʼ��ջָ��
mov sp, 0x7c00

main_loop:

 mov ah, 0x00 ; �ȴ�����ȡ����
 int 16h ; ���ü����ж�

 mov ah, 0x0E ; TTY ģʽ������ַ�
 int 10h ; ������Ƶ�ж��������ַ�
 
 jmp main_loop 
 
times 510 - ($ - $$) db 0 
db 0x55, 0xAA 

