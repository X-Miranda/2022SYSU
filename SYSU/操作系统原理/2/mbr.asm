org 0x7c00
[bits 16]
xor ax, ax ; eax = 0
;��ʼ���μĴ���, �ε�ַȫ����Ϊ0
mov ds, ax
mov ss, ax
mov es, ax
mov fs, ax
mov gs, ax

;��ʼ��ջָ��
mov sp, 0x7c00
mov ax, 0xb800
mov gs, ax


mov ah, 0x01 ;��ɫ
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

jmp $ ; ��ѭ��

times 510 - ($ - $$) db 0
db 0x55, 0xaa

mov ah,2    ;���ܺ�
mov bh,0    ;��0ҳ
mov dh,5    ;dh�з��к�
mov dl,12    ;dl�з��к�
int 10h

mov ah,3 ; ���� 3 �ӹ����ǻ�ȡ���λ�ã���Ҫ���� ah �Ĵ���
mov bh,0   ; bh �Ĵ����Ǵ���ȡ�Ĺ���ҳ��

 int 0x10 ; ����� ch=��꿪ʼ�У�cl=��������
    			; dh = ��������кţ�dl= ��������к�
