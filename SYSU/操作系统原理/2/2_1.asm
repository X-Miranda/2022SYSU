org 0x7c00
[bits 16]
xor ax, ax ; eax = 0
; ��ʼ���μĴ���, �ε�ַȫ����Ϊ0
mov ds, ax
mov ss, ax
mov es, ax
; ��ʼ��ջָ��
mov sp, 0x7c00

mov ah,0x02    ;���ܺ�
mov bh,0x00    ;��0ҳ
mov dh,0x05    ;dh�з��к�
mov dl,0x12    ;dl�з��к�
int 0x10

mov ah,0x03 ; ���� 3 �ӹ����ǻ�ȡ���λ�ã���Ҫ���� ah �Ĵ���
mov bh,0   ; bh �Ĵ����Ǵ���ȡ�Ĺ���ҳ��

 int 0x10 ; ����� ch=��꿪ʼ�У�cl=��������
    			; dh = ��������кţ�dl= ��������к�