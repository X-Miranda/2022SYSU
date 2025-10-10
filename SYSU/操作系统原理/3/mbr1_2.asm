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
mov ax, 1                ; �߼������ŵ�0~15λ
mov cx, 0                ; �߼������ŵ�16~31λ
mov bx, 0x7e00           ; bootloader�ļ��ص�ַ
load_bootloader:
    call asm_read_hard_disk  ; ��ȡӲ��
    inc ax
    cmp ax, 5
    jle load_bootloader
jmp 0x0000:0x7e00        ; ��ת��bootloader

jmp $ ; ��ѭ��

asm_read_hard_disk: 

  mov ch, 0 ; 0==����=lba/(63*18)
  mov dh, 0 ; 0==��ͷ=��lba/63)%18
  mov dl, 80h
  
  mov cl, al ; lba
  inc cl ; ����=lba+1

  mov ah, 02h
  mov al, 1
  int 13h  ;int13h�ж�
  add bx, 512 ; �������׵�ַ+=512
  ret 

times 510 - ($ - $$) db 0
db 0x55, 0xaa