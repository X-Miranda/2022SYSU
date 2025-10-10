%include "boot.inc"
org 0x7e00
[bits 16]

mov ax, 3; clear the screen
int 10h

mov ax, 0xb800
mov gs, ax
mov ah, 0x03 ;��ɫ
mov ecx, bootloader_tag_end - bootloader_tag
xor ebx, 60;ebx
mov esi, bootloader_tag
output_bootloader_tag:
    mov al, [esi]
    mov word[gs:bx], ax
    inc esi
    add ebx,2
    loop output_bootloader_tag

;��������
mov dword [GDT_START_ADDRESS+0x00],0x00
mov dword [GDT_START_ADDRESS+0x04],0x00  

;����������������һ�����ݶΣ���Ӧ0~4GB�����Ե�ַ�ռ�
mov dword [GDT_START_ADDRESS+0x08],0x0000ffff    ; ����ַΪ0���ν���Ϊ0xFFFFF
mov dword [GDT_START_ADDRESS+0x0c],0x00cf9200    ; ����Ϊ4KB���洢���������� 

;��������ģʽ�µĶ�ջ��������      
mov dword [GDT_START_ADDRESS+0x10],0x00000000    ; ����ַΪ0x00000000������0x0 
mov dword [GDT_START_ADDRESS+0x14],0x00409600    ; ����Ϊ1���ֽ�

;��������ģʽ�µ��Դ�������   
mov dword [GDT_START_ADDRESS+0x18],0x80007fff    ; ����ַΪ0x000B8000������0x07FFF 
mov dword [GDT_START_ADDRESS+0x1c],0x0040920b    ; ����Ϊ�ֽ�

;��������ģʽ��ƽ̹ģʽ�����������
mov dword [GDT_START_ADDRESS+0x20],0x0000ffff    ; ����ַΪ0���ν���Ϊ0xFFFFF
mov dword [GDT_START_ADDRESS+0x24],0x00cf9800    ; ����Ϊ4kb������������� 

;��ʼ����������Ĵ���GDTR
mov word [pgdt], 39      ;��������Ľ���   
lgdt [pgdt]
      
in al,0x92                         ;����оƬ�ڵĶ˿� 
or al,0000_0010B
out 0x92,al                        ;��A20

cli                                ;�жϻ�����δ����
mov eax,cr0
or eax,1
mov cr0,eax                        ;����PEλ
      
;���½��뱣��ģʽ
jmp dword CODE_SELECTOR:protect_mode_begin

;16λ��������ѡ���ӣ�32λƫ��
;����ˮ�߲����л�������
[bits 32]           
protect_mode_begin:                              

mov eax, DATA_SELECTOR                     ;�������ݶ�(0..4GB)ѡ����
mov ds, eax
mov es, eax
mov eax, STACK_SELECTOR
mov ss, eax
mov eax, VIDEO_SELECTOR
mov gs, eax

mov ecx, protect_mode_tag_end - protect_mode_tag
mov ebx, 80 * 2 + 58
mov esi, protect_mode_tag
mov ah, 0x3
output_protect_mode_tag:
    mov al, [esi]
    mov word[gs:ebx], ax
    add ebx, 2
    inc esi
    loop output_protect_mode_tag
    
    


;------------------
;print name section
mov ah, 0x03 
mov al, '2'
mov [gs:2 * 30], ax

mov al, '2'
mov [gs:2 * 31], ax

mov al, '3'
mov [gs:2 * 32], ax

mov al, '3'
mov [gs:2 * 33], ax

mov al, '6'
mov [gs:2 * 34], ax

mov al, '2'
mov [gs:2 * 35], ax

mov al, '5'
mov [gs:2 * 36], ax

mov al, '9'
mov [gs:2 * 37], ax

mov al, 'X'
mov [gs:2 * 39], ax

mov al, 'i'
mov [gs:2 * 40], ax

mov al, 'e'
mov [gs:2 * 41], ax

mov al, 'Y'
mov [gs:2 * 43], ax

mov al, 'u'
mov [gs:2 * 44], ax

mov al, 't'
mov [gs:2 * 45], ax

mov al, 'o'
mov [gs:2 * 46], ax

mov al, 'n'
mov [gs:2 * 47], ax

mov al, 'g'
mov [gs:2 * 48], ax

;mov ah, 0x00
;int 16h

;mov ax, 3; clear the screen
;int 10h

mov bl,0x47     ;set the color
mov bh,0


flag_r dw 1
flag_c dw 1; saving for whether add or sub
row dw 0
col dw 2
;----�ַ�����------
loop: ;
    mov cx, [row]
    mov dx, [col]; initialize the register

;------------------------------------
;boundary check
    push ax
    cmp dx, 0
        je boundary_colomn_0
    cmp dx, 79
        je boundary_colomn_80
boundary_colomn_end:
    cmp cx, 0
        je boundary_row_0 
    cmp cx, 23
        je boundary_row_24
boundary_row_end:
    pop ax

;------------------------------------
; move the cursor
    add cx, [flag_r]
    add dx, [flag_c]
    mov ah, dl
    mov al, cl
    and al, 0x7
    add al, 0x30
    mov [row], cx; update to the memory
    mov [col], dx

;------------------------------------
; print the cursor
    imul bx, cx, 80
    add bx, dx ; cor = 80 * row + col
    imul bx, 2    
    mov [gs:bx], ax

;-----------------------------------
; print the character symmetrically
    mov bx, 24
    sub bx, cx
    mov cx, bx
    mov bx, 80
    sub bx, dx
    mov dx, bx

    imul bx, cx, 80
    add bx, dx ; cor = 80 * row + col
    imul bx, 2    
    mov [gs:bx], ax
; -----------------------
; this section is used for delay
;therefore it can print the character slower
    push cx
    mov cx, 0x7fff
loop_delay_1:   
    push cx     
    mov cx, 0x007f
    loop_delay_2:
        loop loop_delay_2
    pop cx
    loop loop_delay_1
    pop cx

    jmp loop




boundary_row_0:
    mov ax, 1
    mov [flag_r], ax
    jmp boundary_row_end
boundary_colomn_0:
    mov ax, 1
    mov [flag_c], ax
    jmp boundary_colomn_end
boundary_row_24:
    mov ax, -1
    mov [flag_r], ax
    jmp boundary_row_end
boundary_colomn_80:
    mov ax, -1
    mov [flag_c], ax
    jmp boundary_colomn_end
loop_end:

jmp $ ; ��ѭ��

pgdt dw 0
     dd GDT_START_ADDRESS

bootloader_tag db 'run bootloader'
bootloader_tag_end:

protect_mode_tag db 'enter protect mode'
protect_mode_tag_end:
