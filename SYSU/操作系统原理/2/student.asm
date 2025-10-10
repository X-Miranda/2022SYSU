; If you meet compile error, try 'sudo apt install gcc-multilib g++-multilib' first

%include "head.include"
; you code here

your_if:
; put your implementation here
    mov eax, [a1] 
    cmp eax, 12
    jl lt12       ; a1 < 12
    cmp eax, 24
    jl lt24       ; 12 <= a1 < 24
                  ; a1 >= 24 
    mov eax, [a1]
    shl eax, 4    ; a1 * 16 
    jmp end_if 
lt12:
    mov eax, [a1]
    shr eax, 1    ; a1 / 2 
    add eax, 1    ; a1 + 1 
    jmp end_if 
lt24:
    mov eax, [a1]
    mov ebx, 24
    sub ebx, eax  ; 24 - a1 
    imul ebx, eax ; (24 - a1) * a1 
    mov eax, ebx 
    jmp end_if 
end_if:
    mov [if_flag], eax


your_while:
; put your implementation here
    mov ebx, [a2]             
    cmp ebx, 12
    jl end_while
loop:
    call my_random        
 ; ���� while_flag �����ƫ�Ƶ�ַ
    sub ebx, 12               ; ����ƫ������a2 - 12��
    mov ecx, [while_flag]   
    mov [ecx + ebx], al       ; while_flag[ebx] = while_flag[a2 - 12] <- al 
    dec ebx                   ; a2--
    mov [a2], ebx             ; ���� a2 
    cmp ebx, 12
    jge loop                  ; ��� a2>=12�� ����ѭ��
end_while:

%include "end.include"

your_function: 
    pusha                   

    ;xor ecx, ecx            ; ѭ������������
    mov edi, [your_string]
    
loop1:
    mov al, [edi]
    test al, al             ; ��鵱ǰ�ַ��Ƿ�Ϊ'\0'
    jz end_loop         
    
    push eax           
    call print_a_char       ; ����Ҫ�ֶ�ѹջ��c���Ի��Զ�ѹջ
    ; print_a_char��ʹ��eax�Ĵ������ǵ�push/pop ����ͻָ�
    pop eax
    
    inc edi                 ; ѭ������������
    jmp loop1

end_loop:
    popa                    ; �ָ����мĴ�����ֵ
    ret                     ; ����
