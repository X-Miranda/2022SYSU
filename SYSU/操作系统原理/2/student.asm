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
 ; 计算 while_flag 数组的偏移地址
    sub ebx, 12               ; 计算偏移量（a2 - 12）
    mov ecx, [while_flag]   
    mov [ecx + ebx], al       ; while_flag[ebx] = while_flag[a2 - 12] <- al 
    dec ebx                   ; a2--
    mov [a2], ebx             ; 更新 a2 
    cmp ebx, 12
    jge loop                  ; 如果 a2>=12， 继续循环
end_while:

%include "end.include"

your_function: 
    pusha                   

    ;xor ecx, ecx            ; 循环计数器清零
    mov edi, [your_string]
    
loop1:
    mov al, [edi]
    test al, al             ; 检查当前字符是否为'\0'
    jz end_loop         
    
    push eax           
    call print_a_char       ; 不需要手动压栈。c语言会自动压栈
    ; print_a_char会使用eax寄存器。记得push/pop 保存和恢复
    pop eax
    
    inc edi                 ; 循环计数器自增
    jmp loop1

end_loop:
    popa                    ; 恢复所有寄存器的值
    ret                     ; 返回
