	.arch armv8-a
	.file	"test_reg.c"
	.text
	.align	2
	.global	main
	.type	main, %function
main:
.LFB3939:
	.cfi_startproc
	sub	sp, sp, #48
	.cfi_def_cfa_offset 48
	mov	x0, 5
	str	x0, [sp, 40]
	ldr	x0, [sp, 40]
	add	x0, x0, 3
	str	x0, [sp, 40]
	adrp	x0, .LC0
	add	x0, x0, :lo12:.LC0
	ldr	h0, [x0]
	str	h0, [sp, 38]
	ldr	h0, [sp, 38]
	str	h0, [sp, 14]
	ldr	h0, [sp, 14]
	dup	v0.8h, v0.h[0]
	str	q0, [sp, 16]
	mov	w0, 0
	add	sp, sp, 48
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE3939:
	.size	main, .-main
	.section	.rodata
	.align	1
.LC0:
	.hword	17152
	.text
	.ident	"GCC: (GNU) 7.3.1 20180712 (Red Hat 7.3.1-13)"
	.section	.note.GNU-stack,"",@progbits
