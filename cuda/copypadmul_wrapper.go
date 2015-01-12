package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import (
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/timer"
	"sync"
	"unsafe"
)

// CUDA handle for copypadmul kernel
var copypadmul_code cu.Function

// Stores the arguments for copypadmul kernel invocation
type copypadmul_args_t struct {
	arg_dst     unsafe.Pointer
	arg_Dx      int
	arg_Dy      int
	arg_Dz      int
	arg_src     unsafe.Pointer
	arg_vol     unsafe.Pointer
	arg_Sx      int
	arg_Sy      int
	arg_Sz      int
	arg_BsatLUT unsafe.Pointer
	arg_regions unsafe.Pointer
	argptr      [11]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for copypadmul kernel invocation
var copypadmul_args copypadmul_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	copypadmul_args.argptr[0] = unsafe.Pointer(&copypadmul_args.arg_dst)
	copypadmul_args.argptr[1] = unsafe.Pointer(&copypadmul_args.arg_Dx)
	copypadmul_args.argptr[2] = unsafe.Pointer(&copypadmul_args.arg_Dy)
	copypadmul_args.argptr[3] = unsafe.Pointer(&copypadmul_args.arg_Dz)
	copypadmul_args.argptr[4] = unsafe.Pointer(&copypadmul_args.arg_src)
	copypadmul_args.argptr[5] = unsafe.Pointer(&copypadmul_args.arg_vol)
	copypadmul_args.argptr[6] = unsafe.Pointer(&copypadmul_args.arg_Sx)
	copypadmul_args.argptr[7] = unsafe.Pointer(&copypadmul_args.arg_Sy)
	copypadmul_args.argptr[8] = unsafe.Pointer(&copypadmul_args.arg_Sz)
	copypadmul_args.argptr[9] = unsafe.Pointer(&copypadmul_args.arg_BsatLUT)
	copypadmul_args.argptr[10] = unsafe.Pointer(&copypadmul_args.arg_regions)
}

// Wrapper for copypadmul CUDA kernel, asynchronous.
func k_copypadmul_async(dst unsafe.Pointer, Dx int, Dy int, Dz int, src unsafe.Pointer, vol unsafe.Pointer, Sx int, Sy int, Sz int, BsatLUT unsafe.Pointer, regions unsafe.Pointer, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("copypadmul")
	}

	copypadmul_args.Lock()
	defer copypadmul_args.Unlock()

	if copypadmul_code == 0 {
		copypadmul_code = fatbinLoad(copypadmul_map, "copypadmul")
	}

	copypadmul_args.arg_dst = dst
	copypadmul_args.arg_Dx = Dx
	copypadmul_args.arg_Dy = Dy
	copypadmul_args.arg_Dz = Dz
	copypadmul_args.arg_src = src
	copypadmul_args.arg_vol = vol
	copypadmul_args.arg_Sx = Sx
	copypadmul_args.arg_Sy = Sy
	copypadmul_args.arg_Sz = Sz
	copypadmul_args.arg_BsatLUT = BsatLUT
	copypadmul_args.arg_regions = regions

	args := copypadmul_args.argptr[:]
	cu.LaunchKernel(copypadmul_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("copypadmul")
	}
}

// maps compute capability on PTX code for copypadmul kernel.
var copypadmul_map = map[int]string{0: "",
	20: copypadmul_ptx_20,
	30: copypadmul_ptx_30,
	35: copypadmul_ptx_35}

// copypadmul PTX code for various compute capabilities.
const (
	copypadmul_ptx_20 = `
.version 4.0
.target sm_20
.address_size 64


.visible .entry copypadmul(
	.param .u64 copypadmul_param_0,
	.param .u32 copypadmul_param_1,
	.param .u32 copypadmul_param_2,
	.param .u32 copypadmul_param_3,
	.param .u64 copypadmul_param_4,
	.param .u64 copypadmul_param_5,
	.param .u32 copypadmul_param_6,
	.param .u32 copypadmul_param_7,
	.param .u32 copypadmul_param_8,
	.param .u64 copypadmul_param_9,
	.param .u64 copypadmul_param_10
)
{
	.reg .pred 	%p<7>;
	.reg .s32 	%r<22>;
	.reg .f32 	%f<9>;
	.reg .s64 	%rd<22>;


	ld.param.u64 	%rd2, [copypadmul_param_0];
	ld.param.u32 	%r4, [copypadmul_param_1];
	ld.param.u32 	%r5, [copypadmul_param_2];
	ld.param.u64 	%rd3, [copypadmul_param_4];
	ld.param.u64 	%rd4, [copypadmul_param_5];
	ld.param.u32 	%r6, [copypadmul_param_6];
	ld.param.u32 	%r7, [copypadmul_param_7];
	ld.param.u32 	%r8, [copypadmul_param_8];
	ld.param.u64 	%rd5, [copypadmul_param_9];
	ld.param.u64 	%rd6, [copypadmul_param_10];
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r15, %r16, %r17;
	setp.lt.s32	%p1, %r1, %r6;
	setp.lt.s32	%p2, %r2, %r7;
	and.pred  	%p3, %p1, %p2;
	setp.lt.s32	%p4, %r3, %r8;
	and.pred  	%p5, %p3, %p4;
	@!%p5 bra 	BB0_5;
	bra.uni 	BB0_1;

BB0_1:
	cvta.to.global.u64 	%rd7, %rd5;
	cvta.to.global.u64 	%rd8, %rd6;
	mad.lo.s32 	%r18, %r3, %r7, %r2;
	mad.lo.s32 	%r19, %r18, %r6, %r1;
	cvt.s64.s32	%rd1, %r19;
	add.s64 	%rd9, %rd8, %rd1;
	ld.global.u8 	%rd10, [%rd9];
	shl.b64 	%rd11, %rd10, 2;
	add.s64 	%rd12, %rd7, %rd11;
	ld.global.f32 	%f1, [%rd12];
	setp.ne.s64	%p6, %rd4, 0;
	@%p6 bra 	BB0_3;

	mov.f32 	%f8, 0f3F800000;
	bra.uni 	BB0_4;

BB0_3:
	cvta.to.global.u64 	%rd13, %rd4;
	shl.b64 	%rd14, %rd1, 2;
	add.s64 	%rd15, %rd13, %rd14;
	ld.global.f32 	%f8, [%rd15];

BB0_4:
	cvta.to.global.u64 	%rd16, %rd2;
	cvta.to.global.u64 	%rd17, %rd3;
	shl.b64 	%rd18, %rd1, 2;
	add.s64 	%rd19, %rd17, %rd18;
	ld.global.f32 	%f5, [%rd19];
	mul.f32 	%f6, %f1, %f8;
	mul.f32 	%f7, %f6, %f5;
	mad.lo.s32 	%r20, %r3, %r5, %r2;
	mad.lo.s32 	%r21, %r20, %r4, %r1;
	mul.wide.s32 	%rd20, %r21, 4;
	add.s64 	%rd21, %rd16, %rd20;
	st.global.f32 	[%rd21], %f7;

BB0_5:
	ret;
}


`
	copypadmul_ptx_30 = `
.version 4.0
.target sm_30
.address_size 64


.visible .entry copypadmul(
	.param .u64 copypadmul_param_0,
	.param .u32 copypadmul_param_1,
	.param .u32 copypadmul_param_2,
	.param .u32 copypadmul_param_3,
	.param .u64 copypadmul_param_4,
	.param .u64 copypadmul_param_5,
	.param .u32 copypadmul_param_6,
	.param .u32 copypadmul_param_7,
	.param .u32 copypadmul_param_8,
	.param .u64 copypadmul_param_9,
	.param .u64 copypadmul_param_10
)
{
	.reg .pred 	%p<7>;
	.reg .s32 	%r<22>;
	.reg .f32 	%f<9>;
	.reg .s64 	%rd<22>;


	ld.param.u64 	%rd2, [copypadmul_param_0];
	ld.param.u32 	%r4, [copypadmul_param_1];
	ld.param.u32 	%r5, [copypadmul_param_2];
	ld.param.u64 	%rd3, [copypadmul_param_4];
	ld.param.u64 	%rd4, [copypadmul_param_5];
	ld.param.u32 	%r6, [copypadmul_param_6];
	ld.param.u32 	%r7, [copypadmul_param_7];
	ld.param.u32 	%r8, [copypadmul_param_8];
	ld.param.u64 	%rd5, [copypadmul_param_9];
	ld.param.u64 	%rd6, [copypadmul_param_10];
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r15, %r16, %r17;
	setp.lt.s32	%p1, %r1, %r6;
	setp.lt.s32	%p2, %r2, %r7;
	and.pred  	%p3, %p1, %p2;
	setp.lt.s32	%p4, %r3, %r8;
	and.pred  	%p5, %p3, %p4;
	@!%p5 bra 	BB0_5;
	bra.uni 	BB0_1;

BB0_1:
	cvta.to.global.u64 	%rd7, %rd5;
	cvta.to.global.u64 	%rd8, %rd6;
	mad.lo.s32 	%r18, %r3, %r7, %r2;
	mad.lo.s32 	%r19, %r18, %r6, %r1;
	cvt.s64.s32	%rd1, %r19;
	add.s64 	%rd9, %rd8, %rd1;
	ld.global.u8 	%rd10, [%rd9];
	shl.b64 	%rd11, %rd10, 2;
	add.s64 	%rd12, %rd7, %rd11;
	ld.global.f32 	%f1, [%rd12];
	setp.ne.s64	%p6, %rd4, 0;
	@%p6 bra 	BB0_3;

	mov.f32 	%f8, 0f3F800000;
	bra.uni 	BB0_4;

BB0_3:
	cvta.to.global.u64 	%rd13, %rd4;
	shl.b64 	%rd14, %rd1, 2;
	add.s64 	%rd15, %rd13, %rd14;
	ld.global.f32 	%f8, [%rd15];

BB0_4:
	cvta.to.global.u64 	%rd16, %rd2;
	cvta.to.global.u64 	%rd17, %rd3;
	shl.b64 	%rd18, %rd1, 2;
	add.s64 	%rd19, %rd17, %rd18;
	ld.global.f32 	%f5, [%rd19];
	mul.f32 	%f6, %f1, %f8;
	mul.f32 	%f7, %f6, %f5;
	mad.lo.s32 	%r20, %r3, %r5, %r2;
	mad.lo.s32 	%r21, %r20, %r4, %r1;
	mul.wide.s32 	%rd20, %r21, 4;
	add.s64 	%rd21, %rd16, %rd20;
	st.global.f32 	[%rd21], %f7;

BB0_5:
	ret;
}


`
	copypadmul_ptx_35 = `
.version 4.0
.target sm_35
.address_size 64


.weak .func  (.param .b32 func_retval0) cudaMalloc(
	.param .b64 cudaMalloc_param_0,
	.param .b64 cudaMalloc_param_1
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

.weak .func  (.param .b32 func_retval0) cudaFuncGetAttributes(
	.param .b64 cudaFuncGetAttributes_param_0,
	.param .b64 cudaFuncGetAttributes_param_1
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}

.visible .entry copypadmul(
	.param .u64 copypadmul_param_0,
	.param .u32 copypadmul_param_1,
	.param .u32 copypadmul_param_2,
	.param .u32 copypadmul_param_3,
	.param .u64 copypadmul_param_4,
	.param .u64 copypadmul_param_5,
	.param .u32 copypadmul_param_6,
	.param .u32 copypadmul_param_7,
	.param .u32 copypadmul_param_8,
	.param .u64 copypadmul_param_9,
	.param .u64 copypadmul_param_10
)
{
	.reg .pred 	%p<7>;
	.reg .s16 	%rs<2>;
	.reg .s32 	%r<22>;
	.reg .f32 	%f<9>;
	.reg .s64 	%rd<23>;


	ld.param.u64 	%rd2, [copypadmul_param_0];
	ld.param.u32 	%r4, [copypadmul_param_1];
	ld.param.u32 	%r5, [copypadmul_param_2];
	ld.param.u64 	%rd3, [copypadmul_param_4];
	ld.param.u64 	%rd4, [copypadmul_param_5];
	ld.param.u32 	%r6, [copypadmul_param_6];
	ld.param.u32 	%r7, [copypadmul_param_7];
	ld.param.u32 	%r8, [copypadmul_param_8];
	ld.param.u64 	%rd5, [copypadmul_param_9];
	ld.param.u64 	%rd6, [copypadmul_param_10];
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r9, %r10, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r12, %r13, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r15, %r16, %r17;
	setp.lt.s32	%p1, %r1, %r6;
	setp.lt.s32	%p2, %r2, %r7;
	and.pred  	%p3, %p1, %p2;
	setp.lt.s32	%p4, %r3, %r8;
	and.pred  	%p5, %p3, %p4;
	@!%p5 bra 	BB2_5;
	bra.uni 	BB2_1;

BB2_1:
	cvta.to.global.u64 	%rd7, %rd5;
	cvta.to.global.u64 	%rd8, %rd6;
	mad.lo.s32 	%r18, %r3, %r7, %r2;
	mad.lo.s32 	%r19, %r18, %r6, %r1;
	cvt.s64.s32	%rd1, %r19;
	add.s64 	%rd9, %rd8, %rd1;
	ld.global.nc.u8 	%rs1, [%rd9];
	cvt.u64.u16	%rd10, %rs1;
	and.b64  	%rd11, %rd10, 255;
	shl.b64 	%rd12, %rd11, 2;
	add.s64 	%rd13, %rd7, %rd12;
	ld.global.nc.f32 	%f1, [%rd13];
	setp.ne.s64	%p6, %rd4, 0;
	@%p6 bra 	BB2_3;

	mov.f32 	%f8, 0f3F800000;
	bra.uni 	BB2_4;

BB2_3:
	cvta.to.global.u64 	%rd14, %rd4;
	shl.b64 	%rd15, %rd1, 2;
	add.s64 	%rd16, %rd14, %rd15;
	ld.global.nc.f32 	%f8, [%rd16];

BB2_4:
	cvta.to.global.u64 	%rd17, %rd2;
	cvta.to.global.u64 	%rd18, %rd3;
	shl.b64 	%rd19, %rd1, 2;
	add.s64 	%rd20, %rd18, %rd19;
	ld.global.nc.f32 	%f5, [%rd20];
	mul.f32 	%f6, %f1, %f8;
	mul.f32 	%f7, %f6, %f5;
	mad.lo.s32 	%r20, %r3, %r5, %r2;
	mad.lo.s32 	%r21, %r20, %r4, %r1;
	mul.wide.s32 	%rd21, %r21, 4;
	add.s64 	%rd22, %rd17, %rd21;
	st.global.f32 	[%rd22], %f7;

BB2_5:
	ret;
}


`
)
