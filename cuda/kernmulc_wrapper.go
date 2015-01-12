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

// CUDA handle for kernmulC kernel
var kernmulC_code cu.Function

// Stores the arguments for kernmulC kernel invocation
type kernmulC_args_t struct {
	arg_fftM unsafe.Pointer
	arg_fftK unsafe.Pointer
	arg_Nx   int
	arg_Ny   int
	argptr   [4]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for kernmulC kernel invocation
var kernmulC_args kernmulC_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	kernmulC_args.argptr[0] = unsafe.Pointer(&kernmulC_args.arg_fftM)
	kernmulC_args.argptr[1] = unsafe.Pointer(&kernmulC_args.arg_fftK)
	kernmulC_args.argptr[2] = unsafe.Pointer(&kernmulC_args.arg_Nx)
	kernmulC_args.argptr[3] = unsafe.Pointer(&kernmulC_args.arg_Ny)
}

// Wrapper for kernmulC CUDA kernel, asynchronous.
func k_kernmulC_async(fftM unsafe.Pointer, fftK unsafe.Pointer, Nx int, Ny int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("kernmulC")
	}

	kernmulC_args.Lock()
	defer kernmulC_args.Unlock()

	if kernmulC_code == 0 {
		kernmulC_code = fatbinLoad(kernmulC_map, "kernmulC")
	}

	kernmulC_args.arg_fftM = fftM
	kernmulC_args.arg_fftK = fftK
	kernmulC_args.arg_Nx = Nx
	kernmulC_args.arg_Ny = Ny

	args := kernmulC_args.argptr[:]
	cu.LaunchKernel(kernmulC_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("kernmulC")
	}
}

// maps compute capability on PTX code for kernmulC kernel.
var kernmulC_map = map[int]string{0: "",
	20: kernmulC_ptx_20,
	30: kernmulC_ptx_30,
	35: kernmulC_ptx_35}

// kernmulC PTX code for various compute capabilities.
const (
	kernmulC_ptx_20 = `
.version 4.0
.target sm_20
.address_size 64


.visible .entry kernmulC(
	.param .u64 kernmulC_param_0,
	.param .u64 kernmulC_param_1,
	.param .u32 kernmulC_param_2,
	.param .u32 kernmulC_param_3
)
{
	.reg .pred 	%p<4>;
	.reg .s32 	%r<13>;
	.reg .f32 	%f<10>;
	.reg .s64 	%rd<8>;


	ld.param.u64 	%rd1, [kernmulC_param_0];
	ld.param.u64 	%rd2, [kernmulC_param_1];
	ld.param.u32 	%r3, [kernmulC_param_2];
	ld.param.u32 	%r4, [kernmulC_param_3];
	mov.u32 	%r5, %ntid.x;
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %tid.x;
	mad.lo.s32 	%r1, %r5, %r6, %r7;
	mov.u32 	%r8, %ntid.y;
	mov.u32 	%r9, %ctaid.y;
	mov.u32 	%r10, %tid.y;
	mad.lo.s32 	%r2, %r8, %r9, %r10;
	setp.ge.s32	%p1, %r2, %r4;
	setp.ge.s32	%p2, %r1, %r3;
	or.pred  	%p3, %p2, %p1;
	@%p3 bra 	BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	cvta.to.global.u64 	%rd4, %rd1;
	mad.lo.s32 	%r11, %r2, %r3, %r1;
	shl.b32 	%r12, %r11, 1;
	mul.wide.s32 	%rd5, %r12, 4;
	add.s64 	%rd6, %rd4, %rd5;
	add.s64 	%rd7, %rd3, %rd5;
	ld.global.f32 	%f1, [%rd7];
	ld.global.f32 	%f2, [%rd6];
	mul.f32 	%f3, %f2, %f1;
	ld.global.f32 	%f4, [%rd7+4];
	ld.global.f32 	%f5, [%rd6+4];
	mul.f32 	%f6, %f5, %f4;
	sub.f32 	%f7, %f3, %f6;
	st.global.f32 	[%rd6], %f7;
	mul.f32 	%f8, %f5, %f1;
	fma.rn.f32 	%f9, %f2, %f4, %f8;
	st.global.f32 	[%rd6+4], %f9;

BB0_2:
	ret;
}


`
	kernmulC_ptx_30 = `
.version 4.0
.target sm_30
.address_size 64


.visible .entry kernmulC(
	.param .u64 kernmulC_param_0,
	.param .u64 kernmulC_param_1,
	.param .u32 kernmulC_param_2,
	.param .u32 kernmulC_param_3
)
{
	.reg .pred 	%p<4>;
	.reg .s32 	%r<13>;
	.reg .f32 	%f<10>;
	.reg .s64 	%rd<8>;


	ld.param.u64 	%rd1, [kernmulC_param_0];
	ld.param.u64 	%rd2, [kernmulC_param_1];
	ld.param.u32 	%r3, [kernmulC_param_2];
	ld.param.u32 	%r4, [kernmulC_param_3];
	mov.u32 	%r5, %ntid.x;
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %tid.x;
	mad.lo.s32 	%r1, %r5, %r6, %r7;
	mov.u32 	%r8, %ntid.y;
	mov.u32 	%r9, %ctaid.y;
	mov.u32 	%r10, %tid.y;
	mad.lo.s32 	%r2, %r8, %r9, %r10;
	setp.ge.s32	%p1, %r2, %r4;
	setp.ge.s32	%p2, %r1, %r3;
	or.pred  	%p3, %p2, %p1;
	@%p3 bra 	BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	cvta.to.global.u64 	%rd4, %rd1;
	mad.lo.s32 	%r11, %r2, %r3, %r1;
	shl.b32 	%r12, %r11, 1;
	mul.wide.s32 	%rd5, %r12, 4;
	add.s64 	%rd6, %rd4, %rd5;
	add.s64 	%rd7, %rd3, %rd5;
	ld.global.f32 	%f1, [%rd7];
	ld.global.f32 	%f2, [%rd6];
	mul.f32 	%f3, %f2, %f1;
	ld.global.f32 	%f4, [%rd7+4];
	ld.global.f32 	%f5, [%rd6+4];
	mul.f32 	%f6, %f5, %f4;
	sub.f32 	%f7, %f3, %f6;
	st.global.f32 	[%rd6], %f7;
	mul.f32 	%f8, %f5, %f1;
	fma.rn.f32 	%f9, %f2, %f4, %f8;
	st.global.f32 	[%rd6+4], %f9;

BB0_2:
	ret;
}


`
	kernmulC_ptx_35 = `
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

.visible .entry kernmulC(
	.param .u64 kernmulC_param_0,
	.param .u64 kernmulC_param_1,
	.param .u32 kernmulC_param_2,
	.param .u32 kernmulC_param_3
)
{
	.reg .pred 	%p<4>;
	.reg .s32 	%r<13>;
	.reg .f32 	%f<10>;
	.reg .s64 	%rd<8>;


	ld.param.u64 	%rd1, [kernmulC_param_0];
	ld.param.u64 	%rd2, [kernmulC_param_1];
	ld.param.u32 	%r3, [kernmulC_param_2];
	ld.param.u32 	%r4, [kernmulC_param_3];
	mov.u32 	%r5, %ntid.x;
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %tid.x;
	mad.lo.s32 	%r1, %r5, %r6, %r7;
	mov.u32 	%r8, %ntid.y;
	mov.u32 	%r9, %ctaid.y;
	mov.u32 	%r10, %tid.y;
	mad.lo.s32 	%r2, %r8, %r9, %r10;
	setp.ge.s32	%p1, %r2, %r4;
	setp.ge.s32	%p2, %r1, %r3;
	or.pred  	%p3, %p2, %p1;
	@%p3 bra 	BB2_2;

	cvta.to.global.u64 	%rd3, %rd2;
	cvta.to.global.u64 	%rd4, %rd1;
	mad.lo.s32 	%r11, %r2, %r3, %r1;
	shl.b32 	%r12, %r11, 1;
	mul.wide.s32 	%rd5, %r12, 4;
	add.s64 	%rd6, %rd4, %rd5;
	add.s64 	%rd7, %rd3, %rd5;
	ld.global.nc.f32 	%f1, [%rd7];
	ld.global.f32 	%f2, [%rd6];
	mul.f32 	%f3, %f2, %f1;
	ld.global.nc.f32 	%f4, [%rd7+4];
	ld.global.f32 	%f5, [%rd6+4];
	mul.f32 	%f6, %f5, %f4;
	sub.f32 	%f7, %f3, %f6;
	st.global.f32 	[%rd6], %f7;
	mul.f32 	%f8, %f5, %f1;
	fma.rn.f32 	%f9, %f2, %f4, %f8;
	st.global.f32 	[%rd6+4], %f9;

BB2_2:
	ret;
}


`
)
