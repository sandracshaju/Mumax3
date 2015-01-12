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

// CUDA handle for regionaddv kernel
var regionaddv_code cu.Function

// Stores the arguments for regionaddv kernel invocation
type regionaddv_args_t struct {
	arg_dstx    unsafe.Pointer
	arg_dsty    unsafe.Pointer
	arg_dstz    unsafe.Pointer
	arg_LUTx    unsafe.Pointer
	arg_LUTy    unsafe.Pointer
	arg_LUTz    unsafe.Pointer
	arg_regions unsafe.Pointer
	arg_N       int
	argptr      [8]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for regionaddv kernel invocation
var regionaddv_args regionaddv_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	regionaddv_args.argptr[0] = unsafe.Pointer(&regionaddv_args.arg_dstx)
	regionaddv_args.argptr[1] = unsafe.Pointer(&regionaddv_args.arg_dsty)
	regionaddv_args.argptr[2] = unsafe.Pointer(&regionaddv_args.arg_dstz)
	regionaddv_args.argptr[3] = unsafe.Pointer(&regionaddv_args.arg_LUTx)
	regionaddv_args.argptr[4] = unsafe.Pointer(&regionaddv_args.arg_LUTy)
	regionaddv_args.argptr[5] = unsafe.Pointer(&regionaddv_args.arg_LUTz)
	regionaddv_args.argptr[6] = unsafe.Pointer(&regionaddv_args.arg_regions)
	regionaddv_args.argptr[7] = unsafe.Pointer(&regionaddv_args.arg_N)
}

// Wrapper for regionaddv CUDA kernel, asynchronous.
func k_regionaddv_async(dstx unsafe.Pointer, dsty unsafe.Pointer, dstz unsafe.Pointer, LUTx unsafe.Pointer, LUTy unsafe.Pointer, LUTz unsafe.Pointer, regions unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("regionaddv")
	}

	regionaddv_args.Lock()
	defer regionaddv_args.Unlock()

	if regionaddv_code == 0 {
		regionaddv_code = fatbinLoad(regionaddv_map, "regionaddv")
	}

	regionaddv_args.arg_dstx = dstx
	regionaddv_args.arg_dsty = dsty
	regionaddv_args.arg_dstz = dstz
	regionaddv_args.arg_LUTx = LUTx
	regionaddv_args.arg_LUTy = LUTy
	regionaddv_args.arg_LUTz = LUTz
	regionaddv_args.arg_regions = regions
	regionaddv_args.arg_N = N

	args := regionaddv_args.argptr[:]
	cu.LaunchKernel(regionaddv_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("regionaddv")
	}
}

// maps compute capability on PTX code for regionaddv kernel.
var regionaddv_map = map[int]string{0: "",
	20: regionaddv_ptx_20,
	30: regionaddv_ptx_30,
	35: regionaddv_ptx_35}

// regionaddv PTX code for various compute capabilities.
const (
	regionaddv_ptx_20 = `
.version 4.0
.target sm_20
.address_size 64


.visible .entry regionaddv(
	.param .u64 regionaddv_param_0,
	.param .u64 regionaddv_param_1,
	.param .u64 regionaddv_param_2,
	.param .u64 regionaddv_param_3,
	.param .u64 regionaddv_param_4,
	.param .u64 regionaddv_param_5,
	.param .u64 regionaddv_param_6,
	.param .u32 regionaddv_param_7
)
{
	.reg .pred 	%p<2>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<10>;
	.reg .s64 	%rd<26>;


	ld.param.u64 	%rd1, [regionaddv_param_0];
	ld.param.u64 	%rd2, [regionaddv_param_1];
	ld.param.u64 	%rd3, [regionaddv_param_2];
	ld.param.u64 	%rd4, [regionaddv_param_3];
	ld.param.u64 	%rd5, [regionaddv_param_4];
	ld.param.u64 	%rd6, [regionaddv_param_5];
	ld.param.u64 	%rd7, [regionaddv_param_6];
	ld.param.u32 	%r2, [regionaddv_param_7];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd8, %rd3;
	cvta.to.global.u64 	%rd9, %rd6;
	cvta.to.global.u64 	%rd10, %rd2;
	cvta.to.global.u64 	%rd11, %rd5;
	cvta.to.global.u64 	%rd12, %rd1;
	cvta.to.global.u64 	%rd13, %rd4;
	cvta.to.global.u64 	%rd14, %rd7;
	cvt.s64.s32	%rd15, %r1;
	add.s64 	%rd16, %rd14, %rd15;
	ld.global.u8 	%rd17, [%rd16];
	shl.b64 	%rd18, %rd17, 2;
	add.s64 	%rd19, %rd13, %rd18;
	mul.wide.s32 	%rd20, %r1, 4;
	add.s64 	%rd21, %rd12, %rd20;
	ld.global.f32 	%f1, [%rd21];
	ld.global.f32 	%f2, [%rd19];
	add.f32 	%f3, %f1, %f2;
	st.global.f32 	[%rd21], %f3;
	add.s64 	%rd22, %rd11, %rd18;
	add.s64 	%rd23, %rd10, %rd20;
	ld.global.f32 	%f4, [%rd23];
	ld.global.f32 	%f5, [%rd22];
	add.f32 	%f6, %f4, %f5;
	st.global.f32 	[%rd23], %f6;
	add.s64 	%rd24, %rd9, %rd18;
	add.s64 	%rd25, %rd8, %rd20;
	ld.global.f32 	%f7, [%rd25];
	ld.global.f32 	%f8, [%rd24];
	add.f32 	%f9, %f7, %f8;
	st.global.f32 	[%rd25], %f9;

BB0_2:
	ret;
}


`
	regionaddv_ptx_30 = `
.version 4.0
.target sm_30
.address_size 64


.visible .entry regionaddv(
	.param .u64 regionaddv_param_0,
	.param .u64 regionaddv_param_1,
	.param .u64 regionaddv_param_2,
	.param .u64 regionaddv_param_3,
	.param .u64 regionaddv_param_4,
	.param .u64 regionaddv_param_5,
	.param .u64 regionaddv_param_6,
	.param .u32 regionaddv_param_7
)
{
	.reg .pred 	%p<2>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<10>;
	.reg .s64 	%rd<26>;


	ld.param.u64 	%rd1, [regionaddv_param_0];
	ld.param.u64 	%rd2, [regionaddv_param_1];
	ld.param.u64 	%rd3, [regionaddv_param_2];
	ld.param.u64 	%rd4, [regionaddv_param_3];
	ld.param.u64 	%rd5, [regionaddv_param_4];
	ld.param.u64 	%rd6, [regionaddv_param_5];
	ld.param.u64 	%rd7, [regionaddv_param_6];
	ld.param.u32 	%r2, [regionaddv_param_7];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd8, %rd3;
	cvta.to.global.u64 	%rd9, %rd6;
	cvta.to.global.u64 	%rd10, %rd2;
	cvta.to.global.u64 	%rd11, %rd5;
	cvta.to.global.u64 	%rd12, %rd1;
	cvta.to.global.u64 	%rd13, %rd4;
	cvta.to.global.u64 	%rd14, %rd7;
	cvt.s64.s32	%rd15, %r1;
	add.s64 	%rd16, %rd14, %rd15;
	ld.global.u8 	%rd17, [%rd16];
	shl.b64 	%rd18, %rd17, 2;
	add.s64 	%rd19, %rd13, %rd18;
	mul.wide.s32 	%rd20, %r1, 4;
	add.s64 	%rd21, %rd12, %rd20;
	ld.global.f32 	%f1, [%rd21];
	ld.global.f32 	%f2, [%rd19];
	add.f32 	%f3, %f1, %f2;
	st.global.f32 	[%rd21], %f3;
	add.s64 	%rd22, %rd11, %rd18;
	add.s64 	%rd23, %rd10, %rd20;
	ld.global.f32 	%f4, [%rd23];
	ld.global.f32 	%f5, [%rd22];
	add.f32 	%f6, %f4, %f5;
	st.global.f32 	[%rd23], %f6;
	add.s64 	%rd24, %rd9, %rd18;
	add.s64 	%rd25, %rd8, %rd20;
	ld.global.f32 	%f7, [%rd25];
	ld.global.f32 	%f8, [%rd24];
	add.f32 	%f9, %f7, %f8;
	st.global.f32 	[%rd25], %f9;

BB0_2:
	ret;
}


`
	regionaddv_ptx_35 = `
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

.visible .entry regionaddv(
	.param .u64 regionaddv_param_0,
	.param .u64 regionaddv_param_1,
	.param .u64 regionaddv_param_2,
	.param .u64 regionaddv_param_3,
	.param .u64 regionaddv_param_4,
	.param .u64 regionaddv_param_5,
	.param .u64 regionaddv_param_6,
	.param .u32 regionaddv_param_7
)
{
	.reg .pred 	%p<2>;
	.reg .s16 	%rs<2>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<10>;
	.reg .s64 	%rd<27>;


	ld.param.u64 	%rd1, [regionaddv_param_0];
	ld.param.u64 	%rd2, [regionaddv_param_1];
	ld.param.u64 	%rd3, [regionaddv_param_2];
	ld.param.u64 	%rd4, [regionaddv_param_3];
	ld.param.u64 	%rd5, [regionaddv_param_4];
	ld.param.u64 	%rd6, [regionaddv_param_5];
	ld.param.u64 	%rd7, [regionaddv_param_6];
	ld.param.u32 	%r2, [regionaddv_param_7];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB2_2;

	cvta.to.global.u64 	%rd8, %rd3;
	cvta.to.global.u64 	%rd9, %rd6;
	cvta.to.global.u64 	%rd10, %rd2;
	cvta.to.global.u64 	%rd11, %rd5;
	cvta.to.global.u64 	%rd12, %rd1;
	cvta.to.global.u64 	%rd13, %rd4;
	cvta.to.global.u64 	%rd14, %rd7;
	cvt.s64.s32	%rd15, %r1;
	add.s64 	%rd16, %rd14, %rd15;
	ld.global.nc.u8 	%rs1, [%rd16];
	cvt.u64.u16	%rd17, %rs1;
	and.b64  	%rd18, %rd17, 255;
	shl.b64 	%rd19, %rd18, 2;
	add.s64 	%rd20, %rd13, %rd19;
	mul.wide.s32 	%rd21, %r1, 4;
	add.s64 	%rd22, %rd12, %rd21;
	ld.global.f32 	%f1, [%rd22];
	ld.global.nc.f32 	%f2, [%rd20];
	add.f32 	%f3, %f1, %f2;
	st.global.f32 	[%rd22], %f3;
	add.s64 	%rd23, %rd11, %rd19;
	add.s64 	%rd24, %rd10, %rd21;
	ld.global.f32 	%f4, [%rd24];
	ld.global.nc.f32 	%f5, [%rd23];
	add.f32 	%f6, %f4, %f5;
	st.global.f32 	[%rd24], %f6;
	add.s64 	%rd25, %rd9, %rd19;
	add.s64 	%rd26, %rd8, %rd21;
	ld.global.f32 	%f7, [%rd26];
	ld.global.nc.f32 	%f8, [%rd25];
	add.f32 	%f9, %f7, %f8;
	st.global.f32 	[%rd26], %f9;

BB2_2:
	ret;
}


`
)
