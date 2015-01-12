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

// CUDA handle for regionselect kernel
var regionselect_code cu.Function

// Stores the arguments for regionselect kernel invocation
type regionselect_args_t struct {
	arg_dst     unsafe.Pointer
	arg_src     unsafe.Pointer
	arg_regions unsafe.Pointer
	arg_region  byte
	arg_N       int
	argptr      [5]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for regionselect kernel invocation
var regionselect_args regionselect_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	regionselect_args.argptr[0] = unsafe.Pointer(&regionselect_args.arg_dst)
	regionselect_args.argptr[1] = unsafe.Pointer(&regionselect_args.arg_src)
	regionselect_args.argptr[2] = unsafe.Pointer(&regionselect_args.arg_regions)
	regionselect_args.argptr[3] = unsafe.Pointer(&regionselect_args.arg_region)
	regionselect_args.argptr[4] = unsafe.Pointer(&regionselect_args.arg_N)
}

// Wrapper for regionselect CUDA kernel, asynchronous.
func k_regionselect_async(dst unsafe.Pointer, src unsafe.Pointer, regions unsafe.Pointer, region byte, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("regionselect")
	}

	regionselect_args.Lock()
	defer regionselect_args.Unlock()

	if regionselect_code == 0 {
		regionselect_code = fatbinLoad(regionselect_map, "regionselect")
	}

	regionselect_args.arg_dst = dst
	regionselect_args.arg_src = src
	regionselect_args.arg_regions = regions
	regionselect_args.arg_region = region
	regionselect_args.arg_N = N

	args := regionselect_args.argptr[:]
	cu.LaunchKernel(regionselect_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("regionselect")
	}
}

// maps compute capability on PTX code for regionselect kernel.
var regionselect_map = map[int]string{0: "",
	20: regionselect_ptx_20,
	30: regionselect_ptx_30,
	35: regionselect_ptx_35}

// regionselect PTX code for various compute capabilities.
const (
	regionselect_ptx_20 = `
.version 4.0
.target sm_20
.address_size 64


.visible .entry regionselect(
	.param .u64 regionselect_param_0,
	.param .u64 regionselect_param_1,
	.param .u64 regionselect_param_2,
	.param .u8 regionselect_param_3,
	.param .u32 regionselect_param_4
)
{
	.reg .pred 	%p<3>;
	.reg .s16 	%rs<3>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<5>;
	.reg .s64 	%rd<13>;


	ld.param.u64 	%rd1, [regionselect_param_0];
	ld.param.u64 	%rd2, [regionselect_param_1];
	ld.param.u64 	%rd3, [regionselect_param_2];
	ld.param.u8 	%rs1, [regionselect_param_3];
	ld.param.u32 	%r2, [regionselect_param_4];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_5;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32	%rd5, %r1;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.u8 	%rs2, [%rd6];
	setp.eq.s16	%p2, %rs2, %rs1;
	@%p2 bra 	BB0_3;

	mov.f32 	%f4, 0f00000000;
	bra.uni 	BB0_4;

BB0_3:
	cvta.to.global.u64 	%rd7, %rd2;
	mul.wide.s32 	%rd8, %r1, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.f32 	%f4, [%rd9];

BB0_4:
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r1, 4;
	add.s64 	%rd12, %rd10, %rd11;
	st.global.f32 	[%rd12], %f4;

BB0_5:
	ret;
}


`
	regionselect_ptx_30 = `
.version 4.0
.target sm_30
.address_size 64


.visible .entry regionselect(
	.param .u64 regionselect_param_0,
	.param .u64 regionselect_param_1,
	.param .u64 regionselect_param_2,
	.param .u8 regionselect_param_3,
	.param .u32 regionselect_param_4
)
{
	.reg .pred 	%p<3>;
	.reg .s16 	%rs<3>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<5>;
	.reg .s64 	%rd<13>;


	ld.param.u64 	%rd1, [regionselect_param_0];
	ld.param.u64 	%rd2, [regionselect_param_1];
	ld.param.u64 	%rd3, [regionselect_param_2];
	ld.param.u8 	%rs1, [regionselect_param_3];
	ld.param.u32 	%r2, [regionselect_param_4];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_5;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32	%rd5, %r1;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.u8 	%rs2, [%rd6];
	setp.eq.s16	%p2, %rs2, %rs1;
	@%p2 bra 	BB0_3;

	mov.f32 	%f4, 0f00000000;
	bra.uni 	BB0_4;

BB0_3:
	cvta.to.global.u64 	%rd7, %rd2;
	mul.wide.s32 	%rd8, %r1, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.f32 	%f4, [%rd9];

BB0_4:
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r1, 4;
	add.s64 	%rd12, %rd10, %rd11;
	st.global.f32 	[%rd12], %f4;

BB0_5:
	ret;
}


`
	regionselect_ptx_35 = `
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

.visible .entry regionselect(
	.param .u64 regionselect_param_0,
	.param .u64 regionselect_param_1,
	.param .u64 regionselect_param_2,
	.param .u8 regionselect_param_3,
	.param .u32 regionselect_param_4
)
{
	.reg .pred 	%p<3>;
	.reg .s16 	%rs<4>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<5>;
	.reg .s64 	%rd<13>;


	ld.param.u64 	%rd1, [regionselect_param_0];
	ld.param.u64 	%rd2, [regionselect_param_1];
	ld.param.u64 	%rd3, [regionselect_param_2];
	ld.param.u8 	%rs1, [regionselect_param_3];
	ld.param.u32 	%r2, [regionselect_param_4];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB2_5;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32	%rd5, %r1;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.u8 	%rs2, [%rd6];
	and.b16  	%rs3, %rs2, 255;
	setp.eq.s16	%p2, %rs3, %rs1;
	@%p2 bra 	BB2_3;

	mov.f32 	%f4, 0f00000000;
	bra.uni 	BB2_4;

BB2_3:
	cvta.to.global.u64 	%rd7, %rd2;
	mul.wide.s32 	%rd8, %r1, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.nc.f32 	%f4, [%rd9];

BB2_4:
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r1, 4;
	add.s64 	%rd12, %rd10, %rd11;
	st.global.f32 	[%rd12], %f4;

BB2_5:
	ret;
}


`
)
