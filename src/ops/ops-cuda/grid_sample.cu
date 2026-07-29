#include "ops_cuda_common.cuh"
#include <cuda_bf16.h>

namespace ggml_ops_ext::cuda {
namespace {
template <typename T> __device__ float cvt(T v) { return float(v); }
template <> __device__ float cvt<half>(half v) { return __half2float(v); }
template <> __device__ float cvt<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ T out(float v) { return T(v); }
template <> __device__ half out<half>(float v) { return __float2half(v); }
template <> __device__ __nv_bfloat16 out<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
struct s4 { size_t v[4]; };
__device__ float unnorm(float v, int64_t size, bool align) { return align ? (v+1)*(size-1)*.5f : ((v+1)*size-1)*.5f; }
__device__ float reflect(float v, int64_t size, bool align) {
    const float lo=align?0.f:-.5f, hi=align?float(size-1):float(size)-.5f, span=hi-lo;
    if (span<=0) return 0; v=fabsf(v-lo); const float extra=fmodf(v,span); const int flips=int(floorf(v/span));
    return (flips&1?span-extra:extra)+lo;
}
template <typename T> __device__ float sample(const T* input,int64_t x,int64_t y,int64_t c,int64_t n,
                                               ops_grid_sample_2d_desc d,s4 s) {
    if(d.padding==ops_grid_padding_mode::border){x=x<0?0:(x>=d.input_width?d.input_width-1:x);y=y<0?0:(y>=d.input_height?d.input_height-1:y);}
    else if(x<0||x>=d.input_width||y<0||y>=d.input_height)return 0;
    return cvt(*reinterpret_cast<const T*>(reinterpret_cast<const char*>(input)+x*s.v[0]+y*s.v[1]+c*s.v[2]+n*s.v[3]));
}
template <typename T,typename G> __global__ void kernel(const T* input,const G* grid,T* output,
 ops_grid_sample_2d_desc d,s4 is,s4 gs,s4 os,int64_t total){
    int64_t idx=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(idx>=total)return;int64_t q=idx;
    int64_t ox=q%d.output_width;q/=d.output_width;int64_t oy=q%d.output_height;q/=d.output_height;int64_t c=q%d.channels,n=q/d.channels;
    size_t gb=ox*gs.v[1]+oy*gs.v[2]+n*gs.v[3];float x=unnorm(cvt(*reinterpret_cast<const G*>(reinterpret_cast<const char*>(grid)+gb)),d.input_width,d.align_corners);
    float y=unnorm(cvt(*reinterpret_cast<const G*>(reinterpret_cast<const char*>(grid)+gb+gs.v[0])),d.input_height,d.align_corners);
    if(d.padding==ops_grid_padding_mode::reflection){x=fminf(fmaxf(reflect(x,d.input_width,d.align_corners),0),d.input_width-1);y=fminf(fmaxf(reflect(y,d.input_height,d.align_corners),0),d.input_height-1);}
    float value;if(d.mode==ops_grid_sample_mode::nearest)value=sample(input,int64_t(nearbyintf(x)),int64_t(nearbyintf(y)),c,n,d,is);
    else{int64_t x0=int64_t(floorf(x)),y0=int64_t(floorf(y));float wx=x-x0,wy=y-y0;value=sample(input,x0,y0,c,n,d,is)*(1-wx)*(1-wy)+sample(input,x0+1,y0,c,n,d,is)*wx*(1-wy)+sample(input,x0,y0+1,c,n,d,is)*(1-wx)*wy+sample(input,x0+1,y0+1,c,n,d,is)*wx*wy;}
    *reinterpret_cast<T*>(reinterpret_cast<char*>(output)+ox*os.v[0]+oy*os.v[1]+c*os.v[2]+n*os.v[3])=out<T>(value);
}
template<typename T,typename G> bool launch(cudaStream_t stream,ggml_tensor* o,const ggml_tensor* i,const ggml_tensor* g,const ops_grid_sample_2d_desc& d){
 s4 is={{i->nb[0],i->nb[1],i->nb[2],i->nb[3]}},gs={{g->nb[0],g->nb[1],g->nb[2],g->nb[3]}},os={{o->nb[0],o->nb[1],o->nb[2],o->nb[3]}};int64_t total=ggml_nelements(o);
 kernel<T,G><<<unsigned((total+255)/256),256,0,stream>>>((const T*)i->data,(const G*)g->data,(T*)o->data,d,is,gs,os,total);return cudaGetLastError()==cudaSuccess;}
template<typename T> bool grid(cudaStream_t s,ggml_tensor*o,const ggml_tensor*i,const ggml_tensor*g,const ops_grid_sample_2d_desc&d){return g->type==GGML_TYPE_F32?launch<T,float>(s,o,i,g,d):g->type==GGML_TYPE_F16?launch<T,half>(s,o,i,g,d):false;}
}
bool ggml_cuda_op_grid_sample_2d_entry(ggml_backend_t b,ggml_tensor*n){if(!n||!n->src[0]||!n->src[1])return false;ops_grid_sample_2d_params p{};memcpy(&p,n->op_params,sizeof(p));ggml_tensor*s[]={n->src[0],n->src[1]};ops_request r={ggml_backend_get_device(b),(int)n->op,s,2,&p,sizeof(p),n};ops_grid_sample_2d_desc d;if(!ops_validate_grid_sample_2d(r,&d))return false;auto stream=ggml_ops_ext_bridge_cuda_get_stream(b);switch(s[0]->type){case GGML_TYPE_F32:return grid<float>(stream,n,s[0],s[1],d);case GGML_TYPE_F16:return grid<half>(stream,n,s[0],s[1],d);case GGML_TYPE_BF16:return grid<__nv_bfloat16>(stream,n,s[0],s[1],d);default:return false;}}
}
