#include "ops/ops.h"
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#ifdef _WIN32
#define OPS_IMPORT extern "C" __declspec(dllimport)
#else
#define OPS_IMPORT extern "C"
#endif
OPS_IMPORT void ggml_ops_ext_cpu_init();
#ifdef GGML_USE_CUDA
OPS_IMPORT void ggml_ops_ext_cuda_init();
#endif
#ifdef GGML_USE_SYCL
OPS_IMPORT void ggml_ops_ext_sycl_init();
#endif
bool run(ggml_backend_t b,const std::string&name,ggml_ops_ext::ops_grid_sample_mode mode,ggml_ops_ext::ops_grid_padding_mode pad){std::vector<float>input={1,2,3,4,5,6},grid={-1,-1,1,-1,-1,1,1,1},expected={1,3,4,6};ggml_context*c=ggml_init({1<<20,nullptr,true});auto*x=ggml_new_tensor_4d(c,GGML_TYPE_F32,3,2,1,1);auto*g=ggml_new_tensor_4d(c,GGML_TYPE_F32,2,2,2,1);ggml_ops_ext::ops_grid_sample_2d_config cfg;cfg.mode=mode;cfg.padding=pad;cfg.align_corners=true;auto*y=ggml_ops_grid_sample_2d(c,x,g,cfg,b);if(!y){ggml_free(c);return false;}auto buf=ggml_backend_alloc_ctx_tensors(c,b);ggml_backend_tensor_set(x,input.data(),0,input.size()*4);ggml_backend_tensor_set(g,grid.data(),0,grid.size()*4);auto*graph=ggml_new_graph(c);ggml_build_forward_expand(graph,y);bool ok=ggml_ops_ext::ops_backend_graph_compute(b,graph)==GGML_STATUS_SUCCESS;std::vector<float>a(4);if(ok)ggml_backend_tensor_get(y,a.data(),0,16);for(int i=0;i<4;i++)ok&=std::abs(a[i]-expected[i])<1e-5f;std::cout<<name<<" grid_sample mode="<<(int)mode<<" pad="<<(int)pad<<(ok?" PASSED\n":" FAILED\n");ggml_backend_buffer_free(buf);ggml_free(c);return ok;}
int main(){ggml_ops_ext_cpu_init();
#ifdef GGML_USE_CUDA
ggml_ops_ext_cuda_init();
#endif
#ifdef GGML_USE_SYCL
ggml_ops_ext_sycl_init();
#endif
ggml_backend_load_all();ggml_ops_ext::acquire_ops_hook();bool ok=true;for(size_t i=0;i<ggml_backend_dev_count();i++){auto d=ggml_backend_dev_get(i);std::string n=d?ggml_backend_dev_name(d):"";if(n.rfind("CPU",0)&&n.rfind("CUDA",0)&&n.rfind("SYCL",0))continue;auto b=ggml_backend_dev_init(d,nullptr);if(!b)continue;for(int m=0;m<2;m++)for(int p=0;p<3;p++)ok&=run(b,n,(ggml_ops_ext::ops_grid_sample_mode)m,(ggml_ops_ext::ops_grid_padding_mode)p);ggml_backend_free(b);}ggml_ops_ext::release_ops_hook();return ok?0:1;}
