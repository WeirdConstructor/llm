/* automatically generated by rust-bindgen 0.65.1 */

extern "C" {
    pub fn ggml_cl_init();
}
extern "C" {
    pub fn ggml_cl_mul(src0: *const ggml_tensor, src1: *const ggml_tensor, dst: *mut ggml_tensor);
}
extern "C" {
    pub fn ggml_cl_can_mul_mat(
        src0: *const ggml_tensor,
        src1: *const ggml_tensor,
        dst: *mut ggml_tensor,
    ) -> bool;
}
extern "C" {
    pub fn ggml_cl_mul_mat_get_wsize(
        src0: *const ggml_tensor,
        src1: *const ggml_tensor,
        dst: *mut ggml_tensor,
    ) -> usize;
}
extern "C" {
    pub fn ggml_cl_mul_mat(
        src0: *const ggml_tensor,
        src1: *const ggml_tensor,
        dst: *mut ggml_tensor,
        wdata: *mut ::std::os::raw::c_void,
        wsize: usize,
    );
}
extern "C" {
    pub fn ggml_cl_host_malloc(size: usize) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    pub fn ggml_cl_host_free(ptr: *mut ::std::os::raw::c_void);
}
extern "C" {
    pub fn ggml_cl_free_data(tensor: *const ggml_tensor);
}
extern "C" {
    pub fn ggml_cl_transform_tensor(tensor: *mut ggml_tensor);
}
extern "C" {
    pub fn ggml_cl_load_data(
        fname: *const ::std::os::raw::c_char,
        tensor: *mut ggml_tensor,
        offset: usize,
    );
}
