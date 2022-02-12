#include <paddle/extension.h>
#include <vector>
#include <iostream>

std::vector<paddle::Tensor> forward_rasterize_cuda(
        paddle::Tensor face_vertices,
        paddle::Tensor depth_buffer,
        paddle::Tensor triangle_buffer,
        paddle::Tensor baryw_buffer,
        int h,
        int w);

std::vector<paddle::Tensor> standard_rasterize(
        const paddle::Tensor &face_vertices,
        const paddle::Tensor &depth_buffer,
        const paddle::Tensor &triangle_buffer,
        const paddle::Tensor &baryw_buffer,
        int height, int width
        ) {
    return forward_rasterize_cuda(
        const_cast<paddle::Tensor&>(face_vertices), 
        const_cast<paddle::Tensor&>(depth_buffer), 
        const_cast<paddle::Tensor&>(triangle_buffer), 
        const_cast<paddle::Tensor&>(baryw_buffer), 
        height, width);
}

std::vector<std::vector<int64_t>> standard_rasterize_inferShape(
    std::vector<int64_t> face_vertices_shape,
    std::vector<int64_t> depth_buffer_shape,
    std::vector<int64_t> triangle_buffer_shape,
    std::vector<int64_t> baryw_buffer_shape) {
    return {depth_buffer_shape, triangle_buffer_shape, baryw_buffer_shape};
}

std::vector<paddle::DataType> standard_rasterize_inferDtype(
    paddle::DataType face_vertices_dtype,
    paddle::DataType depth_buffer_dtype,
    paddle::DataType triangle_buffer_dtype,
    paddle::DataType baryw_buffer_dtype) {
    return {depth_buffer_dtype, triangle_buffer_dtype, baryw_buffer_dtype};
}

std::vector<paddle::Tensor> forward_rasterize_colors_cuda(
        paddle::Tensor face_vertices,
        paddle::Tensor face_colors,
        paddle::Tensor depth_buffer,
        paddle::Tensor triangle_buffer,
        paddle::Tensor images,
        int h,
        int w);

std::vector<paddle::Tensor> standard_rasterize_colors(
        const paddle::Tensor &face_vertices,
        const paddle::Tensor &face_colors,
        const paddle::Tensor &depth_buffer,
        const paddle::Tensor &triangle_buffer,
        const paddle::Tensor &images,
        int height, int width
        ) {
    return forward_rasterize_colors_cuda(
        const_cast<paddle::Tensor&>(face_vertices), 
        const_cast<paddle::Tensor&>(face_colors), 
        const_cast<paddle::Tensor&>(depth_buffer), 
        const_cast<paddle::Tensor&>(triangle_buffer), 
        const_cast<paddle::Tensor&>(images), 
        height, width);
}

std::vector<std::vector<int64_t>> standard_rasterize_colors_inferShape(
    std::vector<int64_t> face_vertices_shape,
    std::vector<int64_t> face_colors_shape,
    std::vector<int64_t> depth_buffer_shape,
    std::vector<int64_t> triangle_buffer_shape,
    std::vector<int64_t> baryw_buffer_shape) {
    return {depth_buffer_shape, triangle_buffer_shape, baryw_buffer_shape};
}

std::vector<paddle::DataType> standard_rasterize_colors_inferDtype(
    paddle::DataType face_vertices_dtype,
    paddle::DataType face_colors_dtype,
    paddle::DataType depth_buffer_dtype,
    paddle::DataType triangle_buffer_dtype,
    paddle::DataType baryw_buffer_dtype) {
    return {depth_buffer_dtype, triangle_buffer_dtype, baryw_buffer_dtype};
}

PD_BUILD_OP(standard_rasterize)
    .Inputs({"face_vertices", "depth_buffer", "triangle_buffer", "baryw_buffer"})
    .Outputs({"depth_buffer_out", "triangle_buffer_out", "baryw_buffer_out"})
    .Attrs({
        "height: int",
        "width: int"
    })
    .SetKernelFn(PD_KERNEL(standard_rasterize))
    .SetInferShapeFn(PD_INFER_SHAPE(standard_rasterize_inferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(standard_rasterize_inferDtype));


PD_BUILD_OP(standard_rasterize_colors)
    .Inputs({"face_vertices", "face_colors", "depth_buffer", "triangle_buffer", "baryw_buffer"})
    .Outputs({"depth_buffer_out", "triangle_buffer_out", "baryw_buffer_out"})
    .Attrs({
        "height: int",
        "width: int"
    })
    .SetKernelFn(PD_KERNEL(standard_rasterize_colors))
    .SetInferShapeFn(PD_INFER_SHAPE(standard_rasterize_colors_inferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(standard_rasterize_colors_inferDtype));
