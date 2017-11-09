/*!
 * \file roi_align.cu
 * \brief gpu implementation of roi align operator, based on roi_pooling.cu
 * \author Hongyu Xie
 */

#include "./roi_align_v1-inl.h"
#include "../../common/cuda_utils.h"
#include "../mxnet_op.h"

namespace mshadow {
namespace cuda {

__device__ inline bool InImage(int x, int y, int width, int height) {
	return x >= 0 && x < width && y >= 0 && y < height;
}

template<typename DType>
__global__ void ROIAlignV1ForwardKernel(const Tensor<gpu, 4, DType> data,
                                        const Tensor<gpu, 2, DType> bbox,
                                        Tensor<gpu, 4, DType> out,
                                        Tensor<gpu, 5, DType> max_idx,
                                        Tensor<gpu, 5, DType> weight,
                                        const float spatial_scale,
                                        const float feat_stride_) {
	const int maximum_index = out.shape_.Size();
	const int channels = data.size(1);
	const int feat_height = data.size(2);
	const int feat_width = data.size(3);

	const int pooled_height = out.size(2);
	const int pooled_width = out.size(3);

	CUDA_KERNEL_LOOP(index, maximum_index){
		int pw = index % pooled_width;
		int ph = (index / pooled_width) % pooled_height;
		int c = (index / pooled_width / pooled_height) % channels;
		int n = index / pooled_width / pooled_height / channels;

		auto roi_batch_ind = static_cast<int>(bbox[n][0]);

		DType roi_start_w = bbox[n][1] * spatial_scale;
		DType roi_start_h = bbox[n][2] * spatial_scale;
		DType roi_end_w = bbox[n][3] * spatial_scale;
		DType roi_end_h = bbox[n][4] * spatial_scale;

		// DType roi_width = fmaxf(roi_end_w - roi_start_w, 0.0f);
		// DType roi_height = fmaxf(roi_end_h - roi_start_h, 0.0f);

		DType roi_width = fmaxf(roi_end_w - roi_start_w, 1.0f);
		DType roi_height = fmaxf(roi_end_h - roi_start_h, 1.0f);
        
		const DType bin_size_w = roi_width / static_cast<DType>(pooled_width);
		const DType bin_size_h = roi_height / static_cast<DType>(pooled_height);

		DType w_start = static_cast<DType>(pw) * bin_size_w;
		DType h_start = static_cast<DType>(ph) * bin_size_h;
		DType w_end = static_cast<DType>(pw + 1) * bin_size_w;
		DType h_end = static_cast<DType>(ph + 1) * bin_size_h;

		w_start = fminf(fmaxf(w_start + roi_start_w, 0.0f), feat_width);
		h_start = fminf(fmaxf(h_start + roi_start_h, 0.0f), feat_height);
		w_end = fminf(fmaxf(w_end + roi_start_w, 0.0f), feat_width);
		h_end = fminf(fmaxf(h_end + roi_start_h, 0.0f), feat_height);

		bool is_empty = (h_end <= h_start) || (w_end <= w_start);

		DType max_val = is_empty ? 0 : -FLT_MAX;
		int maxidx_0 = -1, maxidx_1 = -1, maxidx_2 = -1, maxidx_3 = -1;
		DType weight_0 = 0, weight_1 = 0, weight_2 = 0, weight_3 = 0;

		for (DType h = h_start; h < h_end; h += feat_stride_) {
			for (DType w = w_start; w < w_end; w += feat_stride_) {
				int lt_x = floor(w);
				int lt_y = floor(h);
				int rb_x = ceil(w);
				int rb_y = ceil(h);

				bool is_lt_in = InImage(lt_x, lt_y, feat_width, feat_height);
				bool is_lb_in = InImage(lt_x, rb_y, feat_width, feat_height);
				bool is_rt_in = InImage(rb_x, lt_y, feat_width, feat_height);
				bool is_rb_in = InImage(rb_x, rb_y, feat_width, feat_height);

				DType val = 0;
				DType w_0 = (1.0 - w + lt_x);
				DType w_1 = (1.0 - h + lt_y);
				DType w_2 = (1.0 - rb_x + w);
				DType w_3 = (1.0 - rb_y + h);

				if (is_lt_in) {
					val += w_0 * w_1 * data[roi_batch_ind][c][lt_y][lt_x];
				}
				if (is_lb_in) {
					val += w_0 * w_3 * data[roi_batch_ind][c][rb_y][lt_x];
				}
				if (is_rt_in) {
					val += w_2 * w_1 * data[roi_batch_ind][c][lt_y][rb_x];
				}
				if (is_rb_in) {
					val += w_2 * w_3 * data[roi_batch_ind][c][rb_y][rb_x];
				}
				if (val > max_val) {
					max_val = val;
					maxidx_0 = lt_x;
					maxidx_1 = lt_y;
					maxidx_2 = rb_x;
					maxidx_3 = rb_y;
					weight_0 = w_0 * w_1;
					weight_1 = w_0 * w_3;
					weight_2 = w_2 * w_1;
					weight_3 = w_2 * w_3;
				}
			} // end for (DType w = w_start; w < w_end; w += feat_stride)
		} // end for (DType h = h_start; h < h_end; h += feat_stride)

		out[n][c][ph][pw] = max_val;
		max_idx[n][c][0][ph][pw] = maxidx_0;
		max_idx[n][c][1][ph][pw] = maxidx_1;
		max_idx[n][c][2][ph][pw] = maxidx_2;
		max_idx[n][c][3][ph][pw] = maxidx_3;
		weight[n][c][0][ph][pw] = weight_0;
		weight[n][c][1][ph][pw] = weight_1;
		weight[n][c][2][ph][pw] = weight_2;
		weight[n][c][3][ph][pw] = weight_3;
	} // end for (int index = ...)

} // ROIAlignForwardKernel

template<typename DType>
__global__ void ROIAlignV1BackwardAccKernel(Tensor<gpu, 4, DType> in_grad,
                                            const Tensor<gpu, 4, DType> out_grad,
                                            const Tensor<gpu, 2, DType> bbox,
                                            const Tensor<gpu, 5, DType> max_idx,
                                            const Tensor<gpu, 5, DType> weight) {

	const int maximum_index = out_grad.shape_.Size();
	const int channels = in_grad.size(1);
	const int feat_height = in_grad.size(2);
	const int feat_width = in_grad.size(3);

	const int pooled_height = out_grad.size(2);
	const int pooled_width = out_grad.size(3);

	CUDA_KERNEL_LOOP(index, maximum_index){
		int pw = index % pooled_width;
		int ph = (index / pooled_width) % pooled_height;
		int c = (index / pooled_width / pooled_height) % channels;
		int n = index / pooled_width / pooled_height / channels;

		int roi_batch_ind = static_cast<int>(bbox[n][0]);

		const auto lt_x = static_cast<int>(max_idx[n][c][0][ph][pw]);
		const auto lt_y = static_cast<int>(max_idx[n][c][1][ph][pw]);
		const auto rb_x = static_cast<int>(max_idx[n][c][2][ph][pw]);
		const auto rb_y = static_cast<int>(max_idx[n][c][3][ph][pw]);
		const DType weight_0 = weight[n][c][0][ph][pw];
		const DType weight_1 = weight[n][c][1][ph][pw];
		const DType weight_2 = weight[n][c][2][ph][pw];
		const DType weight_3 = weight[n][c][3][ph][pw];

		bool is_lt_in = InImage(lt_x, lt_y, feat_width, feat_height);
		bool is_lb_in = InImage(lt_x, rb_y, feat_width, feat_height);
		bool is_rt_in = InImage(rb_x, lt_y, feat_width, feat_height);
		bool is_rb_in = InImage(rb_x, rb_y, feat_width, feat_height);

		if (is_lt_in) {
			atomicAdd(&in_grad[roi_batch_ind][c][lt_y][lt_x], weight_0 * out_grad[n][c][ph][pw]);
		}
		if (is_lb_in) {
			atomicAdd(&in_grad[roi_batch_ind][c][rb_y][lt_x], weight_1 * out_grad[n][c][ph][pw]);
		}
		if (is_rt_in) {
			atomicAdd(&in_grad[roi_batch_ind][c][lt_y][rb_x], weight_2 * out_grad[n][c][ph][pw]);
		}
		if (is_rb_in) {
			atomicAdd(&in_grad[roi_batch_ind][c][rb_y][rb_x], weight_3 * out_grad[n][c][ph][pw]);
		}
	}
} // ROIAlignBackwardAccKernel

} // namespace cuda

template<typename DType>
inline void ROIAlignV1Forward(const Tensor<gpu, 4, DType>& data,
                              const Tensor<gpu, 2, DType>& bbox,
                              Tensor<gpu, 4, DType>& out,
                              Tensor<gpu, 5, DType>& max_idx,
                              Tensor<gpu, 5, DType>& weight,
                              const float spatial_stride_,
                              const float feat_stride_) {
	using namespace cuda;
	const int count = out.shape_.Size();
	dim3 dimGrid(mxnet::op::mxnet_op::cuda_get_num_blocks(count));
	dim3 dimBlock(kBaseThreadNum);
	CheckLaunchParam(dimGrid, dimBlock, "ROIAlign Forward");
	cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
	ROIAlignV1ForwardKernel<DType><<<dimGrid, dimBlock, 0, stream>>>(
			data, bbox, out, max_idx, weight,
			spatial_stride_, feat_stride_
	);
	MSHADOW_CUDA_POST_KERNEL_CHECK(ROIAlignV1ForwardKernel);
} // ROIAlignForward

template<typename DType>
inline void ROIAlignV1BackwardAcc(Tensor<gpu, 4, DType>& in_grad,
                                  const Tensor<gpu, 4, DType>& out_grad,
                                  const Tensor<gpu, 2, DType>& bbox,
                                  const Tensor<gpu, 5, DType>& max_idx,
                                  const Tensor<gpu, 5 ,DType>& weight) {
	using namespace cuda;
	const int count = out_grad.shape_.Size();
	dim3 dimGrid(mxnet::op::mxnet_op::cuda_get_num_blocks(count));
	dim3 dimBlock(kBaseThreadNum);
	CheckLaunchParam(dimGrid, dimBlock, "ROIAlign Backward");
	cudaStream_t stream = Stream<gpu>::GetStream(in_grad.stream_);
	ROIAlignV1BackwardAccKernel<DType><<<dimGrid, dimBlock, 0, stream>>>(
			in_grad, out_grad, bbox, max_idx, weight
	);
	MSHADOW_CUDA_POST_KERNEL_CHECK(ROIAlignV1BackwardAccKernel);
} // ROIAlignBackwardAcc

} // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(ROIAlignV1Param param, int dtype) {
	Operator* op = nullptr;
	MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
			op = new ROIAlignV1Op<gpu, DType>(param);
	});
	return op;
}

} // namespace op
} // namespace mxnet

