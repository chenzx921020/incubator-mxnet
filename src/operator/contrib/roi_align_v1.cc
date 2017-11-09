/*!
 * \file roi_align.cc
 * \brief cpu implementation of roi align operator, based on roi_pooling.cc
 * \author Hongyu Xie
 */

#include "./roi_align_v1-inl.h"

namespace mshadow {

inline bool InImage(int x, int y, int width, int height) {
	return x >= 0 && x < width && y >= 0 && y < height;
}

template<typename DType>
inline void ROIAlignV1Forward(const Tensor<cpu, 4, DType>& data,
                              const Tensor<cpu, 2, DType>& bbox,
                              Tensor<cpu, 4, DType>& out,
                              Tensor<cpu, 5, DType>& max_idx,
                              Tensor<cpu, 5, DType>& weight,
                              const float spatial_scale_, const float feat_stride) {

	/*
	 * Follow the direction: left top -> left bottom -> right top -> right bottom
	 *  data: [batch, channels, height, width]
	 *  bbox: [num_rois, 5] (5 = batch_index + lt_x, lt_y, rb_x, rb_y)
	 *  out: [num_rois, channels, pooled_height, pooled_width]
	 *  max_idx: [num_rois, channels, 4, pooled_height, pooled_width] (4 = lt_x + lt_y + rb_x + rb_y)
	 *  weight: [num_rois, channels, 4, pooled_height, pooled_width] (4 = lt_x + lt_y + rb_x + rb_y)
	 */
	const int batch_size = data.size(0);
	const int channels = data.size(1);
	const int feat_height = data.size(2);
	const int feat_width = data.size(3);
	const int pooled_height = out.size(2);
	const int pooled_width = out.size(3);

	const int num_rois = bbox.size(0);

	#pragma omp parallel for
	for (int n = 0; n < num_rois; ++n) {
		auto roi_batch_ind = static_cast<int>(bbox[n][0]);
		DType roi_start_w = bbox[n][1] * spatial_scale_;
		DType roi_start_h = bbox[n][2] * spatial_scale_;
		DType roi_end_w = bbox[n][3] * spatial_scale_;
		DType roi_end_h = bbox[n][4] * spatial_scale_;

		CHECK_GE(roi_batch_ind, 0);
		CHECK_LT(roi_batch_ind, batch_size);

		// DType roi_height = fmaxf(roi_end_h - roi_start_h, 0.0f);
		// DType roi_width = fmaxf(roi_end_w - roi_start_w, 0.0f);

        DType roi_height = fmaxf(roi_end_h - roi_start_h, 1.0f);
        DType roi_width = fmaxf(roi_end_w - roi_start_w, 1.0f);

		const DType bin_size_h = roi_height / static_cast<DType>(pooled_height);
		const DType bin_size_w = roi_width / static_cast<DType>(pooled_width);

		for (int c = 0; c < channels; ++c) {
			for (int ph = 0; ph < pooled_height; ++ph) {
				for (int pw = 0; pw < pooled_width; ++pw) {
					DType h_start = static_cast<DType>(ph) * bin_size_h;
					DType w_start = static_cast<DType>(pw) * bin_size_w;
					DType h_end = static_cast<DType>(ph + 1) * bin_size_h;
					DType w_end = static_cast<DType>(pw + 1) * bin_size_w;

					h_start = fminf(fmaxf(h_start + roi_start_h, 0.0f), feat_height);
					h_end = fminf(fmaxf(h_end + roi_start_h, 0.0f), feat_height);
					w_start = fminf(fmaxf(w_start + roi_start_w, 0.0f), feat_width);
					w_end = fminf(fmaxf(w_end + roi_start_w, 0.0f), feat_width);

					bool is_empty = (h_end <= h_start) || (w_end <= w_start);

					DType max_val = is_empty ? 0 : -FLT_MAX;
					int maxidx_0 = -1, maxidx_1 = -1, maxidx_2 = -1, maxidx_3 = -1;
					DType weight_0 = 0, weight_1 = 0, weight_2 = 0, weight_3 = 0;

					// traverse a bin region to get maximum value
					for (DType h = h_start; h < h_end; h += feat_stride) {
						for (DType w = w_start; w < w_end; w += feat_stride) {
							auto lt_x = static_cast<int>(floor(w));
							auto lt_y = static_cast<int>(floor(h));
							auto rb_x = static_cast<int>(ceil(w));
							auto rb_y = static_cast<int>(ceil(h));

							bool is_lt_in = InImage(lt_x, lt_y, feat_width, feat_height);
							bool is_lb_in = InImage(lt_x, rb_y, feat_width, feat_height);
							bool is_rt_in = InImage(rb_x, lt_y, feat_width, feat_height);
							bool is_rb_in = InImage(rb_x, rb_y, feat_width, feat_height);

							DType val = 0.0f;
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
				} // end for (int pw = 0; pw < pooled_width; ++pw)
			} // end for (int ph = 0; ph < pooled_height; ++ph)
		} // end for (int c = 0; c < channels; ++c)
	} // end for (int n = 0; n < num_rois; ++n)

} // ROIAlignForward(...)

template<typename DType>
inline void ROIAlignV1BackwardAcc(Tensor<cpu, 4, DType>& in_grad,
                                  const Tensor<cpu, 4, DType>& out_grad,
                                  const Tensor<cpu, 2, DType>& bbox,
                                  const Tensor<cpu, 5, DType>& max_idx,
                                  const Tensor<cpu, 5, DType>& weight) {

	const int batch_size = in_grad.size(0);
	const int channels = in_grad.size(1);
	const int feat_height = in_grad.size(2);
	const int feat_width = in_grad.size(3);
	const int pooled_height = out_grad.size(2);
	const int pooled_width = out_grad.size(3);
	const int num_rois = bbox.size(0);

	for (int n = 0; n < num_rois; ++n) {
		for (int c = 0; c < channels; ++c) {
			for (int h = 0; h < pooled_height; ++h) {
				for (int w = 0; w < pooled_width; ++w) {
					const auto roi_batch_ind = static_cast<int>(bbox[n][0]);
					CHECK_GE(roi_batch_ind, 0);
					CHECK_LT(roi_batch_ind, batch_size);
					const auto lt_x = static_cast<int>(max_idx[n][c][0][h][w]);
					const auto lt_y = static_cast<int>(max_idx[n][c][1][h][w]);
					const auto rb_x = static_cast<int>(max_idx[n][c][2][h][w]);
					const auto rb_y = static_cast<int>(max_idx[n][c][3][h][w]);
					const DType weight_0 = weight[n][c][0][h][w];
					const DType weight_1 = weight[n][c][1][h][w];
					const DType weight_2 = weight[n][c][2][h][w];
					const DType weight_3 = weight[n][c][3][h][w];

					bool is_lt_in = InImage(lt_x, lt_y, feat_width, feat_height);
					bool is_lb_in = InImage(lt_x, rb_y, feat_width, feat_height);
					bool is_rt_in = InImage(rb_x, lt_y, feat_width, feat_height);
					bool is_rb_in = InImage(rb_x, rb_y, feat_width, feat_height);

					if (is_lt_in) {
						in_grad[roi_batch_ind][c][lt_y][lt_x] += weight_0 * out_grad[n][c][h][w];
					}
					if (is_lb_in) {
						in_grad[roi_batch_ind][c][rb_y][lt_x] += weight_1* out_grad[n][c][h][w];
					}
					if (is_rt_in) {
						in_grad[roi_batch_ind][c][lt_y][rb_x] += weight_2 * out_grad[n][c][h][w];
					}
					if (is_rb_in) {
						in_grad[roi_batch_ind][c][rb_y][rb_x] += weight_3 * out_grad[n][c][h][w];
					}

				} // end for (int w = 0; w < pooled_width; ++w)
			} // end for (int h = 0; h < pooled_height; ++h)
		} // end for (int c = 0; c < channels; ++c)
	} // end for (int n = 0; n < num_rois; ++n)
} // ROIAlignBackwardAcc(...)

} // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(ROIAlignV1Param param, int dtype) {
	Operator* op = nullptr;
	MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
		op = new ROIAlignV1Op<cpu, DType>(param);
	});
	return op;
}

Operator* ROIAlignV1Prop::CreateOperatorEx(Context ctx,
                                           std::vector<TShape>* in_shape,
                                           std::vector<int>* in_type) const {
	std::vector<TShape> out_shape, aux_shape;
	std::vector<int> out_type, aux_type;
	CHECK(InferType(in_type, &out_type, &aux_type));
	CHECK(InferShape(in_shape, &out_shape, &aux_shape));
	DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(ROIAlignV1Param);

MXNET_REGISTER_OP_PROPERTY(_contrib_ROIAlign_v1, ROIAlignV1Prop)
.describe(R"code(roi align)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "The input array, a 4D Feature Maps (NCHW)")
.add_argument("rois", "NDArray-or-Symbol", "The bounding box coordinates, a 2D array "
		"of shape (num_rois, 5), where 5 is [batch_index, x1, y1, x2, y2], where "
		"(x1, y1) is top left and (x2, y2) is right bottom. ``batch_index`` "
		"indicates the index of feature map in ``data``")
.add_arguments(ROIAlignV1Param::__FIELDS__());

} // namespace op
} // namespace mxnet
