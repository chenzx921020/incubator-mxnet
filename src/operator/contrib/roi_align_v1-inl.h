/*!
 * \file roi_align-inl.h
 * \brief roi align operator, based on roi pooling operator
 * \author Hongyu Xie
 */

#ifndef MXNET_OPERATOR_CONTRIB_ROI_ALIGN_INL_H_
#define MXNET_OPERATOR_CONTRIB_ROI_ALIGN_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../mshadow_op.h"
#include "../operator_common.h"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace mxnet {
namespace op {

namespace roi_align_v1 {
enum ROIAlignOpInputs {kData, kBox};
enum ROIAlignOpOutputs {kOut, kMaxIdx, kWeight};
} // namespace roi_align_v1

struct ROIAlignV1Param : public dmlc::Parameter<ROIAlignV1Param> {
	TShape pooled_size;
	float spatial_scale;
	float feat_stride;
	DMLC_DECLARE_PARAMETER(ROIAlignV1Param) {
			DMLC_DECLARE_FIELD(pooled_size).set_expect_ndim(2).enforce_nonzero()
			.describe("ROI pooling output shape (h,w) ");
			DMLC_DECLARE_FIELD(spatial_scale).set_range(0.0, 1.0)
			.describe("Ratio of input feature map height (or w) to raw image height (or w). "
			"Equals the reciprocal of total stride in convolutional layers");
			DMLC_DECLARE_FIELD(feat_stride).set_lower_bound(0.0f).set_default(0.01f)
			.describe("Stride in the feature map");
	}
};

template<typename xpu, typename DType>
class ROIAlignV1Op : public Operator {
	public:
		explicit ROIAlignV1Op(ROIAlignV1Param p) {
			this->param_ = p;
		}

		void Forward(const OpContext& ctx,
		             const std::vector<TBlob>& in_data,
		             const std::vector<OpReqType>& req,
		             const std::vector<TBlob>& out_data,
		             const std::vector<TBlob>& aux_args) override {
			using namespace mshadow;
			CHECK_EQ(in_data.size(), 2U) << "Input: [data, rois]";
			CHECK_EQ(out_data.size(), 3U) << "Output: [output, maxidx, weight]";
			CHECK_EQ(out_data[roi_align_v1::kOut].shape_[0], in_data[roi_align_v1::kBox].shape_[0]);
			CHECK_EQ(out_data[roi_align_v1::kMaxIdx].shape_[0], in_data[roi_align_v1::kBox].shape_[0]);
			CHECK_EQ(out_data[roi_align_v1::kWeight].shape_[0], in_data[roi_align_v1::kBox].shape_[0]);
			Stream<xpu> *s = ctx.get_stream<xpu>();

			Tensor<xpu, 4, DType> data = in_data[roi_align_v1::kData].get<xpu, 4, DType>(s);
			Tensor<xpu, 2, DType> bbox = in_data[roi_align_v1::kBox].get<xpu, 2, DType>(s);
			Tensor<xpu, 4, DType> out = out_data[roi_align_v1::kOut].get<xpu, 4, DType>(s);
			Tensor<xpu, 5, DType> max_idx = out_data[roi_align_v1::kMaxIdx].get<xpu, 5, DType>(s);
			Tensor<xpu, 5, DType> weight = out_data[roi_align_v1::kWeight].get<xpu, 5, DType>(s);

			CHECK(data.CheckContiguous());
			CHECK(bbox.CheckContiguous());
			CHECK(out.CheckContiguous());
			CHECK(max_idx.CheckContiguous());
			CHECK(weight.CheckContiguous());

			ROIAlignV1Forward(data, bbox, out, max_idx, weight, param_.spatial_scale, param_.feat_stride);
		}

		void Backward(const OpContext& ctx,
		              const std::vector<TBlob>& out_grad,
		              const std::vector<TBlob>& in_data,
		              const std::vector<TBlob>& out_data,
		              const std::vector<OpReqType>& req,
		              const std::vector<TBlob>& in_grad,
		              const std::vector<TBlob>& aux_args) override {
			using namespace mshadow;
			CHECK_EQ(in_data.size(), 2U);
			CHECK_EQ(out_data.size(), 3U);
			CHECK_EQ(out_grad[roi_align_v1::kOut].shape_[0], in_data[roi_align_v1::kBox].shape_[0]);
			CHECK_EQ(out_data[roi_align_v1::kMaxIdx].shape_[0], in_data[roi_align_v1::kBox].shape_[0]);
			CHECK_EQ(out_data[roi_align_v1::kWeight].shape_[0], in_data[roi_align_v1::kBox].shape_[0]);
			CHECK_NE(req[roi_align_v1::kData], kWriteInplace) << "ROIAlign: Backward doesn't support kWriteInplace.";
			CHECK_NE(req[roi_align_v1::kBox], kWriteInplace) << "ROIAlign: Backward doesn't support kWriteInplace.";

			Stream<xpu>* s = ctx.get_stream<xpu>();

			Tensor<xpu, 4, DType> grad_out = out_grad[roi_align_v1::kOut].get<xpu, 4, DType>(s);
			Tensor<xpu, 2, DType> bbox = in_data[roi_align_v1::kBox].get<xpu, 2, DType>(s);
			Tensor<xpu, 5, DType> max_idx = out_data[roi_align_v1::kMaxIdx].get<xpu, 5, DType>(s);
			Tensor<xpu, 5, DType> weight = out_data[roi_align_v1::kWeight].get<xpu, 5, DType>(s);
			Tensor<xpu, 4, DType> grad_in = in_grad[roi_align_v1::kData].get<xpu, 4, DType>(s);
			Tensor<xpu, 2, DType> grad_roi = in_grad[roi_align_v1::kBox].get<xpu, 2, DType>(s);

			CHECK(grad_out.CheckContiguous());
			CHECK(bbox.CheckContiguous());
			CHECK(max_idx.CheckContiguous());
			CHECK(weight.CheckContiguous());
			CHECK(grad_in.CheckContiguous());

			if (kAddTo == req[roi_align_v1::kData] || kWriteTo == req[roi_align_v1::kData]) {
				if (kWriteTo == req[roi_align_v1::kData]) {
					grad_in = 0.0f;
				}
				ROIAlignV1BackwardAcc(grad_in, grad_out, bbox, max_idx, weight);
			}
			if (kWriteTo == req[roi_align_v1::kBox]) {
				grad_roi = 0.0f;
			}
		}

	private:
		ROIAlignV1Param param_;
}; // class ROIAlignV1Op

template<typename xpu>
Operator* CreateOp(ROIAlignV1Param param, int dtype);

#ifdef DMLC_USE_CXX11
class ROIAlignV1Prop : public OperatorProperty {
	public:
		std::vector<std::string> ListArguments() const override {
			return {"data", "rois"};
		}

		std::vector<std::string> ListOutputs() const override {
			return {"output", "maxidx", "weight"};
		}

		int NumOutputs() const override {
			return 3;
		}

		int NumVisibleOutputs() const override {
			return 1;
		}

		void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
			param_.Init(kwargs);
		}

		std::map<std::string, std::string> GetParams() const override {
			return param_.__DICT__();
		}

		bool InferShape(std::vector<TShape>* in_shape,
		                std::vector<TShape>* out_shape,
		                std::vector<TShape>* aux_shape) const override {
			using namespace mshadow;
			CHECK_EQ(in_shape->size(), 2U) << "Input: [data, rois]";

			// data: [batch_size, c, h, w]
			TShape dshape = in_shape->at(roi_align_v1::kData);
			CHECK_EQ(dshape.ndim(), 4U) << "data should be a 4D tensor (NCHW)";

			// bbox: [num_rois, 5]
			TShape bshape = in_shape->at(roi_align_v1::kBox);
			CHECK_EQ(bshape.ndim(), 2U) << "bbox should be a 2D tensor of shape [batch, 5]";
			CHECK_EQ(bshape[1], 5U) << "bbox should be a 2D tensor of shape [batch, 5]";

			// out: [num_rois, c, pooled_h, pooled_w]
			// max_idx: [num_rois, c, 4, pooled_h, pooled_w], where 4 = lt + lb + rt + rb
			out_shape->clear();
			out_shape->push_back(Shape4(bshape[0], dshape[1], param_.pooled_size[0], param_.pooled_size[1]));
			out_shape->push_back(Shape5(bshape[0], dshape[1], 4, param_.pooled_size[0], param_.pooled_size[1]));
			out_shape->push_back(Shape5(bshape[0], dshape[1], 4, param_.pooled_size[0], param_.pooled_size[1]));
			return true;
		}

		bool InferType(std::vector<int>* in_type,
		               std::vector<int>* out_type,
		               std::vector<int>* aux_type) const override {
			CHECK_EQ(in_type->size(), 2U);
			int dtype = (*in_type)[0];
			CHECK_EQ(dtype, (*in_type)[1]);
			CHECK_NE(dtype, -1) << "Input must have specified type";

			out_type->clear();
			out_type->push_back(dtype);
			out_type->push_back(dtype);
			out_type->push_back(dtype);
			return true;
		}

		OperatorProperty* Copy() const override {
			auto ptr = new ROIAlignV1Prop();
			ptr->param_ = this->param_;
			return ptr;
		}

		std::string TypeString() const override {
			return "_contrib_ROIAlign_v1";
		}

		std::vector<int> DeclareBackwardDependency(const std::vector<int>& out_grad,
		                                           const std::vector<int>& in_data,
		                                           const std::vector<int>& out_data) const override {
			return {out_grad[roi_align_v1::kOut], in_data[roi_align_v1::kBox],
			        out_data[roi_align_v1::kMaxIdx], out_data[roi_align_v1::kWeight]};
		}

		Operator* CreateOperator(Context ctx) const override {
			LOG(FATAL) << "Not Implemented";
			return nullptr;
		}

		Operator* CreateOperatorEx(Context ctx, std::vector<TShape>* in_shape,
		                           std::vector<int>* in_type) const override;

	private:
		ROIAlignV1Param param_;
}; // class ROIAlignProp
#endif // DMLC_USE_CXX11

} // namespace op
} // namespace mxnet

#endif // MXNET_OPERATOR_CONTRIB_ROI_ALIGN_INL_H_
