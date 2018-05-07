#ifndef LAYERS_HPP
#define LAYERS_HPP

#include "tensor.hpp"
#include "function.hpp"

struct ConvDesc {
	hipdnnConvolutionDescriptor_t desc;

    ConvDesc(int pad_h, int pad_w, int u, int v, int upscalex, int upscaley) {
        CHECK_HIPDNN(hipdnnCreateConvolutionDescriptor(&desc));
        CHECK_HIPDNN(hipdnnSetConvolution2dDescriptor(desc, pad_h, pad_w, u, v, upscalex, upscaley, HIPDNN_CONVOLUTION));
    }

    // create with padding and stride, default upscale = 1
    ConvDesc(int pad_h, int pad_w, int u, int v) : ConvDesc(pad_h, pad_w, u, v, 1, 1) {
    }

    // default stride = 1, upscale = 1
    ConvDesc(int pad_h, int pad_w) : ConvDesc(pad_h, pad_w, 1, 1, 1, 1) {
    }

    // default pad = 0, stride = 1, upscale = 1
    ConvDesc() : ConvDesc(0, 0, 1, 1, 1, 1) {
    }

    ~ConvDesc() {
        CHECK_HIPDNN(hipdnnDestroyConvolutionDescriptor(desc));
    }
};

// parameters for a 2D convolutional layer
struct ConvLayerDesc {
    int batch_size;
    int height;
    int width;
    int channels_in;
    int channels_out;
    int kernel_size;
    int padding;
    int stride;
};


static Dim getConvOutputDim(int padding, int stride, const TensorDesc& input, const TensorDesc& weights) {
    int n, c, h, w;
    ConvDesc d(padding, padding, stride, stride, 1, 1);
    CHECK_HIPDNN(hipdnnGetConvolution2dForwardOutputDim(d.desc, input.desc, weights.desc, &n, &c, &h, &w));
    return Dim(n, c, h, w);
}

struct ConvLayer : public ConvDesc, public ConvLayerDesc, public Layer {
    Tensor weights;
    Tensor dweights;
    const Tensor* input_ref;

    // algorithm selection:
    hipdnnConvolutionFwdAlgo_t fwd_algo;
    hipdnnConvolutionBwdFilterAlgo_t bwd_weights_algo;
    hipdnnConvolutionBwdDataAlgo_t bwd_data_algo;


    virtual std::ostream& write_name(std::ostream& os) const {
        //return os << "Conv(" << kernel_size << "x" << kernel_size << ")";
        return os << "Conv(" << kernel_size << "x" << kernel_size << ",pad=" << padding << ",s=" << stride << ")";
    }

    ConvLayer(const TensorDesc& input_dims, int channels_out, int kernel_size, int padding, int stride)
        : ConvDesc(padding, padding, stride, stride, 1, 1),
          ConvLayerDesc({input_dims.n, input_dims.h, input_dims.w, input_dims.c, channels_out, kernel_size, padding, stride}),
          Layer((Dim&)input_dims, getConvOutputDim(padding, stride, input_dims, TensorDesc(channels_out, input_dims.c, kernel_size, kernel_size))),
          weights(channels_out, input_dims.c, kernel_size, kernel_size),
          dweights(channels_out, input_dims.c, kernel_size, kernel_size)
    {
    }

    /* default stride = 1 */
    ConvLayer(const TensorDesc& input_dims, int channels_out, int kernel_size, int padding)
        : ConvLayer(input_dims, channels_out, kernel_size, padding, 1) {}

    /* default padding = 0, stride = 1 */
    ConvLayer(const TensorDesc& input_dims, int channels_out, int kernel_size)
        : ConvLayer(input_dims, channels_out, kernel_size, 0, 1) {}

    /* construct via conv parameters */
    ConvLayer(const ConvLayerDesc& l)
        : ConvLayer(TensorDesc(l.batch_size, l.channels_in, l.height, l.width), l.channels_out, l.kernel_size, l.padding, l.stride) {}

    // estimate the number of muliplications for a direct implementation
    double num_flops() {
        return batch_size * 1.0 * height * width * channels_in * channels_out * kernel_size * kernel_size;
    }

    void init_forward(const Tensor& input, Tensor& output) override {
        size_t fwd_workspace_size;
        // PRNSOS: Note Passing uninitialized fwd_algo to hipdnnGetConvolutionForwardWorkspaceSize Since //in miopen, workspace size does not depend on algo
        CHECK_HIPDNN(hipdnnGetConvolutionForwardWorkspaceSize(hipdnn::handle(), input.desc, weights.desc, this->desc, output.desc, this->fwd_algo, &fwd_workspace_size));
        DEBUG("Init fwd " << *this << " req workspace: " << fwd_workspace_size);

        DevBuffer& buffer = WorkSpace::get(fwd_workspace_size);

        // find best algo, and benchmark!
        hipdnnConvolutionFwdAlgoPerf_t perfs[4];
        int returned_algos;
        CHECK_HIPDNN(hipdnnFindConvolutionForwardAlgorithmEx(hipdnn::handle(), input.desc, input.data, weights.desc, weights.data, this->desc, output.desc, output.data, 4, &returned_algos, perfs, buffer.data, fwd_workspace_size));

        INFO("\tHipDNN Found " << returned_algos << " fwd algorithms, choosing " << perfs[0].algo << ": ");
        for (int i = 0; i < returned_algos; ++i) {
            INFO("\t\t" << i << ") " << perfs[i].algo << " - time: " << perfs[i].time << ", Memory: " << perfs[i].memory);
        }

        fwd_algo = perfs[0].algo;
    }

    void find_bwd_data_algo(const Tensor& doutput, Tensor& dinput) {
        size_t bwd_data_workspace_size;
        // PRNSOS: Note Passing uninitialized fwd_algo to hipdnnGetConvolutionBackwardDataWorkspaceSize Since //in miopen, workspace size does not depend on algo
        CHECK_HIPDNN(hipdnnGetConvolutionBackwardDataWorkspaceSize(hipdnn::handle(), weights.desc, doutput.desc, this->desc, dinput.desc, this->bwd_data_algo, &bwd_data_workspace_size));
        DEBUG("Init bwd_data " << *this << " req workspace: " << bwd_data_workspace_size);

        DevBuffer& buffer = WorkSpace::get(bwd_data_workspace_size);

        // find best algo, and benchmark!
        hipdnnConvolutionBwdDataAlgoPerf_t perfs[5];
        int returned_algos;
        CHECK_HIPDNN(hipdnnFindConvolutionBackwardDataAlgorithmEx(hipdnn::handle(), weights.desc, weights.data, doutput.desc, doutput.data, this->desc, dinput.desc, dinput.data, 5, &returned_algos, perfs, buffer.data, bwd_data_workspace_size));

        INFO("\tHipDNN Found " << returned_algos << " bwd_data algorithms, choosing " << perfs[0].algo << ": ");
        for (int i = 0; i < returned_algos; ++i) {
            INFO("\t\t" << i << ") " << perfs[i].algo << " - time: " << perfs[i].time << ", Memory: " << perfs[i].memory);
        }

        bwd_data_algo = perfs[0].algo;
    }

    void find_bwd_weights_algo(const Tensor& doutput, Tensor& input) {
        size_t bwd_weights_workspace_size;
        // PRNSOS: Note Passing uninitialized fwd_algo to hipdnnGetConvolutionBackwardFilterWorkspaceSize Since //in miopen, workspace size does not depend on algo
        CHECK_HIPDNN(hipdnnGetConvolutionBackwardFilterWorkspaceSize(hipdnn::handle(), input.desc,  doutput.desc, this->desc, weights.desc, this->bwd_weights_algo,  &bwd_weights_workspace_size));
        DEBUG("Init bwd_weights " << *this << " req workspace: " << bwd_weights_workspace_size);

        DevBuffer& buffer = WorkSpace::get(bwd_weights_workspace_size);

        // find best algo, and benchmark!
        hipdnnConvolutionBwdFilterAlgoPerf_t perfs[5];
        int returned_algos;
        CHECK_HIPDNN(hipdnnFindConvolutionBackwardFilterAlgorithmEx(hipdnn::handle(), input.desc, input.data, doutput.desc, doutput.data, this->desc, dweights.desc, dweights.data, 5, &returned_algos, perfs, buffer.data, bwd_weights_workspace_size));

        INFO("\tHipDNN Found " << returned_algos << " bwd_weights algorithms, choosing " << perfs[0].algo << ": ");
        for (int i = 0; i < returned_algos; ++i) {
            INFO("\t\t" << i << ") " << perfs[i].algo << " - time: " << perfs[i].time << ", Memory: " << perfs[i].memory);
        }

        bwd_weights_algo = perfs[0].algo;
    }

    void init_backward(const Tensor& doutput, Tensor& dinput) override {
        find_bwd_data_algo(doutput, dinput);
        find_bwd_weights_algo(doutput, dinput);
    }

    void forward(const Tensor& input, Tensor& output) override {
        float alpha = 1.f;
        float beta = 0.f;
        DevBuffer& buffer = WorkSpace::get();
        CHECK_HIPDNN(hipdnnConvolutionForward(hipdnn::handle(), &alpha, input.desc, input.data, weights.desc, weights.data, this->desc, fwd_algo, buffer.data, buffer.size, &beta, output.desc, output.data));
        // save for backward
        input_ref = &input;
    }

    void backward(const Tensor& doutput, Tensor& dinput) override {
        float alpha = 1.f;
        float beta = 0.f;
        DevBuffer& buffer = WorkSpace::get();
        CHECK_HIPDNN(hipdnnConvolutionBackwardData(hipdnn::handle(), &alpha, weights.desc, weights.data, doutput.desc, doutput.data, this->desc, bwd_data_algo,  buffer.data, buffer.size, &beta, dinput.desc, dinput.data));
        CHECK_HIPDNN(hipdnnConvolutionBackwardFilter(hipdnn::handle(), &alpha,  input_ref->desc, input_ref->data, doutput.desc, doutput.data, this->desc, bwd_weights_algo,  buffer.data, buffer.size, &beta, dweights.desc, dweights.data));
    }
};


struct PoolingLayer : public Layer {
	hipdnnPoolingMode_t pool_mode;
	hipdnnPoolingDescriptor_t desc;

    // needed for backward: original input, original output, indeces (as workspace)
    DevBuffer indeces_buf;

    const Tensor* input;
    const Tensor* output;

    int kernel_size, padding, stride;

    static Dim getOutputDim(const TensorDesc& input, int kernel_size, int padding, int stride, hipdnnPoolingMode_t pool_mode) {
        int n, c, h, w;

        hipdnnPoolingDescriptor_t pool_desc;
        hipdnnNanPropagation_t maxpoolingNanOpt; // Dummy variable and doesn't affect miopenSetPooling2Ddescriptor
        CHECK_HIPDNN(hipdnnCreatePoolingDescriptor(&pool_desc));
        CHECK_HIPDNN(hipdnnSetPooling2dDescriptor(pool_desc, pool_mode, maxpoolingNanOpt, kernel_size, kernel_size, padding, padding, stride, stride));
        CHECK_HIPDNN(hipdnnGetPooling2dForwardOutputDim(pool_desc, input.desc, &n, &c, &h, &w));
        CHECK_HIPDNN(hipdnnDestroyPoolingDescriptor(pool_desc));
        return Dim(n, c, h, w);
    }

    virtual std::ostream& write_name(std::ostream& os) const override {
        if (pool_mode == HIPDNN_POOLING_MAX)
            os << "MaxPool(";
        else
            os << "AvgPool(";
        return os << kernel_size << "x" << kernel_size << ")";
    }

    PoolingLayer(const TensorDesc& input_dim, int kernel_size, int padding, int stride, hipdnnPoolingMode_t pool_mode)
        : Layer((Dim&)input_dim, PoolingLayer::getOutputDim(input_dim, kernel_size, padding, stride, pool_mode)),
          pool_mode(pool_mode),
          kernel_size(kernel_size), padding(padding), stride(stride) {
        CHECK_HIPDNN(hipdnnCreatePoolingDescriptor(&desc));
        hipdnnNanPropagation_t maxpoolingNanOpt; // Dummy variable and doesn't affect miopenSetPooling2Ddescriptor
        CHECK_HIPDNN(hipdnnSetPooling2dDescriptor(desc, pool_mode, maxpoolingNanOpt, kernel_size, kernel_size, padding, padding, stride, stride));
    }

    ~PoolingLayer() {
        CHECK_HIPDNN(hipdnnDestroyPoolingDescriptor(desc));
    }

    virtual void init_forward(const Tensor&, Tensor&) override {
        size_t size;
        //TODO: miopen to hip conversion of below PRNSOS
        CHECK_HIPDNN(miopenPoolingGetWorkSpaceSize(output_desc.desc, &size));
        indeces_buf = DevBuffer(size);
    }

    virtual void forward(const Tensor& input, Tensor& output) override {
        float alpha = 1.f;
        float beta = 0.f;
        CHECK_HIPDNN(hipdnnPoolingForward(hipdnn::handle(), desc, &alpha, input.desc, input.data, &beta, output.desc, output.data));
        // save for backward
        this->input = &input;
        this->output = &output;
    }

    virtual void backward(const Tensor& doutput, Tensor& dinput) override {
        float alpha = 1.f;
        float beta = 0.f;
        CHECK_HIPDNN(hipdnnPoolingBackward(hipdnn::handle(), desc, &alpha, getOutputDesc().desc, output->data, doutput.desc, doutput.data, getInputDesc().desc, input->data, &beta, dinput.desc, dinput.data));
    }
};

struct MaxPool : public PoolingLayer {
    MaxPool(const TensorDesc& input_dim, int kernel_size, int padding, int stride)
        : PoolingLayer(input_dim, kernel_size, padding, stride, HIPDNN_POOLING_MAX) {}
};

struct AvgPool : public PoolingLayer {
    AvgPool(const TensorDesc& input_dim, int kernel_size, int padding, int stride)
        : PoolingLayer(input_dim, kernel_size, padding, stride, HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING) {}
};

struct ReLU : public Layer {
	hipdnnActivationDescriptor_t desc;

    const Tensor* input_ref;
    const Tensor* output_ref;


    virtual std::ostream& write_name(std::ostream& os) const {
        return os << "ReLU()";
    }

    ReLU(const TensorDesc& input_dim) : Layer(input_dim, input_dim) {
        CHECK_HIPDNN(hipdnnCreateActivationDescriptor(&desc));
        hipdnnNanPropagation_t reluNanOpt; // dummy Variable for hipdnn sake PRNSOS
        CHECK_HIPDNN(hipdnnSetActivationDescriptor(desc, HIPDNN_ACTIVATION_RELU, reluNanOpt,  0.0, 0.0, 1.0));
    }


    ~ReLU() {
        CHECK_HIPDNN(hipdnnDestroyActivationDescriptor(desc));
    }

    void forward(const Tensor& input, Tensor& output) {
        float alpha = 1.f;
        float beta = 0.f;
        CHECK_HIPDNN(hipdnnActivationForward(hipdnn::handle(), desc, &alpha, input.desc, input.data, &beta, output.desc, output.data));
        // save for backward
        this->input_ref = &input;
        this->output_ref = &output;
    }

    void backward(const Tensor& doutput, Tensor& dinput) {
        float alpha = 1.f;
        float beta = 0.f;
        CHECK_HIPDNN(hipdnnActivationBackward(hipdnn::handle(), desc, &alpha, output_ref->desc, output_ref->data, doutput.desc, doutput.data, input_ref->desc, input_ref->data, &beta, dinput.desc, dinput.data));
    }
};


void mm_blas(const Tensor& A, bool transA, const Tensor& B, bool transB, Tensor& C) {
    assert(A.h == 1 && A.w == 1);
    assert(B.h == 1 && B.w == 1);
    assert(C.h == 1 && C.w == 1);

    int M = transA ? A.c : A.n;
    int K = transA ? A.n : A.c;
    int N = transB ? B.n : B.c;
    assert(transB ? K == B.c : K == B.n);
    assert(C.n == M && C.c == N);

    float alpha = 1.f;
    float beta = 0.f;
    int lda = A.c;
    int ldb = B.c;
    int ldc = C.c;
    hipblasHandle_t blas_handle;
    hipblasCreate(&blas_handle);
    hipblasOperation_t opA = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t opB = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    // call Sgemm with A<->B swapped (since we have rowmaj, but blas expects colmajor)
    hipblasStatus_t err = hipblasSgemm(blas_handle, opB, opA, N, M, K, &alpha, (const float*)B.data, ldb, (const float*)A.data, lda, &beta, (float*)C.data, ldc);
    assert(err == 0);
}

// (batch_size * size) -> (batch_size * size)
struct Linear : public Layer {
    int batch_size;
    int in_size;
    int out_size;

    Tensor weights; // dim (out_channels, in_channels, 1, 1)
    Tensor dweights;

    const Tensor* input_ref;

    virtual std::ostream& write_name(std::ostream& os) const {
        return os << "Linear(" << in_size << "," << out_size << ")";
    }

    Linear(const TensorDesc& input_dim, int out_size)
        : Layer(input_dim, TensorDesc(input_dim.n, out_size, 1, 1)),
          batch_size(input_dim.n),
          in_size(input_dim.c * input_dim.h * input_dim.w),
          out_size(out_size),
          weights(out_size, in_size, 1, 1),
          dweights(out_size, in_size, 1, 1)
    {
    }

    void forward(const Tensor& input, Tensor& output) {
        assert(batch_size == input.n);
        assert(batch_size == output.n);
        assert(out_size = output.c);
        assert(in_size == input.c * input.h * input.w);
        mm_blas(input, false, weights, true, output); // O <- I * W^T
        input_ref = &input;
    }

    void backward(const Tensor& doutput, Tensor& dinput) {
        // two MMs
        mm_blas(doutput, true, *input_ref, false, dweights); // dW <- dO^T * I
        mm_blas(doutput, false, weights, false, dinput); // dI <- dO * W
    }
};


struct BatchNorm : public Layer {
    // size of internal tensors (spatial: 1C11, per activation: 1CHW)
	hipdnnBatchNormMode_t bn_mode;
    TensorDesc bn_dim;

    Tensor scale;
    Tensor dscale;
    Tensor bias;
    Tensor dbias;
    double exp;
    Tensor running_mean;
    Tensor running_var;
    double epsilon;
    Tensor saved_mean; // saved mean for backward
    Tensor saved_ivar; // saved inverse variance for backward

    const Tensor* input_ref; // save reference to input for backward pass

    static TensorDesc get_bn_dim(const TensorDesc& input_dim, hipdnnBatchNormMode_t bn_mode) {
        TensorDesc bn(0,0,0,0);
        CHECK_HIPDNN(hipdnnDeriveBNTensorDescriptor(bn.desc, input_dim.desc, bn_mode));
        bn.update_get();
        return bn;
    }

    BatchNorm(const TensorDesc& input_dim, hipdnnBatchNormMode_t bn_mode=HIPDNN_BATCHNORM_SPATIAL, double eps = 1e-05, double momentum = 0.1)
        : Layer(input_dim, input_dim),
          bn_mode(bn_mode),
          bn_dim(get_bn_dim(input_dim, bn_mode)),
          scale(bn_dim),
          dscale(bn_dim),
          bias(bn_dim),
          dbias(bn_dim),
          exp(momentum),
          running_mean(bn_dim),
          running_var(bn_dim),
          epsilon(eps),
          saved_mean(bn_dim),
          saved_ivar(bn_dim)
    {
    }

    virtual std::ostream& write_name(std::ostream& os) const {
        return os << "BatchNorm()";
    }

    void forward(const Tensor& input, Tensor& output) {
        float alpha = 1.f;
        float beta = 0.f;
        CHECK_HIPDNN(hipdnnBatchNormalizationForwardTraining(hipdnn::handle(),
                 bn_mode,
                 &alpha,
                 &beta,
                 input.desc,
                 input.data,
                 output.desc,
                 output.data,
                 bn_dim.desc,
                 scale.data,
                 bias.data,
                 exp,
                 running_mean.data,
                 running_var.data,
                 epsilon,
                 saved_mean.data,
                 saved_ivar.data));
        input_ref = &input;
    }

    void backward(const Tensor& doutput, Tensor& dinput) {
        float alpha = 1.f;
        float beta = 0.f;
        CHECK_HIPDNN(hipdnnBatchNormalizationBackward(hipdnn::handle(),
                     bn_mode,
                     &alpha, 
                     &beta,
                     &alpha,
                     &beta,
                     input_ref->desc,
                     input_ref->data,
                     doutput.desc,
                     doutput.data,
                     dinput.desc,
                     dinput.data,
                     bn_dim.desc,
                     scale.data,
                     dscale.data,
                     dbias.data,
                     epsilon,
                     saved_mean.data,
                     saved_ivar.data));
    }
};

struct Reshape : public Layer {

    Reshape(const TensorDesc& input_dim, int n, int c, int h, int w)
        : Layer(input_dim, TensorDesc(n, c, h, w)) {
        assert(input_dim.n == n);
        assert(input_dim.c * input_dim.h * input_dim.w == c*h*w);
    }

    void init_forward(const Tensor& input, Tensor& output) override {
        output = input.viewAs(getOutputDesc());
    }

    void forward(const Tensor& input, Tensor& output) override {
        output = input.viewAs(getOutputDesc());
    }

    void init_backward(const Tensor& doutput, Tensor& dinput) override {
        dinput = doutput.viewAs(getInputDesc());
    }

    void backward(const Tensor& doutput, Tensor& dinput) override {
        dinput = doutput.viewAs(getInputDesc());
    }
};



#endif // LAYERS_HPP
